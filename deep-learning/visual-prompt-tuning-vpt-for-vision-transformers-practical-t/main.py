import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as D
import timm
from tqdm import tqdm

# Simple Visual Prompt Tuning (VPT-shallow) wrapper for timm ViT models
class PromptedViT(nn.Module):
    def __init__(self, backbone_name='vit_base_patch16_224', prompt_len=20, num_classes=10, pretrained=True):
        super().__init__()
        # Load pretrained ViT from timm
        self.vit = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
        # We expect a ViT with a patch embedding and a cls_token
        # Freeze backbone parameters
        for p in self.vit.parameters():
            p.requires_grad = False

        embed_dim = self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else self.vit.num_features

        # Prompt tokens: learnable vectors inserted into input sequence
        self.prompt_len = prompt_len
        self.prompt_tokens = nn.Parameter(torch.randn(1, prompt_len, embed_dim) * 0.02)

        # Optional small trainable layernorm on prompts (helps stability)
        self.prompt_ln = nn.LayerNorm(embed_dim)

        # Classification head: map transformer output (cls token) to num_classes
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize head
        nn.init.normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # Use timm ViT's patch embedding & cls token handling
        # For timm ViT: patch_embed -> returns x of shape (B, N, C), then add class token
        # We'll replicate the forward steps but keep the backbone frozen.

        B = x.shape[0]

        # 1) patch embedding (this returns sequence without class token for some timm variants)
        if hasattr(self.vit, 'patch_embed'):
            x = self.vit.patch_embed(x)  # (B, N, C)
        else:
            raise RuntimeError('Unexpected ViT structure: patch_embed not found')

        # 2) class token and position embeddings
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, 1+N, C)

        # 3) prepend prompt tokens (repeat for batch)
        prompts = self.prompt_tokens.expand(B, -1, -1)  # (B, P, C)
        prompts = self.prompt_ln(prompts)

        x = torch.cat((prompts, x), dim=1)  # (B, P+1+N, C)

        # 4) add position embeddings: timm ViT has pos_embed for original length; we need to adapt
        # The timm pos_embed expects (1, 1+N, C). We'll concatenate prompt position embeddings by
        # extending pos_embed with zeros for prompts (simple and effective).
        if hasattr(self.vit, 'pos_embed'):
            pos = self.vit.pos_embed  # (1, 1+N, C)
            # Create prompt pos embedding zeros
            prompt_pos = torch.zeros(1, self.prompt_len, pos.size(-1), device=pos.device, dtype=pos.dtype)
            full_pos = torch.cat((prompt_pos, pos), dim=1)  # (1, P+1+N, C)
            x = x + full_pos
        else:
            # If no pos_embed (unlikely), skip
            pass

        # 5) forward through transformer encoder layers
        # timm ViT's forward_features will normally do patch->cls->transformer->norm
        # We'll run the transformer blocks manually, but reuse the vit.blocks and vit.norm.
        for blk in self.vit.blocks:
            x = blk(x)

        if hasattr(self.vit, 'norm') and self.vit.norm is not None:
            x = self.vit.norm(x)

        # 6) extract cls token which is located after prompts: prompts_len + 0 index
        cls_index = self.prompt_len  # because prompts are prepended, class token is at this index
        cls = x[:, cls_index, :]

        out = self.head(cls)
        return out


def get_data(batch_size=64):
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train = D.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = D.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc='train', leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total += yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': f'{100*correct/total:.2f}%'})


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
    return correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Config
    backbone = 'vit_base_patch16_224'
    prompt_len = 30
    epochs = 5
    batch_size = 64
    lr = 1e-3

    train_loader, test_loader = get_data(batch_size=batch_size)

    model = PromptedViT(backbone_name=backbone, prompt_len=prompt_len, num_classes=10, pretrained=True)
    model.to(device)

    # Only prompt tokens, prompt layernorm and head parameters should be trainable
    trainable = [p for p in model.parameters() if p.requires_grad]
    print('Trainable params:', sum(p.numel() for p in trainable))

    opt = optim.Adam(trainable, lr=lr, weight_decay=1e-4)

    for ep in range(epochs):
        print(f'Epoch {ep+1}/{epochs}')
        train_one_epoch(model, train_loader, opt, device)
        acc = evaluate(model, test_loader, device)
        print(f'Validation accuracy after epoch {ep+1}: {acc*100:.2f}%')


if __name__ == '__main__':
    main()
