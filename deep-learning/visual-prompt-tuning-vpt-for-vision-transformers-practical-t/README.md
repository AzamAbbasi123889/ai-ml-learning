# Visual Prompt Tuning (VPT) for Vision Transformers — Practical Tutorial + Small PyTorch Project

**Category:** educational

## Overview

This resource explains Visual Prompt Tuning (VPT) — a parameter-efficient transfer learning method for Vision Transformers (ViT) that keeps the pretrained backbone frozen and learns a small set of input-space (prompt) tokens per task. It includes an accessible technical explanation, key concepts, limitations, and a compact runnable PyTorch example that trains only prompt tokens (plus a lightweight classification head) on CIFAR-10. The example demonstrates the core idea in a few dozen lines and is reproducible on a single GPU or CPU.

## Problem

Full fine-tuning of large pretrained vision transformers is parameter- and storage-heavy: each new task requires copying and updating the entire model. For practitioners who want efficient adaptation (low storage, fast iteration, and reduced compute/memory), we need methods that adapt small per-task parameters while leveraging a frozen pretrained backbone.

## Technical Explanation

Visual Prompt Tuning (VPT) places a small set of learnable tokens (visual prompts) into the input embedding space of a Vision Transformer. For patch-based ViT models the image is mapped to a sequence of patch embeddings; VPT prepends (or inserts) k prompt tokens to that sequence and only optimizes these prompt token embeddings (and optionally a small classification head) while keeping the ViT backbone frozen. This is analogous to textual prompt tuning in NLP. VPT variants: "VPT-shallow" inserts prompts only once at the input layer; "VPT-deep" inserts prompt tokens into multiple transformer layers. Advantages: (1) very few trainable parameters (often <1% of full model), (2) small per-task storage, (3) competitive performance when downstream data is limited. Practical considerations: choose prompt length k (e.g., 10–100), learning rate (prompts often require higher LR than head), whether to train layernorms or head, input resizing and normalization consistent with the pretrained model, and whether to use shallow vs deep prompts depending on task complexity and compute.

## Key Concepts

- Prompt tokens: learnable vector embeddings prepended to ViT patch embeddings
- Parameter-efficient tuning: only prompts (+small head) are updated
- VPT-shallow vs VPT-deep: single insertion vs multiple layer insertions
- Compatibility: work with pretrained ViT backbones (e.g., timm models)
- Hyperparameters: prompt length, learning rate, weight decay, insertion layers
- Trade-offs: storage and compute vs possible performance gap on large datasets

## Practical Example

A complete PyTorch example (main.py) that: (1) loads a pretrained ViT from timm; (2) freezes backbone weights; (3) introduces a small number of prompt tokens that are prepended to patch embeddings; (4) adds a linear classification head; (5) trains only the prompt tokens and the head on CIFAR-10 (resized to 224×224). The script includes data loading, training loop, evaluation, and comments explaining where VPT is applied. This is a compact, single-file runnable project intended for experimentation and learning.

## Python Implementation

```python
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

```

## Code Explanation

This single-file example implements VPT-shallow. PromptedViT wraps a timm ViT backbone, freezes backbone parameters, and adds prompt tokens (self.prompt_tokens). During forward, it obtains patch embeddings from the backbone's patch_embed, constructs the sequence CLS token + patches, prepends prompts, adds position embeddings (we extend the backbone pos_embed with zeros for prompt positions), runs transformer blocks, applies the backbone norm, and reads the CLS embedding at index prompt_len (since prompts were prepended). A lightweight linear head maps the CLS embedding to class logits. The training loop updates only the prompt tokens, prompt layernorm, and the head. The model is trained on CIFAR-10 resized to 224x224 for a few epochs to allow quick experiments.

## Real-World Applications

- Fast adaptation of large ViT backbones to new classification tasks with limited storage per task (few‑shot and low-data regimes)
- On-device personalization: store small prompt vectors per user/task instead of full model copies
- Multi-task systems where many tasks share a frozen backbone but require different per-task parameters
- Rapid prototyping: quickly test downstream tasks without full fine-tuning expense

## Limitations

- VPT may underperform full fine-tuning on large-scale downstream datasets where full model capacity is necessary
- Prompt tokens are sensitive to prompt length and learning rate; tuning is required
- This simplified example uses a heuristic (zero prompt position embeddings). For best results follow careful alignment or learn prompt positions
- The example uses timm internals and may need minor adjustments for different ViT implementations or timm versions

## Further Learning

- https://arxiv.org/abs/2203.12119
- https://github.com/KMnP/vpt
- https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930696.pdf
- https://github.com/czczup/ViT-Adapter
- https://github.com/ChengHan111/E2VPT
- https://github.com/huggingface/transformers

## Sources

- https://arxiv.org/abs/2203.12119
- https://github.com/KMnP/vpt
- https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930696.pdf
- https://github.com/ChengHan111/E2VPT
- https://timm.fast.ai/

