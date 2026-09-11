import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# === Config ===
model_name = "gpt2"  # small GPT-2; works without bitsandbytes
adapter_save_path = "./lora_adapter"
use_cuda = torch.cuda.is_available()

# Guarded fp16: only enable if CUDA is present (user can further refine checks)
fp16 = use_cuda

# === Tokenizer & Model ===
# Add pad token if missing and resize embeddings after loading model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# Load base model (not quantized) for this minimal example
model = AutoModelForCausalLM.from_pretrained(model_name)
# Resize token embeddings after adding tokens to tokenizer
model.resize_token_embeddings(len(tokenizer))

# === Tiny dataset (for demo/training) ===
texts = [
    "Hello, my name is Alice and I like machine learning.",
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are powerful for sequence modeling.",
]
raw = {"text": texts}

# Tokenize
enc = tokenizer(texts, padding=True, truncation=True, return_tensors=None)
# Create Hugging Face Dataset
ds = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
# For causal LM, labels = input_ids. Use batched mapping for performance on larger datasets.
def add_labels(batch):
    batch["labels"] = batch["input_ids"]
    return batch

ds = ds.map(add_labels, batched=True)

# === Inspect model modules to choose target_modules conservatively ===
# Different model families name attention/proj layers differently.
# Use this helper to see which module names match a given heuristic.
conservative_targets = ["c_attn", "c_proj"]
print("== Module name matches for conservative targets ==")
for name, module in model.named_modules():
    if any(t in name for t in conservative_targets):
        print(name)

# If you have a different model (e.g., LLaMA/Falcon), inspect the names and adapt.

# === PEFT LoRA setup ===
# Use a conservative default for target_modules. Exact names vary by model.
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=conservative_targets,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # sanity check

# === Data collator ===
# For causal LM use DataCollatorForLanguageModeling with mlm=False so labels are preserved for shift
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === TrainingArguments & Trainer ===
training_args = TrainingArguments(
    output_dir="./lora_out",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="no",  # we save adapter separately below
    fp16=fp16,
    remove_unused_columns=False,  # keep labels/attention_mask
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

# === Train (this will only run a single epoch on tiny data) ===
trainer.train()

# === Save adapter weights only ===
# If `model` is a PeftModel/get_peft_model wrapper, save_pretrained writes adapter weights
os.makedirs(adapter_save_path, exist_ok=True)
model.save_pretrained(adapter_save_path)
print(f"Saved LoRA adapter to {adapter_save_path}")

# === Example: load base model and apply adapter for inference ===
# Loading workflow: load the original base model, then wrap with PeftModel.from_pretrained
base = AutoModelForCausalLM.from_pretrained(model_name)
base.resize_token_embeddings(len(tokenizer))
peft_loaded = PeftModel.from_pretrained(base, adapter_save_path)
peft_loaded.eval()

prompt = "Write a short sentence about transformers:"
inputs = tokenizer(prompt, return_tensors="pt")
if use_cuda:
    peft_loaded.to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    out = peft_loaded.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(out[0], skip_special_tokens=True))
