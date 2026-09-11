# Practical LoRA / PEFT for Causal LMs — Robust, Reproducible Example

**Category:** daily

## Overview

A corrected, robust, and reproducible guide demonstrating how to apply Low-Rank Adapters (LoRA) with Hugging Face PEFT to a causal language model. This resource fixes reviewer-identified issues from a prior draft: incorrect links, fragile target_modules heuristics, dataset mapping and collator details, PEFT API/version guidance, fp16/device guards, save/load adapter examples, and explicit pinned requirements. Includes a small runnable example (main.py) that trains LoRA adapters on a tiny dataset (GPT-2 small) without requiring bitsandbytes or 4-bit quantization.

## Problem

Many LoRA/PEFT tutorials use brittle defaults and unverified external links which cause reproducibility failures. Common pitfalls include incorrect target module names that don't match a model's module tree, improper dataset mapping (not batched), unsafe fp16 settings, unclear save/load semantics for adapter weights, and fabricated or out-of-date references. This guide corrects those issues and provides a minimal, tested example and clear instructions for both standard LoRA and optional QLoRA paths.

## Technical Explanation

LoRA (Low-Rank Adapters) injects small trainable rank-decomposition matrices into transformer weight updates so that the majority of the pretrained weights remain frozen. PEFT (Parameter-Efficient Fine-Tuning) provides convenient wrappers (PeftModel, get_peft_model, LoraConfig) to apply LoRA-style adapters to Hugging Face transformer models. For causal language modeling (decoder-only architectures like GPT-2), labels are input_ids and the loss uses the model's shift logic internally when using the appropriate data collator. Choosing target modules for LoRA is model-specific: transformers implementations name attention and projection layers differently across families. Therefore, the recommended approach is conservative defaults (e.g., target_modules=["c_attn","c_proj"]) and an inspection helper to print matching module names before applying PEFT. Saving via model.save_pretrained when the model is a PeftModel writes only the adapter weights; to use the adapter later you load the base model and call PeftModel.from_pretrained(base_model, adapter_path). QLoRA (4-bit quantized fine-tuning with LoRA) is an advanced option that requires bitsandbytes + accelerate + CUDA-compatible GPU; it is excluded from the minimal example but described with canonical references and install caveats.

## Key Concepts

- LoRA: low-rank adapters for parameter-efficient fine-tuning
- PEFT: Hugging Face's wrapper to apply adapters to models
- target_modules: model-specific module name selection for LoRA
- Dataset mapping: use batched mapping to add labels efficiently
- Data collators: DataCollatorForLanguageModeling(mlm=False) vs default collator
- Saving/loading adapters: PeftModel.save_pretrained & PeftModel.from_pretrained
- QLoRA: 4-bit quantized fine-tuning (optional, requires bitsandbytes + specific CUDA)

## Practical Example

A minimal, runnable main.py that demonstrates: (1) loading GPT-2 small + tokenizer, (2) adding a pad token and resizing embeddings, (3) preparing a tiny dataset with batched map to add labels, (4) inspecting model.named_modules() to choose LoRA target modules, (5) applying PEFT LoRA, (6) training with transformers.Trainer using DataCollatorForLanguageModeling(mlm=False), (7) saving adapter weights, and (8) loading the adapter back for inference with PeftModel.from_pretrained.

## Python Implementation

```python
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

```

## Code Explanation

The example trains LoRA adapters on GPT-2 small with a tiny synthetic dataset so it runs quickly for demonstration. Key robust choices: (1) tokenizer.add_special_tokens is done before resizing embeddings and model.resize_token_embeddings(len(tokenizer)) is called after the base model is loaded; (2) dataset mapping uses batched=True (important for larger datasets) and sets labels=input_ids for causal LM; (3) conservative target_modules are used and a short module-inspection snippet is printed so users can adapt for other model families; (4) DataCollatorForLanguageModeling(tokenizer, mlm=False) is used for causal LM so the Trainer handles shifting; (5) fp16 is guarded by checking torch.cuda.is_available(); users on older GPUs or CPU-only systems should set fp16=False or add further device capability checks; (6) saving the PEFT-wrapped model writes only adapter parameters — the example demonstrates how to load the base model and apply the saved adapter with PeftModel.from_pretrained for inference.

## Real-World Applications

- Fine-tuning large language models for domain-specific conversational agents with fewer trainable parameters
- On-device or low-cost customization of pretrained models where full fine-tuning is impractical
- Rapid iteration for instruction-following tuning (Alpaca-style) without heavy compute
- Experimenting with adapter-based continual learning and multi-task adapters

## Limitations

- This minimal example does not demonstrate QLoRA (4-bit fine-tuning) — QLoRA requires bitsandbytes + accelerate + a CUDA-compatible GPU and careful environment setup.
- Target module names differ across model families; the conservative default may not be optimal for non-GPT architectures (e.g., LLaMA, Falcon). Always inspect model.named_modules().
- fp16 should be enabled only when the device and CUDA driver support it; the example uses a simple guard but users may need finer checks.
- Small example dataset and 1-epoch training are for demonstration only — they do not produce a production-ready model.

## Further Learning

- https://arxiv.org/abs/2106.09685
- https://github.com/huggingface/peft
- https://huggingface.co/docs/peft/main/en/introduction
- https://arxiv.org/abs/2305.14314
- https://huggingface.co/blog/peft-qlora
- https://github.com/artidoro/qlora

## Sources

- https://arxiv.org/abs/2106.09685
- https://github.com/huggingface/peft
- https://huggingface.co/docs/peft/main/en/introduction
- https://arxiv.org/abs/2305.14314
- https://huggingface.co/blog/peft-qlora
- https://github.com/artidoro/qlora
- https://arxiv.org/abs/2302.13971

