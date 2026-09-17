#!/usr/bin/env python3
"""
main.py
Minimal, runnable example that fine-tunes a sentence-transformers model on a tiny triplet set.
Fixes from reviewer:
 - Uses built-in TripletEvaluator and passes it to model.fit
 - Uses batched encoding internally where needed
 - Avoids comparing tensors in Python conditionals
 - Clarifies DataLoader usage
 - Sets random seeds for reproducibility
"""
import random
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Simple synthetic dataset of triplets (anchor, positive, negative)
triplets = [
    ("A man is playing a guitar", "A person plays the guitar", "A dog is barking"),
    ("A woman is cooking in a kitchen", "Someone makes food in a kitchen", "A car is driving"),
    ("Children play football in the park", "Kids are playing soccer outside", "The stock market fell"),
    ("A cat sleeps on the sofa", "A feline rests on the couch", "Astronauts walk in space"),
]

# Convert to InputExample list (sentence-transformers friendly)
train_examples = [InputExample(texts=[a, p]) for (a, p, n) in triplets]
# For TripletEvaluator we need anchors/positives/negatives lists
anchors = [a for (a, p, n) in triplets]
positives = [p for (a, p, n) in triplets]
negatives = [n for (a, p, n) in triplets]

# Model and training setup
model_name = "all-MiniLM-L6-v2"  # lightweight model for example
model = SentenceTransformer(model_name)

# DataLoader: SentenceTransformer.fit accepts a DataLoader that yields lists of InputExample.
# For small datasets we can use a plain torch DataLoader over the InputExample list.
# For large datasets consider sentence_transformers.datasets.SentenceDataset or huggingface datasets pipelines.
batch_size = 4
train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)

# Loss: MultipleNegativesRankingLoss benefits from larger batch_size because negatives come from other
# examples in the same batch (in-batch negatives). If GPU memory is constrained, use gradient accumulation
# or MegaBatching strategies (not shown here).
train_loss = losses.MultipleNegativesRankingLoss(model)

# Use the built-in TripletEvaluator for robust metric logging and integration with model.fit
triplet_evaluator = evaluation.TripletEvaluator(anchors=anchors, positives=positives, negatives=negatives, name="toy-triplets")

# Train for a very small number of epochs for the demo
num_epochs = 1
warmup_steps = 0
output_path = "./output_toy"
if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Starting training (toy example)...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=triplet_evaluator,            # pass evaluator for validation/checkpointing
    epochs=num_epochs,
    evaluation_steps=50,                    # evaluate during training (none for tiny dataset)
    output_path=output_path,
    warmup_steps=warmup_steps,
)

# After training, run the evaluator manually to get the metric
results = triplet_evaluator(model)
print("Triplet evaluator results:", results)

# Demonstration: if you ever write a simple custom evaluator, remember:
# - Encode sentences in batches, not one-by-one
# - Convert 0-d tensors to Python scalars with .item() before using in 'if' checks
# Example (batch encode):
sample_sentences = ["A man is playing a guitar", "A dog is barking"]
# Efficient batched encoding
embeddings = model.encode(sample_sentences, convert_to_tensor=True, batch_size=8)
print("Embeddings shape:", embeddings.shape)
