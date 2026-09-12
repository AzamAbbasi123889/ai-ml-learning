# Practical Multimodal Image-Text Search with Sentence-Transformers (clip-ViT-B-32) and FAISS

**Category:** daily

## Overview

A compact, practical tutorial and small Python project showing how to build a local multimodal image-text search system. We use the sentence-transformers CLIP model (clip-ViT-B-32) to embed images and text into the same vector space, index image embeddings with FAISS, and run fast nearest-neighbor retrieval for text queries. The resource covers data preparation, embedding, indexing, search, and evaluation tips.

## Problem

Many applications (image search, zero-shot classification, image deduplication) require mapping images and text to a shared embedding space so text queries can retrieve relevant images. Practitioners need a small, reproducible codebase that demonstrates how to: 1) embed images and text with a multimodal model, 2) build a FAISS index for fast retrieval, and 3) run and evaluate text-to-image search locally.

## Technical Explanation

CLIP-based Sentence-Transformers models (e.g., clip-ViT-B-32) map images and text to the same vector space using dual encoders trained with contrastive objectives. This enables computing cosine similarity between image and text embeddings. For scalable nearest-neighbor retrieval we use FAISS (Facebook AI Similarity Search). Steps: 1) Load the multimodal SentenceTransformer that supports images and text. 2) Encode images (PIL Image objects) to fixed-size vectors. 3) Build a FAISS index (Normalized embeddings -> inner product on L2-normalized vectors equals cosine similarity). 4) Encode text queries and normalize vectors consistently. 5) Query FAISS for top-k nearest images and present results. Important practical details: install sentence-transformers with image support (or the base package with torchvision/transforms available), batch encode to avoid OOM, normalize embeddings for cosine similarity, persist index and metadata (filenames) for reuse, and handle different image sizes using PIL load. Model and usage examples are documented by Sentence-Transformers and the model is available on Hugging Face.

## Key Concepts

- CLIP and multimodal embeddings
- Sentence-Transformers (clip-ViT-B-32)
- Image and text encoding into shared vector space
- FAISS for approximate nearest neighbor search
- Cosine similarity via normalized embeddings
- Batching and GPU vs CPU tradeoffs
- Index persistence and metadata mapping

## Practical Example

A runnable Python script (main.py) is included. It scans an images/ folder, computes image embeddings with sentence-transformers/clip-ViT-B-32, builds a FAISS index (IndexFlatIP on normalized vectors), saves the index and metadata, and accepts text queries to retrieve top-k similar images. The example is self-contained and uses standard Python libraries; swap the model string to other supported models if desired.

## Python Implementation

```python
"""
main.py
Practical multimodal text-to-image search using sentence-transformers (clip-ViT-B-32) and faiss-cpu.
Usage:
  1) Place images in ./images/ (create the folder and add .jpg/.png files)
  2) python main.py --index   # creates embeddings/index
  3) python main.py --query "a dog playing in snow" --top_k 5

Produces: prints top-k filenames and similarity scores.
"""
import os
import argparse
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer, util
import faiss

IMAGES_DIR = Path("images")
INDEX_DIR = Path("index_data")
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"


def load_image_paths(images_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    return sorted(paths)


def encode_images(model, image_paths, batch_size=16):
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(img)
            except Exception as e:
                print(f"Skipping {p}: {e}")
        if not imgs:
            continue
        emb = model.encode(imgs, convert_to_numpy=True, batch_size=len(imgs), show_progress_bar=False)
        embeddings.append(emb)
    if embeddings:
        return np.vstack(embeddings)
    else:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)


def build_faiss_index(embeddings, index_path):
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized vectors = cosine similarity
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    return index


def save_metadata(paths, meta_path):
    with open(meta_path, "wb") as f:
        pickle.dump([str(p) for p in paths], f)


def load_index_and_meta(index_path, meta_path):
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        paths = pickle.load(f)
    return index, paths


def query_text(model, index, paths, text, top_k=5):
    text_emb = model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(text_emb)
    D, I = index.search(text_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(paths):
            continue
        results.append((paths[idx], float(score)))
    return results


def ensure_dirs():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Create embeddings and build index from images/")
    parser.add_argument("--query", type=str, help="Text query to search for images")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    ensure_dirs()

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded. Modalities:", getattr(model, "modalities", None))

    index_path = INDEX_DIR / "images.faiss"
    meta_path = INDEX_DIR / "metadata.pkl"

    if args.index:
        image_paths = load_image_paths(IMAGES_DIR)
        if not image_paths:
            print(f"No images found in {IMAGES_DIR.resolve()}. Add images and rerun.")
            return
        print(f"Found {len(image_paths)} images. Encoding in batches of {args.batch_size}...")
        embeddings = encode_images(model, image_paths, batch_size=args.batch_size).astype(np.float32)
        print("Embeddings shape:", embeddings.shape)
        print("Building FAISS index...")
        build_faiss_index(embeddings, index_path)
        save_metadata(image_paths, meta_path)
        print(f"Index and metadata saved to {INDEX_DIR.resolve()}")
        return

    if args.query:
        if not index_path.exists() or not meta_path.exists():
            print("Index not found. Run with --index first to create index from images/")
            return
        index, paths = load_index_and_meta(index_path, meta_path)
        results = query_text(model, index, paths, args.query, top_k=args.top_k)
        print(f"Top {len(results)} results for query: '{args.query}'")
        for i, (p, score) in enumerate(results, 1):
            print(f"{i}. {p}  (score={score:.4f})")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

```

## Code Explanation

main.py is a minimal, self-contained CLI tool. --index scans ./images/ for common image files, batch-encodes them with the SentenceTransformer CLIP model (convert_to_numpy=True), normalizes embeddings, builds a FAISS IndexFlatIP index, and saves the index and a metadata mapping of image filenames. --query encodes a single text query, normalizes it, queries FAISS for the top-k nearest image embeddings and prints filenames with similarity scores. The script normalizes both image and text embeddings so inner product equals cosine similarity. Adjust batch_size and model name as needed; for large collections consider FAISS IVF or HNSW indexes and storing embeddings on disk.

## Real-World Applications

- Image search and retrieval for photo libraries or e-commerce catalogs
- Zero-shot image classification and filtering
- Image deduplication and clustering
- Multimodal search in digital asset management (DAM) systems
- Prototype multimodal features for chatbots and assistants that reference images

## Limitations

- clip-ViT-B-32 is a relatively small CLIP-style model — for higher accuracy consider larger VLMs (e.g., Qwen3-VL variants) but expect higher resource needs
- FAISS IndexFlatIP is exact nearest neighbor and does not scale optimally to millions of vectors — use IVF/HNSW or on-disk indexes for larger datasets
- Image preprocessing and variant domains (medical, satellite) may require domain-specific fine-tuning
- Cosine similarity on global image embeddings may miss fine-grained object details; consider region-based or cross-encoder re-ranking for high-precision retrieval

## Further Learning

- https://sbert.net/examples/sentence_transformer/applications/image-search/README.html
- https://huggingface.co/sentence-transformers/clip-ViT-B-32
- https://www.faiss.ai/
- https://sbert.net/docs/sentence_transformer/pretrained_models.html

## Sources

- https://sbert.net/examples/sentence_transformer/applications/image-search/README.html
- https://sbert.net/docs/sentence_transformer/pretrained_models.html
- https://huggingface.co/sentence-transformers/clip-ViT-B-32
- https://www.faiss.ai/

