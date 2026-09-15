#!/usr/bin/env python3
"""
main.py
Minimal local semantic search demo using sentence-transformers + hnswlib.
Usage:
  - Prepare a plaintext file with one document per line, or run without args to use sample data.
  - Run: python main.py --build --data docs.txt --index_path my_index.bin --meta_path meta.json
  - Query: python main.py --query "your search text" --index_path my_index.bin --meta_path meta.json
The script shows building, saving, loading, and querying an HNSW index.
"""
import argparse
import json
import os
import sys
from typing import List

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

# Defaults
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = "hnsw_index.bin"
DEFAULT_META_PATH = "meta.json"

SAMPLE_DOCS = [
    "Install Python and pip, then run a simple script.",
    "How to bake sourdough bread: starter, feeding, and baking tips.",
    "Python data science libraries include numpy, pandas, scikit-learn and matplotlib.",
    "The Earth orbits the Sun with an approximately elliptical orbit.",
    "Machine learning pipelines often include preprocessing, training, and evaluation steps.",
]


def load_documents(path: str) -> List[str]:
    if not path:
        return SAMPLE_DOCS
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return docs or SAMPLE_DOCS


def build_index(docs: List[str], model_name: str, index_path: str, meta_path: str,
                space: str = "cosine", m: int = 16, ef_construction: int = 200):
    print(f"Loading model '{model_name}'... (this may download ~50-100MB)")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(docs)} documents...")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    num_elements = len(docs)

    # If using cosine, hnswlib supports 'cosine' directly. Optionally normalize when using l2.
    if space == "cosine":
        # hnswlib's cosine distance expects unnormalized vectors but many prefer normalization for clarity
        # We will L2-normalize here to ensure consistent cosine similarity behavior.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    print(f"Initializing hnswlib index (space={space}, dim={dim}) with M={m}, ef_construction={ef_construction}...")
    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=m)

    print("Adding items to index...")
    ids = np.arange(num_elements)
    p.add_items(embeddings, ids)

    # Save index and metadata
    print(f"Saving index to {index_path} and metadata to {meta_path}...")
    p.save_index(index_path)
    meta = {str(i): docs[i] for i in ids}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Index build complete.")


def load_index(index_path: str, meta_path: str):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    print(f"Loading index from {index_path}...")
    # We need dimension to construct the Index object before loading; hnswlib allows loading without init by creating a dummy index
    p = hnswlib.Index(space='cosine', dim=1)
    p.load_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return p, meta


def query_index(index: hnswlib.Index, meta: dict, query_text: str, model_name: str, top_k: int = 5, ef_search: int = 50):
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query_text], convert_to_numpy=True)
    # Normalize if index was built with normalized vectors
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    index.set_ef(ef_search)
    labels, distances = index.knn_query(q_emb, k=top_k)
    labels = labels[0]
    distances = distances[0]

    results = []
    for i, (lbl, dist) in enumerate(zip(labels, distances)):
        doc = meta.get(str(int(lbl)), "")
        # For cosine space with normalized vectors, hnswlib returns 1 - cosine_similarity as distance.
        # We convert back to a similarity score in [ -1, 1 ] approx via similarity = 1 - dist
        sim = None
        try:
            sim = 1.0 - float(dist)
        except Exception:
            sim = None
        results.append({
            "rank": i + 1,
            "id": int(lbl),
            "score": sim,
            "text": doc,
        })
    return results


def repl(index_path: str, meta_path: str, model_name: str):
    print("Loading index and metadata...")
    index, meta = load_index(index_path, meta_path)
    print("Ready. Type queries (empty line or Ctrl-C to exit).")
    while True:
        try:
            q = input("> ").strip()
            if not q:
                print("Exiting.")
                break
            results = query_index(index, meta, q, model_name)
            for r in results:
                print(f"[{r['rank']}] id={r['id']} score={r['score']:.4f}  text={r['text']}")
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Local semantic search demo using hnswlib + sentence-transformers")
    parser.add_argument("--build", action="store_true", help="Build index from data file")
    parser.add_argument("--data", type=str, default="", help="Path to newline-separated documents (one per line)")
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX_PATH, help="Path to save/load hnsw index")
    parser.add_argument("--meta_path", type=str, default=DEFAULT_META_PATH, help="Path to save/load metadata (json)")
    parser.add_argument("--query", type=str, default="", help="Run a single query against a saved index")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="SentenceTransformer model name")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.build:
        docs = load_documents(args.data)
        build_index(docs, args.model, args.index_path, args.meta_path)
        return

    if args.query:
        index, meta = load_index(args.index_path, args.meta_path)
        results = query_index(index, meta, args.query, args.model)
        for r in results:
            print(f"[{r['rank']}] id={r['id']} score={r['score']:.4f}  text={r['text']}")
        return

    # Default: interactive REPL
    if os.path.exists(args.index_path) and os.path.exists(args.meta_path):
        repl(args.index_path, args.meta_path, args.model)
    else:
        print("Index or metadata not found locally. Use --build to create an index from data, or provide --index_path and --meta_path.")


if __name__ == "__main__":
    main()
