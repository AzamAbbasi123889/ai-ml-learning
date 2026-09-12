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
