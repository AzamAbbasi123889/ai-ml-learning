# Practical IVF‑PQ with FAISS: Build, Train, and Tune an IVFPQ Index for Scalable Semantic Search

**Category:** daily

## Overview

A focused, practical guide to using FAISS IndexIVFPQ (inverted file + product quantization) for memory‑ and speed‑efficient approximate nearest neighbor (ANN) search. Covers intuition, recommended parameter choices, training requirements, trade‑offs, and a small reproducible Python project that trains an IVFPQ index, compares latency/recall against an exact IndexFlat baseline, and demonstrates tuning nlist/nprobe/M/nbits.

## Problem

Large embedding collections (millions of vectors) make exact nearest neighbor search (IndexFlat) slow and memory‑heavy. Engineers need a reproducible workflow to build an ANN index that balances search latency, memory footprint, and recall. IndexIVFPQ is widely used but sensitive to hyperparameters (nlist, nprobe, M, nbits) and training set size; practitioners need concrete examples and sensible defaults.

## Technical Explanation

IndexIVFPQ combines two ideas: a coarse quantizer (IVF)—which partitions the vector space into nlist Voronoi cells—and product quantization (PQ), which compresses vectors in each cell by splitting vectors into M subspaces and encoding each subvector with 2^nbits centroids. Building an IVFPQ index requires: 1) choosing a quantizer (commonly IndexFlatL2) and nlist (number of coarse clusters), 2) training the IVF+PQ on representative vectors (recommended: max(1000*nlist, 2^nbits*1000) for IVF-based indexes), 3) adding vectors, and 4) setting nprobe at query time to control how many IVF lists are searched (trade‑off: higher nprobe => higher recall and latency). Key performance knobs: nlist (coarse granularity), nprobe (query-time lists), M (codebook count — affects compression and accuracy), nbits (bits per subquantizer — affects codebook size and memory). Typical patterns: larger nlist helps scale but needs more training and increases indexing time; M * nbits determines compressed code size per vector (e.g., M=16, nbits=8 => 16 bytes). After training, using a small exact re‑rank (e.g., retrieve top-K from IVFPQ then re‑score with exact distances) improves final recall. The tutorial demonstrates measurement of recall@k, index size on disk, and latency comparisons.

## Key Concepts

- FAISS Index types (Flat, IVF, PQ, IVFPQ)
- Product Quantization (PQ) — split vectors into M subspaces
- Inverted File (IVF) — coarse quantizer with nlist clusters
- nprobe — controls how many IVF lists are searched at query time
- Tradeoffs: memory vs recall vs latency
- Training data size recommendations for IVFPQ
- Practical tuning workflow and re‑ranking strategy

## Practical Example

The included Python project generates synthetic or user‑supplied embeddings (optionally uses sentence‑transformers to embed text), trains an IndexIVFPQ, adds vectors, and compares search latency and recall against IndexFlatL2. It shows how to vary nlist, nprobe, M, and nbits, measures index file size, and demonstrates a simple re‑ranking step to recover recall while keeping search fast.

## Python Implementation

```python
#!/usr/bin/env python3
"""
main.py
Practical IVFPQ demo: build and evaluate FAISS IndexIVFPQ vs IndexFlatL2.
Produces small synthetic dataset or uses sentence-transformers if available to embed sample texts.
"""
import time
import os
import argparse
import numpy as np

try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss is required. Install with `pip install faiss-cpu` (or faiss-gpu).")

# Optional: use sentence-transformers to produce realistic embeddings
USE_ST_MODEL = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    USE_ST_MODEL = False


def generate_embeddings(n, d, use_st=False):
    if use_st:
        # small sample sentences
        texts = [f"This is sample sentence number {i}." for i in range(n)]
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb = model.encode(texts, batch_size=64, show_progress_bar=False)
        return emb.astype('float32')
    else:
        rng = np.random.RandomState(123)
        return rng.randn(n, d).astype('float32')


def index_ivfpq(xb, d, nlist=128, M=8, nbits=8, nprobe=8):
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
    # Need to train on representative vectors
    train_samples = xb[: min(xb.shape[0], max(1000 * nlist, (2 ** nbits) * 1000))]
    print(f"Training IVFPQ on {train_samples.shape[0]} vectors (nlist={nlist}, M={M}, nbits={nbits})")
    index.train(train_samples)
    print("Adding vectors to IVFPQ index...")
    index.add(xb)
    index.nprobe = nprobe
    return index


def index_flat(xb, d):
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    return index


def evaluate(index, xq, xb_ids, k=10):
    t0 = time.time()
    D, I = index.search(xq, k)
    latency = (time.time() - t0) / xq.shape[0]
    return D, I, latency


def recall_at_k(I_true, I_pred, k=10):
    # I_true and I_pred: (nq, k)
    nq = I_true.shape[0]
    hits = 0
    for i in range(nq):
        set_true = set(I_true[i, :k])
        set_pred = set(I_pred[i, :k])
        hits += len(set_true & set_pred) / float(k)
    return hits / nq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000, help='number of base vectors')
    parser.add_argument('--d', type=int, default=128, help='dimension')
    parser.add_argument('--nq', type=int, default=100, help='number of queries')
    parser.add_argument('--use_st', action='store_true', help='use sentence-transformers for embeddings when possible')
    args = parser.parse_args()

    n = args.n
    d = args.d
    nq = args.nq
    use_st = args.use_st and USE_ST_MODEL

    print(f"Building dataset: n={n}, d={d}, nq={nq}, use_st_model={use_st}")
    xb = generate_embeddings(n, d, use_st=use_st)
    xq = generate_embeddings(nq, d, use_st=use_st)

    # baseline exact index
    print("\nBuilding exact IndexFlatL2 baseline...")
    flat = index_flat(xb, d)
    D_true, I_true, flat_latency = evaluate(flat, xq, np.arange(n), k=10)
    print(f"Flat: latency per query={flat_latency*1000:.3f} ms")

    # IVFPQ settings to try
    settings = [
        {'nlist': 64, 'M': 8, 'nbits':8, 'nprobe':4},
        {'nlist': 128, 'M': 8, 'nbits':8, 'nprobe':8},
        {'nlist': 256, 'M': 16, 'nbits':8, 'nprobe':16},
    ]

    for s in settings:
        print('\n--- IVFPQ setting: ' + str(s))
        ivfpq = index_ivfpq(xb, d, nlist=s['nlist'], M=s['M'], nbits=s['nbits'], nprobe=s['nprobe'])
        D_ivf, I_ivf, ivf_latency = evaluate(ivfpq, xq, np.arange(n), k=10)
        rec = recall_at_k(I_true, I_ivf, k=10)
        # save index to get size on disk
        fname = f"ivfpq_nlist{s['nlist']}_M{s['M']}_nbits{s['nbits']}.index"
        faiss.write_index(ivfpq, fname)
        size_bytes = os.path.getsize(fname)
        os.remove(fname)
        print(f"IVFPQ: latency per query={ivf_latency*1000:.3f} ms, recall@10={rec*100:.2f}%, on-disk size={size_bytes/1024/1024:.3f} MB")

        # quick re-rank: retrieve top-50 from IVFPQ then re-score against exact vectors
        k_retr = 50
        D_ivf50, I_ivf50, _ = ivfpq.search(xq, k_retr)
        # compute exact L2 for re-ranking
        reranked_I = np.zeros((xq.shape[0], 10), dtype='int64')
        for i in range(xq.shape[0]):
            cand_ids = I_ivf50[i]
            cand_vectors = xb[cand_ids]
            dists = np.sum((cand_vectors - xq[i:i+1])**2, axis=1)
            idx_sorted = np.argsort(dists)[:10]
            reranked_I[i] = cand_ids[idx_sorted]
        rec_rerank = recall_at_k(I_true, reranked_I, k=10)
        print(f"After re-rank (top {k_retr} => 10): recall@10={rec_rerank*100:.2f}%")

if __name__ == '__main__':
    main()

```

## Code Explanation

main.py builds a synthetic (or optional sentence-transformers) embedding dataset, creates an exact IndexFlatL2 baseline, then trains several IndexIVFPQ configurations. It measures per‑query latency, recall@10 against the baseline, and index on‑disk size. It also demonstrates a fast re‑ranking step (retrieve top‑50 from IVFPQ then compute exact distances) to recover recall. The script includes sensible default parameter combinations to illustrate tradeoffs.

## Real-World Applications

- Large-scale semantic search and RAG backends
- Recommendation engines where memory is constrained
- Embedding storage for search over millions of documents
- On-prem and edge deployments where GPU resources are limited

## Limitations

- IVFPQ is approximate: high recall at very high compression can be unattainable; careful tuning and adequate training data are required
- PQ compresses vectors and discards information—cannot recover exact vector values from codes
- IndexIVFPQ training requires representative data and enough samples (see recommended formulas); poor training yields low recall
- faiss-cpu may be slower than GPU builds for extremely large corpora

## Further Learning

- https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- https://www.pinecone.io/learn/series/faiss/product-quantization/
- https://docs.opensearch.org/latest/vector-search/optimizing-storage/faiss-product-quantization/
- https://github.com/facebookresearch/faiss

## Sources

- https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- https://www.pinecone.io/learn/series/faiss/product-quantization/
- https://docs.opensearch.org/latest/vector-search/optimizing-storage/faiss-product-quantization/

