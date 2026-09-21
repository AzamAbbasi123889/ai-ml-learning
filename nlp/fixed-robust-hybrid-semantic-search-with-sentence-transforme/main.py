"""main.py
Runnable demo: hybrid BM25 + dense search with FAISS (or sklearn fallback) and optional CrossEncoder rerank.
Run: python main.py
"""
import os
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Retrieval libs
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    # SentenceTransformer is required; if import fails the script will stop below when used
    CROSS_ENCODER_AVAILABLE = False

from rank_bm25 import BM25Okapi

# FAISS optional
USE_FAISS = True
try:
    import faiss
except Exception:
    faiss = None
    USE_FAISS = False

# sklearn fallback for dense search if FAISS not available
from sklearn.neighbors import NearestNeighbors

# Small utility functions

def ensure_f32_contiguous(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    return x


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


# Score normalization methods
from scipy.special import softmax

def normalize_scores(scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Supported methods: 'minmax', 'zscore', 'softmax', 'rank'.
    Note: minmax can be brittle for outliers or tiny ranges. zscore is robust to scale but still influenced by outliers.
    rank uses 1/(rank + 60) style mapping (like RRF mapping) and ignores absolute magnitudes.
    """
    if method == 'minmax':
        mn = scores.min()
        mx = scores.max()
        if mx - mn < 1e-6:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)
    elif method == 'zscore':
        mu = scores.mean()
        sd = scores.std()
        if sd < 1e-6:
            return np.zeros_like(scores)
        return (scores - mu) / sd
    elif method == 'softmax':
        return softmax(scores)
    elif method == 'rank':
        # rank: higher score => better rank (1 is best). Convert to ranks and map to 1/(rank + k)
        ranks = np.argsort(np.argsort(-scores))  # 0 is best
        k = 60
        return 1.0 / (ranks + 1 + k)
    else:
        raise ValueError('Unknown normalization method')


# RRF merging (reciprocal rank fusion)
def rrf_merge(rank_lists, k=60):
    # rank_lists: list of lists of doc ids in order (best->worst)
    scores = {}
    for rl in rank_lists:
        for rank, doc_id in enumerate(rl):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + 1 + rank)
    # return sorted doc ids
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def build_faiss_index(embeddings: np.ndarray, metric='ip'):
    # embeddings must be float32 and contiguous
    d = embeddings.shape[1]
    if metric == 'ip':
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
    # For inner product search with normalized vectors, use IP; for L2 use L2.
    index.add(ensure_f32_contiguous(embeddings))
    return index


def build_sklearn_index(embeddings: np.ndarray):
    nbrs = NearestNeighbors(n_neighbors=min(10, len(embeddings)), algorithm='brute', metric='cosine')
    nbrs.fit(ensure_f32_contiguous(embeddings))
    return nbrs


def dense_search_faiss(index, query_emb: np.ndarray, top_k=5, normalized=True):
    q = ensure_f32_contiguous(query_emb)
    if normalized:
        q = l2_normalize(q, axis=1)
    D, I = index.search(q, top_k)
    # For IP, faiss returns similarity (higher is better). For L2, returns distances.
    return I[0].tolist(), D[0].tolist()


def dense_search_sklearn(nbrs, query_emb: np.ndarray, top_k=5):
    q = ensure_f32_contiguous(query_emb)
    dists, idxs = nbrs.kneighbors(q, n_neighbors=top_k)
    # sklearn cosine returns distances (lower is better) -> convert to similarity
    sims = 1.0 - dists[0]
    return idxs[0].tolist(), sims.tolist()


def bm25_search(bm25, tokenized_corpus, query, top_k=5):
    q_tokens = query.split()
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return top_idx.tolist(), scores[top_idx].tolist()


if __name__ == '__main__':
    # Small demo corpus
    corpus = [
        "The Apollo program landed the first humans on the Moon.",
        "Python is a popular programming language for data science and machine learning.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "The Eiffel Tower is located in Paris.",
        "Neural networks are used for image and text tasks.",
        "Sentence-Transformers provide easy-to-use sentence embeddings.",
    ]

    # Prepare BM25
    tokenized = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    # Load dense encoder
    model_name = 'all-MiniLM-L6-v2'
    print('Loading SentenceTransformer:', model_name)
    dense_model = SentenceTransformer(model_name)

    # Encode corpus (convert_to_numpy=True, show_progress_bar=False)
    corpus_emb = dense_model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    corpus_emb = ensure_f32_contiguous(corpus_emb)
    corpus_emb = l2_normalize(corpus_emb, axis=1)

    # Build dense index: prefer faiss if available
    if USE_FAISS and faiss is not None:
        print('Building FAISS index (faiss-cpu/faiss-gpu)')
        # We'll use inner product over L2 when embeddings are normalized
        index = build_faiss_index(corpus_emb, metric='ip')
        dense_index_type = 'faiss'
    else:
        print('FAISS not available, using sklearn NearestNeighbors fallback (exact search)')
        nbrs = build_sklearn_index(corpus_emb)
        dense_index_type = 'sklearn'

    # Optional CrossEncoder (reranker)
    cross_encoder = None
    if CROSS_ENCODER_AVAILABLE:
        try:
            # Use a small cross-encoder to keep CPU runs reasonable; users with GPU can install a larger one.
            reranker_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
            print('Loading CrossEncoder (optional reranker):', reranker_name)
            cross_encoder = CrossEncoder(reranker_name)
        except Exception as e:
            print('CrossEncoder not available or failed to load (will skip rerank):', e)
            cross_encoder = None
    else:
        print('sentence_transformers.cross_encoder.CrossEncoder not importable; skipping rerank')

    # Example queries
    queries = [
        'moon landing',
        'how to do machine learning in python',
        'fast similarity search library',
    ]

    # Per-query retrieval in parallel (BM25 + dense). This reduces per-query latency when both methods are used.
    # If you prefer sequential execution, replace with a simple for loop.
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {}
        for q in queries:
            futures[ex.submit(lambda x: x, q)] = q

        for fut in as_completed(futures):
            q = futures[fut]
            print('\nQuery:', q)
            # Run both retrievals concurrently for this query
            with ThreadPoolExecutor(max_workers=2) as inner_ex:
                f_bm25 = inner_ex.submit(bm25_search, bm25, tokenized, q, 5)
                if dense_index_type == 'faiss':
                    q_emb = dense_model.encode([q], convert_to_numpy=True, show_progress_bar=False)
                    q_emb = ensure_f32_contiguous(q_emb.astype(np.float32, copy=False))
                    q_emb = l2_normalize(q_emb, axis=1)
                    f_dense = inner_ex.submit(dense_search_faiss, index, q_emb, 5, True)
                else:
                    q_emb = dense_model.encode([q], convert_to_numpy=True, show_progress_bar=False)
                    q_emb = ensure_f32_contiguous(q_emb.astype(np.float32, copy=False))
                    q_emb = l2_normalize(q_emb, axis=1)
                    f_dense = inner_ex.submit(dense_search_sklearn, nbrs, q_emb, 5)

                bm25_idx, bm25_scores = f_bm25.result()
                dense_idx, dense_scores = f_dense.result()

            print('BM25 top ids/scores:', bm25_idx, bm25_scores)
            print('Dense top ids/scores:', dense_idx, dense_scores)

            # Score normalization and fusion example
            bm25_norm = normalize_scores(np.array(bm25_scores), method='minmax')
            dense_norm = normalize_scores(np.array(dense_scores), method='minmax')

            # Weighted fusion (simple): weight dense higher
            alpha = 0.6
            fused = {}
            for i, doc_id in enumerate(bm25_idx):
                fused[doc_id] = fused.get(doc_id, 0.0) + (1 - alpha) * bm25_norm[i]
            for i, doc_id in enumerate(dense_idx):
                fused[doc_id] = fused.get(doc_id, 0.0) + alpha * dense_norm[i]

            fused_sorted = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
            print('Fused results (minmax):', fused_sorted)

            # Demonstrate alternatives and limitations: zscore and softmax
            bm25_z = normalize_scores(np.array(bm25_scores), method='zscore')
            dense_soft = normalize_scores(np.array(dense_scores), method='softmax')
            # Note: mixing zscore and softmax may be inappropriate; shown for demonstration only

            # Reciprocal Rank Fusion (RRF) example with two k values
            rrf_k_small = rrf_merge([bm25_idx, dense_idx], k=30)
            rrf_k_large = rrf_merge([bm25_idx, dense_idx], k=60)
            print('RRF k=30 top:', rrf_k_small[:5])
            print('RRF k=60 top:', rrf_k_large[:5])

            # Optional CrossEncoder rerank (if available)
            if cross_encoder is not None:
                # Prepare pairs (query, candidate_text) for top-N fused results
                top_candidates = [doc_id for doc_id, _ in fused_sorted][:5]
                pairs = [[q, corpus[cid]] for cid in top_candidates]
                rerank_scores = cross_encoder.predict(pairs)
                # CrossEncoder returns logits/scores: higher is better
                reranked = sorted(zip(top_candidates, rerank_scores), key=lambda kv: kv[1], reverse=True)
                print('CrossEncoder reranked top:', reranked)
            else:
                print('Cross-encoder not used (optional). To enable, ensure sentence-transformers CrossEncoder is available and you have sufficient resources.')

            results.append({
                'query': q,
                'bm25': list(zip(bm25_idx, bm25_scores)),
                'dense': list(zip(dense_idx, dense_scores)),
                'fused': fused_sorted,
            })

    # Print a short snapshot
    print('\nExample output snapshot (first query):')
    import json
    print(json.dumps(results[0], indent=2))
