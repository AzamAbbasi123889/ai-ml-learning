# Practical Local RAG: Build a Retrieval‑Augmented Generation Pipeline with LangChain, Chroma, and Local Embeddings/LLM

**Category:** daily

## Overview

Step-by-step, practical guide to build a local Retrieval‑Augmented Generation (RAG) pipeline that indexes documents into a Chroma vector store, uses a sentence‑transformer embedding model, and performs simple generation with a local HuggingFace model. Includes a compact, runnable Python example (main.py + requirements.txt) you can run locally to experiment with RAG without any paid APIs.

## Problem

Modern LLMs are powerful but can hallucinate or lack up-to-date, private, or domain-specific knowledge. How do you ground model answers in your own documents while keeping data local and avoiding API costs and data leakage?

## Technical Explanation

RAG combines two components: (1) a retrieval subsystem that finds relevant pieces of text from an external knowledge base using embeddings and a vector database, and (2) a generation subsystem where an LLM conditions on the retrieved context to produce grounded answers. Typical pipeline steps: document loading → chunking/splitting → embedding → indexing into a vector store → query embedding → nearest‑neighbor retrieval → prompt construction with retrieved passages → LLM generation. Key engineering choices include embedding model (trade-off quality vs latency), chunk size/overlap, vector store (Chroma, FAISS, Qdrant), retrieval parameters (top_k), and whether to rerank retrieved candidates with a cross-encoder or HyDE (hypothetical document embeddings). The included example uses sentence-transformers (all-MiniLM-L6-v2) for embeddings, Chroma (local persistence/in-memory) for vector indexing, LangChain utilities for splitting/flow, and a small HuggingFace generation model to demonstrate end-to-end local RAG.

## Key Concepts

- Retrieval-Augmented Generation (RAG)
- Embeddings (semantic vector representations)
- Vector stores / similarity search (Chroma, FAISS, Qdrant)
- Chunking & overlap strategies
- Retriever → Generator interaction
- Grounding and faithfulness
- Hybrid and reranking techniques (dense + sparse, cross-encoder rerank)

## Practical Example

Small, self-contained Python project that: 1) loads plain text files from a local 'docs/' folder, 2) splits documents into chunks, 3) computes embeddings with sentence-transformers, 4) stores embeddings + metadata into a Chroma vector store, 5) runs a query, retrieves top-k chunks, builds a context-aware prompt, and generates an answer with a local HuggingFace model. This demonstrates a minimal, local RAG without external APIs and is suitable for learning and prototyping.

## Python Implementation

```python
import os
import glob
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- Config ---
DOCS_DIR = "docs"
CHROMA_PERSIST_DIR = "./chromadb_local"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, CPU-friendly
GEN_MODEL = "gpt2"  # small local generator for demo (replace with a larger local model if available)
TOP_K = 4
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Helpers ---

def load_documents(folder: str) -> List[dict]:
    """Load all .txt files from folder and return list of {'id', 'text', 'meta'}"""
    docs = []
    for path in glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({
            "id": os.path.relpath(path),
            "text": text,
            "meta": {"source": os.path.relpath(path)}
        })
    return docs


def chunk_documents(docs: List[dict]) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for d in docs:
        pieces = splitter.split_text(d["text"])  # list of strings
        for i, p in enumerate(pieces):
            chunks.append({
                "id": f"{d['id']}__{i}",
                "text": p,
                "meta": {**d["meta"], "chunk": i}
            })
    return chunks


class SentenceTransformersEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


def build_chroma_client(persist_directory: str):
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    return client


# --- Main flow ---

def main():
    # 1) Load and chunk docs
    docs = load_documents(DOCS_DIR)
    if not docs:
        print(f"No documents found in '{DOCS_DIR}'. Put some .txt files there and re-run.")
        return
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")

    # 2) Create embedder and chroma client
    embedder = SentenceTransformersEmbedder(EMBED_MODEL_NAME)
    client = build_chroma_client(CHROMA_PERSIST_DIR)

    # 3) Create / get collection
    collection = None
    try:
        collection = client.get_collection("local_rag")
    except Exception:
        collection = client.create_collection("local_rag")

    # 4) Compute embeddings and upsert
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["meta"] for c in chunks]
    embeddings = embedder.embed_documents(texts)

    # Upsert replaces previous data if ids match
    collection.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=embeddings)
    client.persist()
    print(f"Indexed {len(ids)} chunks into Chroma at '{CHROMA_PERSIST_DIR}'")

    # 5) Simple query -> retrieve
    query = input("Enter a question: ")
    q_emb = embedder.embed_query(query)
    results = collection.query(query_embeddings=[q_emb], n_results=TOP_K, include=['documents', 'metadatas', 'distances'])

    retrieved_docs = results.get('documents', [[]])[0]
    retrieved_meta = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0]

    print("\nRetrieved chunks:")
    context_parts = []
    for i, (doc, m, d) in enumerate(zip(retrieved_docs, retrieved_meta, distances)):
        print(f"[{i}] source={m.get('source')} chunk={m.get('chunk')} dist={d:.4f}")
        print(doc[:400].replace('\n', ' ') + ("..." if len(doc) > 400 else ""))
        print("---")
        context_parts.append(f"Source: {m.get('source')}\n{doc}")

    # 6) Build prompt and generate answer with a local HF model
    prompt = (
        "You are an assistant that answers user questions using only the provided context. "
        "If the answer is not present, say 'I don't know.'\n\nContext:\n" + "\n\n".join(context_parts)
        + "\n\nQuestion: " + query + "\nAnswer:")

    print("\nConstructed prompt (truncated):\n", prompt[:1000])

    # Load generator pipeline (small model for demo). For better quality, replace with a local Llama-style model.
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

    out = gen(prompt, max_new_tokens=200, do_sample=False)
    answer = out[0]["generated_text"][len(prompt):].strip()

    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()

```

## Code Explanation

main.py implements a minimal local RAG pipeline: it finds .txt files in docs/, splits them into chunks using LangChain's RecursiveCharacterTextSplitter, computes embeddings with sentence-transformers (all-MiniLM-L6-v2), stores them and their embeddings in a local Chroma collection, then accepts a user query, retrieves top-k chunks by vector similarity, constructs a context-aware prompt and generates an answer using a small HuggingFace causal LM (gpt2) via the transformers pipeline. Replace GEN_MODEL with a stronger local model (quantized Llama/Mistral variants) for production; replace or augment the embedder with higher-quality models if needed.

## Real-World Applications

- Internal knowledge base Q&A for enterprises with sensitive documents (on‑premises RAG).
- Customer support assistants that answer from product manuals and past tickets.
- Developer helpdesk: search code docs, RFCs, and generate contextual explanations.
- Research assistant: index papers, notes and query with grounded answers for reproducibility.

## Limitations

- Small local generation models (like gpt2) produce low‑quality answers; use larger or specialized local models for useful output.
- Embedding quality limits retrieval relevance — higher-quality embed models increase accuracy and cost/latency.
- Chroma in-process is good for prototypes; at scale consider Qdrant, Milvus, or managed vector DBs with replication/filters.
- No reranking or faithfulness checking is implemented — production systems should add cross‑encoder rerankers, citation tracebacks, and hallucination detection.

## Further Learning

- https://python.langchain.com/en/latest/use_cases/question_answering.html
- https://www.trychroma.com/docs/overview
- https://www.sbert.net/
- https://huggingface.co/docs/transformers/index
- https://github.com/langchain-ai/langchain

## Sources

- https://python.langchain.com/en/latest/use_cases/question_answering.html
- https://www.trychroma.com/docs/overview
- https://www.sbert.net/
- https://huggingface.co/docs/transformers/index
- https://localaimaster.com/blog/rag-local-setup-guide
- https://developer.dataiku.com/12/tutorials/machine-learning/genai/nlp/gpt-lc-chroma-rag/index.html

