# Practical Agent Memory Tools — Build Manage/Search Memory for Language Agents (FAISS + Sentence‑Transformers)

**Category:** daily

## Overview

Agent memory lets models recall past facts, user preferences and long‑term context. This resource explains memory-tool concepts used by modern agent frameworks (e.g., LangChain’s langmem), verifies design tradeoffs, and provides a small, self-contained Python project that implements two memory tools — manage_memory (write) and search_memory (semantic retrieval) — using sentence-transformers + FAISS. The example is framework‑agnostic and suitable for local experiments, debugging, and learning how agent memory interacts with retrieval and prompt construction.

## Problem

Agents that repeatedly interact with users or systems need persistent, retrievable context. Naive approaches (storing raw chat logs or always passing full conversation history) either blow up token budgets or fail to surface the right facts at the right time. Developers need practical memory tools that allow: (1) compact, semantic storage of memory items, (2) efficient retrieval of relevant memories for a given query, and (3) simple interfaces that an agent can call (manage/search).

## Technical Explanation

Modern agent memory is usually separated into two capabilities: (A) Write/manage memory: add, update, tag, or delete memory items; (B) Search/retrieve memory: given a query or conversational state, return the most relevant memory items (semantic retrieval). Implementations commonly encode text into dense vectors (embeddings) and store them in a vector index (FAISS, HNSWlib, Milvus, etc.). On retrieval, the query is embedded and nearest neighbors are returned; results are then filtered or summarized and added to the model prompt. This reduces token cost and helps the agent surface relevant long‑term facts and user preferences. The LangChain langmem project provides memory tools with similar semantics; the example here reproduces the same manage/search semantics with minimal dependencies so you can inspect, extend and integrate with any agent framework.

## Key Concepts

- Memory types: short‑term (buffer), summary memory, long‑term (semantic) memory
- Manage tool: API to add/update/delete memory items (text + metadata)
- Search tool: semantic retrieval using embeddings + vector index
- Embeddings: sentence-transformers (all‑MiniLM‑L6‑v2) for compact, fast vectors
- Index: FAISS for efficient nearest‑neighbor lookup
- Prompt composition: include retrieved memory items selectively before calling the model
- Tradeoffs: freshness vs. compression, privacy, token budget, retrieval latency

## Practical Example

The included Python project implements a tiny memory service exposing two functions: manage_memory(action, payload) and search_memory(query, top_k). manage_memory supports adding and deleting memory items (each item has id, text, timestamp, tags). search_memory returns the top‑k semantically similar memories for a query. The demo CLI shows adding memories and asking the system to respond — the response is a simple template that includes retrieved memories, which simulates how you would augment a prompt before calling an LLM in production.

## Python Implementation

```python
#!/usr/bin/env python3
"""main.py
Simple local memory tools using sentence-transformers + FAISS.
- manage_memory: add/delete items
- search_memory: semantic search returning top-k items
This script is intentionally self-contained and doesn't call any external LLM so it runs offline.
"""
import uuid
import time
import json
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ----- Simple in-memory metadata store -----
class MemoryStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner-product on normalized vectors == cosine
        self.embeddings: List[np.ndarray] = []
        self.metadatas: List[Dict[str, Any]] = []

    def _ensure_normalized(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]):
        emb = self._ensure_normalized(embedding).astype('float32')
        self.index.add(np.expand_dims(emb, axis=0))
        self.embeddings.append(emb)
        self.metadatas.append(metadata)

    def delete_by_id(self, item_id: str) -> bool:
        # FAISS IndexFlat doesn't support deletion; perform logical delete in metadata
        for m in self.metadatas:
            if m['id'] == item_id and not m.get('deleted', False):
                m['deleted'] = True
                return True
        return False

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        q = self._ensure_normalized(query_embedding).astype('float32')
        if len(self.embeddings) == 0:
            return []
        D, I = self.index.search(np.expand_dims(q, axis=0), min(top_k, len(self.embeddings)))
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            meta = self.metadatas[idx]
            if meta.get('deleted'):
                continue
            results.append({'score': float(dist), 'metadata': meta})
        return results

# ----- Memory tool wrapper -----
class MemoryTools:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        print('Loading embedding model:', model_name)
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.store = MemoryStore(self.dim)

    def manage_memory(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Supports actions: add, delete
        add payload: {"text": str, "tags": [str,...] (optional)}
        delete payload: {"id": str}
        """
        if action == 'add':
            text = payload.get('text')
            if not text:
                return {'ok': False, 'error': 'text required'}
            item_id = str(uuid.uuid4())
            metadata = {
                'id': item_id,
                'text': text,
                'tags': payload.get('tags', []),
                'timestamp': time.time(),
            }
            emb = self.embedder.encode(text)
            self.store.add(emb, metadata)
            return {'ok': True, 'id': item_id}
        elif action == 'delete':
            item_id = payload.get('id')
            if not item_id:
                return {'ok': False, 'error': 'id required'}
            ok = self.store.delete_by_id(item_id)
            return {'ok': ok}
        else:
            return {'ok': False, 'error': f'unknown action {action}'}

    def search_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode(query)
        results = self.store.search(q_emb, top_k=top_k)
        # Map to simple structure
        out = []
        for r in results:
            out.append({'id': r['metadata']['id'], 'text': r['metadata']['text'], 'score': r['score'], 'tags': r['metadata'].get('tags', [])})
        return out

# ----- Demo CLI -----
def pretty_print_mem(mem_list: List[Dict[str, Any]]):
    if not mem_list:
        print('  (no memories found)')
        return
    for i, m in enumerate(mem_list, 1):
        print(f"{i}. id={m['id']} score={m['score']:.4f} tags={m.get('tags', [])}\n   {m['text']}")


def main():
    tools = MemoryTools()
    print('\nSimple memory CLI. Commands:')
    print('  add <text> [|tag1,tag2]')
    print('  del <id>')
    print('  search <query> [k]')
    print('  exit')

    while True:
        try:
            raw = input('\n> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting')
            break
        if not raw:
            continue
        parts = raw.split(' ', 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ''
        if cmd == 'add':
            # optional tags after a '|' character
            if '|' in arg:
                text, tags = [p.strip() for p in arg.split('|', 1)]
                tags = [t.strip() for t in tags.split(',') if t.strip()]
            else:
                text = arg.strip()
                tags = []
            if not text:
                print('Provide text to add')
                continue
            res = tools.manage_memory('add', {'text': text, 'tags': tags})
            print('Added:', json.dumps(res))
        elif cmd in ('del', 'delete'):
            item_id = arg.strip()
            if not item_id:
                print('Provide id to delete')
                continue
            res = tools.manage_memory('delete', {'id': item_id})
            print('Deleted:' , res)
        elif cmd == 'search':
            if not arg:
                print('Provide a query')
                continue
            # optional k
            qparts = arg.rsplit(' ', 1)
            if len(qparts) == 2 and qparts[1].isdigit():
                query = qparts[0]
                k = int(qparts[1])
            else:
                query = arg
                k = 5
            results = tools.search_memory(query, top_k=k)
            print('\nRetrieved memories:')
            pretty_print_mem(results)
            # Simulate prompt augmentation
            print('\nSimulated agent prompt (you would pass this to an LLM):')
            prompt = 'Relevant memories:\n'
            for r in results:
                prompt += f"- {r['text']}\n"
            prompt += f"\nUser query: {query}\nAgent response: "
            print(prompt)
        elif cmd in ('exit', 'quit'):
            print('Bye')
            break
        else:
            print('Unknown command:', cmd)

if __name__ == '__main__':
    main()

```

## Code Explanation

main.py implements a tiny memory service: it loads the sentence-transformers all-MiniLM-L6-v2 model, creates a FAISS inner-product index (used with normalized vectors as a cosine proxy), and offers manage_memory and search_memory functions. The CLI demonstrates adding memories (with optional tags), deleting by id (logical delete), and performing semantic search that returns the top-K relevant memories. The script intentionally does not call an external LLM; instead it prints a simulated prompt that shows how retrieved memories would be concatenated and supplied to a model.

## Real-World Applications

- Personal assistants that remember user preferences (likes/dislikes) across sessions
- Customer support agents that recall previous tickets or account details to avoid repeating steps
- Multi-step agents that persist partial results and facts between planning and execution
- Customization/personalization for chatbots (e.g., preferred name, frequent topics)

## Limitations

- This demo uses FAISS IndexFlatIP which does not support deletions — we implement logical deletes in metadata. Production systems typically use indexes that support deletion or persistent vector DBs (Milvus, Weaviate, Pinecone)
- No built-in summarization or freshness control — long-term stores benefit from compaction (summarize older memories)
- Privacy and compliance: storing user memories must meet privacy, consent and data retention requirements
- Embedding model tradeoffs: all-MiniLM-L6-v2 is fast and compact but less capable on nuanced semantics than larger embedding models

## Further Learning

- https://langchain-ai.github.io/langmem/
- https://github.com/langchain-ai/langmem
- https://docs.langchain.com/oss/python/langchain/quickstart
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://faiss.ai/

## Sources

- https://langchain-ai.github.io/langmem/
- https://github.com/langchain-ai/langmem
- https://docs.langchain.com/oss/python/langchain/quickstart
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://faiss.ai/

