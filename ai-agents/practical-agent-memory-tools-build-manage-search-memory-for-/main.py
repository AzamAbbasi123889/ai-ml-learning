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
