# main.py
import os
import sys
import argparse
import math
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)


SAMPLE_SNIPPETS = [
    "def add(a, b):\n    return a + b",
    "def subtract(a, b):\n    return a - b",
    "class Greeter:\n    def greet(self, name):\n        return f'Hello, {name}'",
]


def get_tokenizer_and_model(model_name: str = 'microsoft/codebert-base'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    # model_output.last_hidden_state: (B, T, D)
    token_embeddings = model_output.last_hidden_state
    # attention_mask: (B, T)
    counts = attention_mask.sum(dim=1).unsqueeze(-1)  # (B,1)
    counts = counts.clamp(min=1e-9).to(token_embeddings.dtype)
    summed = (token_embeddings * attention_mask.unsqueeze(-1).to(token_embeddings.dtype)).sum(dim=1)
    return summed / counts


def l2_normalize(vecs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norms = torch.norm(vecs, p=2, dim=1, keepdim=True).clamp(min=eps)
    return vecs / norms


def embed_texts(texts: List[str], tokenizer, model, device='cpu', max_length=512) -> List[List[float]]:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = mean_pooling(outputs, attention_mask)  # (B, D)
    pooled = l2_normalize(pooled)  # normalize for COSINE metric
    return pooled.cpu().numpy().tolist()


def read_files_from_dir(directory: str, exts=('.py', '.java', '.txt'), max_length=512) -> List[Tuple[str, str]]:
    items = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(exts):
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                        text = fh.read()
                        if not text.strip():
                            continue
                        items.append((path, text))
                except Exception as e:
                    print(f"Warning: could not read {path}: {e}", file=sys.stderr)
    return items


def ensure_collection(milvus_uri: str, collection_name: str, dim: int):
    # milvus_uri is unused for pymilvus connections.connect which takes host, port; keep milvus_uri as hint or 'host:port'
    host = 'localhost'
    port = '19530'
    if ':' in milvus_uri:
        host, port = milvus_uri.split(':')
    try:
        connections.connect(host=host, port=port)
    except Exception as e:
        raise RuntimeError(f'Failed to connect to Milvus at {host}:{port}: {e}')

    try:
        if utility.has_collection(collection_name):
            coll = Collection(collection_name)
            return coll
    except Exception:
        # utility.has_collection may raise on some versions; fallthrough to creation
        pass

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="code snippets collection")
    try:
        coll = Collection(name=collection_name, schema=schema)
    except Exception as e:
        raise RuntimeError(f'Failed to create collection: {e}')
    return coll


def insert_documents(collection: Collection, codes: List[str], embeddings: List[List[float]]):
    if len(codes) != len(embeddings):
        raise ValueError('codes and embeddings length mismatch')
    try:
        # auto_id=True -> do NOT pass id column. Only pass fields excluding primary auto id.
        result = collection.insert([codes, embeddings])
    except Exception as e:
        raise RuntimeError(f'Failed to insert vectors into Milvus: {e}')
    # result is InsertResult: can read .insert_count and .primary_keys
    inserted_count = getattr(result, 'insert_count', None)
    pks = getattr(result, 'primary_keys', None)
    print(f'Inserted rows: {inserted_count}, pks sample: {pks[:5] if pks is not None else None}')
    return result


def create_hnsw_index(collection: Collection, field_name: str = 'emb'):
    index_params = {
        'index_type': 'HNSW',
        'metric_type': 'COSINE',
        'params': {'M': 16, 'efConstruction': 200},
    }
    try:
        # use named arg index_params for clarity
        collection.create_index(field_name=field_name, index_params=index_params)
        collection.load()  # ensure collection is loaded into memory for searching
    except Exception as e:
        raise RuntimeError(f'Failed to create/load index: {e}')


def search(collection: Collection, query_emb: List[float], top_k: int = 5):
    search_params = {'metric_type': 'COSINE', 'params': {'ef': 64}}
    try:
        results = collection.search([query_emb], "emb", params=search_params, limit=top_k, output_fields=["code"])
    except TypeError:
        # Older/newer versions could require 'param' key — try fallback
        try:
            results = collection.search([query_emb], "emb", param=search_params, limit=top_k, output_fields=["code"])
        except Exception as e:
            raise RuntimeError(f'Failed to search collection: {e}')
    except Exception as e:
        raise RuntimeError(f'Failed to search collection: {e}')

    out = []
    for hits in results:
        for hit in hits:
            out.append({'id': hit.id, 'score': hit.score, 'code': hit.entity.get('code') if hit.entity else None})
    return out


def main(args):
    tokenizer, model = get_tokenizer_and_model()
    device = 'cpu'
    model.to(device)

    # Load data
    docs = []
    if args.dir:
        files = read_files_from_dir(args.dir, max_length=args.max_length)
        if not files:
            print('No files found in directory; falling back to SAMPLE_SNIPPETS')
            docs = [(f'snippet_{i}.py', s) for i, s in enumerate(SAMPLE_SNIPPETS)]
        else:
            docs = files
    else:
        docs = [(f'snippet_{i}.py', s) for i, s in enumerate(SAMPLE_SNIPPETS)]

    codes = [text for _, text in docs]

    emb = embed_texts(codes, tokenizer, model, device=device, max_length=args.max_length)

    # connect/create collection
    coll = ensure_collection(args.milvus, args.collection, dim=len(emb[0]))

    # insert
    insert_documents(coll, codes, emb)

    # create index (HNSW) and load
    create_hnsw_index(coll, field_name='emb')

    # Do a sample search with the first snippet
    q_emb = emb[0]
    results = search(coll, q_emb, top_k=args.top_k)
    print('Top results:')
    for r in results:
        print(r['score'], r['code'][:200].replace('\n', ' '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local Milvus code snippet index demo')
    parser.add_argument('--milvus', type=str, default='localhost:19530', help='Milvus host:port')
    parser.add_argument('--collection', type=str, default='code_snippets', help='Milvus collection name')
    parser.add_argument('--dir', type=str, default='', help='Optional directory to index (.py, .java, .txt)')
    parser.add_argument('--max_length', type=int, default=512, help='Max tokens to keep per file (truncates)')
    parser.add_argument('--top_k', type=int, default=5, help='Search top-k')
    args = parser.parse_args()
    main(args)
