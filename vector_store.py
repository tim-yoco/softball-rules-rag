"""Lightweight vector store using pre-computed embeddings and numpy.

At ingest time (local dev): uses sentence-transformers to compute embeddings,
saves everything to a JSON + numpy file.

At query time (production): loads pre-computed data, embeds the query using
the same model, and does cosine similarity search. If sentence-transformers
is not available (low-memory deploy), uses a fallback that searches by
keyword matching only.
"""

import json
import os
import numpy as np

STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_store_data")
EMBEDDINGS_FILE = os.path.join(STORE_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(STORE_DIR, "metadata.json")


def get_embedding_model():
    """Returns the embedding model if available (local dev with chromadb installed)."""
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        return DefaultEmbeddingFunction()
    except Exception:
        return None


def has_embedding_model() -> bool:
    return get_embedding_model() is not None


def save_store(chunks: list[dict]):
    """Save chunks with pre-computed embeddings."""
    os.makedirs(STORE_DIR, exist_ok=True)

    model = get_embedding_model()
    if model is None:
        raise RuntimeError("Embedding model not available. Run ingestion locally.")

    texts = [c["text"] for c in chunks]
    embeddings = model(input=texts)
    embeddings_array = np.array(embeddings, dtype=np.float32)

    np.save(EMBEDDINGS_FILE, embeddings_array)

    metadata = [{
        "text": c["text"],
        "source": c["metadata"]["source"],
        "priority": c["metadata"]["priority"],
        "context": c["metadata"].get("context", ""),
        "filename": c["metadata"]["filename"],
    } for c in chunks]

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    print(f"  Saved {len(chunks)} chunks to {STORE_DIR}")


def load_store() -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    return embeddings, metadata


def query_store(query_texts: list[str], n_results: int = 10,
                where: dict | None = None) -> list[list[dict]]:
    """Search the vector store. Returns results per query text."""
    embeddings, metadata = load_store()
    model = get_embedding_model()

    if model is None:
        return _keyword_fallback(query_texts, metadata, n_results, where)

    query_embeddings = np.array(model(input=query_texts), dtype=np.float32)

    all_results = []
    for q_emb in query_embeddings:
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
        similarities = np.dot(embeddings, q_emb) / np.maximum(norms, 1e-10)
        distances = 1 - similarities

        indices = np.argsort(distances)

        results = []
        for idx in indices:
            entry = metadata[int(idx)]
            if where:
                if not all(entry.get(k) == v for k, v in where.items()):
                    continue
            results.append({
                "id": f"{entry['source']}_c{idx}",
                "text": entry["text"],
                "metadata": entry,
                "distance": float(distances[idx]),
            })
            if len(results) >= n_results:
                break
        all_results.append(results)

    return all_results


def _keyword_fallback(query_texts, metadata, n_results, where):
    """Fallback when embedding model isn't available."""
    import re
    all_results = []
    for query in query_texts:
        words = set(query.lower().split())
        scored = []
        for i, entry in enumerate(metadata):
            if where and not all(entry.get(k) == v for k, v in where.items()):
                continue
            doc_lower = entry["text"].lower()
            score = sum(1 for w in words if w in doc_lower)
            if score > 0:
                scored.append({
                    "id": f"{entry['source']}_c{i}",
                    "text": entry["text"],
                    "metadata": entry,
                    "distance": 1.0 - (score / len(words)),
                })
        scored.sort(key=lambda x: x["distance"])
        all_results.append(scored[:n_results])
    return all_results
