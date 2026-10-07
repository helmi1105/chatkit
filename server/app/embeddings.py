"""OpenAI text embeddings with a local, content-addressed course cache."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np
from openai import OpenAI

_lock = RLock()


class EmbeddingError(RuntimeError):
    pass


def _settings():
    return (os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
            Path(os.getenv('OPENAI_EMBEDDING_CACHE_DIR', str(Path(__file__).parent / 'data' / 'openai_embeddings'))))


def _client():
    key = os.getenv('OPENAI_EMBEDDING_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not key:
        raise EmbeddingError('Configurez OPENAI_EMBEDDING_API_KEY ou OPENAI_API_KEY sur le serveur pour la recherche dans le cours.')
    return OpenAI(api_key=key, base_url='https://api.openai.com/v1', timeout=45.0, max_retries=2)


def _normalize(vectors, rows):
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError('Invalid embedding response or cache')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Empty embedding vector')
    return values / norms


def _embed(client, model, texts):
    vectors = []
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        # Course chunks are <=1300 characters. Bound query inputs as well.
        inputs = [text.encode('utf-8')[:6000].decode('utf-8', errors='ignore').strip() for text in batch]
        if any(not text for text in inputs):
            raise ValueError('Empty embedding input')
        response = client.embeddings.create(model=model, input=inputs, encoding_format='float')
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(batch))):
            raise ValueError('Incomplete embedding response')
        vectors.extend(item.embedding for item in ordered)
    return _normalize(vectors, len(texts))


@lru_cache(maxsize=2)
def _course_vectors(model, cache_dir, chunks):
    digest = hashlib.sha256(json.dumps(['openai-v1', model, chunks], ensure_ascii=False).encode()).hexdigest()
    directory = Path(cache_dir)
    path = directory / f'{digest}.npz'
    try:
        with np.load(path, allow_pickle=False) as data:
            return _normalize(data['vectors'], len(chunks))
    except (OSError, ValueError, KeyError):
        pass
    with _client() as client:
        vectors = _embed(client, model, [text for _, text in chunks])
    directory.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.npz', delete=False) as temp:
            temp_path = Path(temp.name)
            np.savez_compressed(temp, vectors=vectors)
        os.replace(temp_path, path)
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
    return vectors


def prepare_course(chunks):
    if not chunks:
        raise EmbeddingError('Aucun contenu de cours disponible.')
    model, cache_dir = _settings()
    with _lock:
        return _course_vectors(model, str(cache_dir), tuple(chunks))


def semantic_scores(chunks, queries):
    if not chunks or not queries:
        return np.empty((len(queries), len(chunks)), dtype=np.float32)
    try:
        with _lock:
            passages = prepare_course(chunks)
        model, _ = _settings()
        with _client() as client:
            vectors = _embed(client, model, queries)
        return vectors @ passages.T
    except EmbeddingError:
        raise
    except Exception as exc:
        # Do not expose credentials, request payloads or provider response bodies.
        raise EmbeddingError('La recherche OpenAI embeddings a échoué. Vérifiez la clé serveur, son crédit et la connexion.') from exc


def hybrid_ranking(lexical_scores, dense_scores, limit=5):
    lexical = sorted((i for i, score in enumerate(lexical_scores) if score > 0),
                     key=lambda i: lexical_scores[i], reverse=True)[:20]
    semantic = sorted(range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True)[:20]
    fused = {}
    for ranking in (lexical, semantic):
        for rank, i in enumerate(ranking, 1):
            fused[i] = fused.get(i, 0.) + 1. / (60 + rank)
    return sorted(fused, key=lambda i: fused[i], reverse=True)[:limit]
