"""Run python -m app.prepare_embeddings to index the course once."""
from app.content import doctrine
from app.embeddings import prepare_course, _settings
from app.visual_retrieval import build_visual_chunks

if __name__ == '__main__':
    chunks = build_visual_chunks(doctrine())
    vectors = prepare_course(chunks)
    print(f'Ready: {len(chunks)} chunks, {vectors.shape[1]} dimensions, {_settings()[0]}')
