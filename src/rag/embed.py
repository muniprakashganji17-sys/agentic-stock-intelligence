"""Phase 3.2 (RAG): turn every chunk into a vector of numbers ("embedding").

Why: two texts that mean similar things get vectors that point in similar directions, so we can
later search "what was said about NVIDIA's guidance?" by comparing vectors, not exact words.

Model: BAAI/bge-small-en-v1.5 (free, open-source, 384 numbers per chunk). It runs best on the
free Colab GPU (T4): about 106k chunks in roughly 5-10 minutes; on CPU it would take hours.

Safe to stop and restart: vectors are saved to Drive in blocks of BLOCK chunks. A block is reused
only if its chunk ids are identical (a changed chunk table can never pick up old vectors).

Output (in processed/):
  chunk_embeddings.npy : float32 matrix (n_chunks x 384), each row normalised to length 1
  chunk_ids.npy        : the chunk_id of every row, same order (checked on load)
Embeddings only describe text. Dates and tickers stay in chunks.parquet; the retrieval step
filters by effective_date BEFORE ranking, so no look-ahead can happen here.
"""
import hashlib
import os

import numpy as np
import pandas as pd

import config

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BLOCK = 8000
MAX_SEQ = 320          # tokens; a 1200-character chunk is about 250-300 tokens


def load_encoder(device=None):
    """Load the sentence-transformers model (downloads about 130 MB the first time)."""
    import torch
    from sentence_transformers import SentenceTransformer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ
    if device == "cuda":
        model.half()
    print(f"Loaded {MODEL_NAME} on {device}")
    return model


def _encode_block(texts, encoder, batch_size=64):
    """Encode one block. Texts are sorted by length (faster batches), then put back in order."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    vecs = encoder.encode([texts[i] for i in order], batch_size=batch_size,
                          normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    out = np.empty_like(vecs, dtype=np.float32)
    out[order] = vecs.astype(np.float32)
    return out


def embed_chunks(chunks, encoder, cache_dir, block=BLOCK, log=print):
    """chunks: DataFrame with chunk_id and text. Returns float32 matrix aligned with its rows."""
    os.makedirs(cache_dir, exist_ok=True)
    ids, texts = chunks["chunk_id"].tolist(), chunks["text"].tolist()
    parts, reused = [], 0
    for start in range(0, len(texts), block):
        key = hashlib.md5("|".join(ids[start:start + block]).encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, f"emb_{start:07d}_{key}.npy")
        if os.path.exists(path):
            parts.append(np.load(path))
            reused += 1
        else:
            vec = _encode_block(texts[start:start + block], encoder)
            np.save(path, vec)
            parts.append(vec)
        log(f"  {min(start + block, len(texts))}/{len(texts)} chunks done")
    if reused:
        log(f"  (reused {reused} saved blocks)")
    return np.vstack(parts)


def run(encoder=None):
    chunks = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "chunks.parquet"))
    encoder = encoder or load_encoder()
    emb = embed_chunks(chunks, encoder, os.path.join(config.RAW_DIR, "embeddings"))
    assert emb.shape[0] == len(chunks)
    np.save(os.path.join(config.PROCESSED_DIR, "chunk_embeddings.npy"), emb)
    np.save(os.path.join(config.PROCESSED_DIR, "chunk_ids.npy"), chunks["chunk_id"].to_numpy().astype(str))

    norms = np.linalg.norm(emb, axis=1)
    print(f"\nSaved {emb.shape[0]} vectors x {emb.shape[1]} numbers "
          f"({emb.nbytes / 1e6:.0f} MB). Vector length: min {norms.min():.3f}, max {norms.max():.3f} (should be 1.0)")
    print("Any NaN:", bool(np.isnan(emb).any()))

    # quick feel-check: nearest chunk to a few sentences (no dates or tickers involved here)
    queries = ["quarterly revenue guidance and earnings results",
               "new product launch announced", "dividend declared by the board"]
    q = encoder.encode(["Represent this sentence for searching relevant passages: " + s for s in queries],
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    sims = emb @ q.T
    for j, s in enumerate(queries):
        i = int(sims[:, j].argmax())
        r = chunks.iloc[i]
        print(f"\nQuery: {s}\n  best match ({r['doc_type']}, {r['ticker']}, {str(r['effective_date'])[:10]}, "
              f"similarity {sims[i, j]:.2f}): {r['text'][:200]}")
    return emb