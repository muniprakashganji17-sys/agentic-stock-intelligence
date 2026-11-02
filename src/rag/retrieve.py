"""Phase 3.3 (RAG): search the chunks - with the no-look-ahead rule built in.

    r = Retriever.load()
    r.search("earnings results and guidance", ticker="NVDA", as_of="2024-05-23", k=5)

`as_of` = the day t at whose CLOSE the prediction is made. Only chunks of that ticker with
    effective_date <= as_of            (and, if given, effective_date >= as_of - lookback_days)
are even considered. The filter runs BEFORE ranking, so a later chunk can never win, however
similar it is. Search is exact (every candidate is compared), no approximation.
"""
import os

import numpy as np
import pandas as pd

import config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # recommended by the BGE model
SHOW = ["rank", "similarity", "ticker", "effective_date", "doc_type", "text", "url", "chunk_id", "source_id"]


class Retriever:
    def __init__(self, chunks, emb, encoder):
        if len(chunks) != len(emb):
            raise ValueError(f"{len(chunks)} chunks but {len(emb)} vectors")
        self.chunks = chunks.reset_index(drop=True)
        self.emb = emb
        self.encoder = encoder
        self._ticker = self.chunks["ticker"].to_numpy()
        self._date = pd.to_datetime(self.chunks["effective_date"]).to_numpy()
        self._type = self.chunks["doc_type"].to_numpy()

    @classmethod
    def load(cls, encoder=None):
        from src.rag import embed
        chunks = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "chunks.parquet"))
        emb = np.load(os.path.join(config.PROCESSED_DIR, "chunk_embeddings.npy"))
        ids = np.load(os.path.join(config.PROCESSED_DIR, "chunk_ids.npy"))
        if not np.array_equal(ids, chunks["chunk_id"].to_numpy().astype(str)):
            raise ValueError("chunk_embeddings.npy does not match chunks.parquet - rerun embed.run()")
        return cls(chunks, emb, encoder or embed.load_encoder())

    def embed_query(self, query):
        v = self.encoder.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.astype(np.float32)

    def candidates(self, ticker, as_of, lookback_days=None, doc_types=None):
        """Row numbers allowed at prediction time `as_of` (this is the no-look-ahead filter)."""
        as_of = np.datetime64(pd.Timestamp(as_of).normalize())
        ok = (self._ticker == ticker) & (self._date <= as_of)
        if lookback_days is not None:
            ok &= self._date >= as_of - np.timedelta64(int(lookback_days), "D")
        if doc_types is not None:
            ok &= np.isin(self._type, list(doc_types))
        return np.flatnonzero(ok)

    def search(self, query, ticker, as_of, k=5, lookback_days=None, doc_types=None, qvec=None):
        rows = self.candidates(ticker, as_of, lookback_days, doc_types)
        if len(rows) == 0:
            return pd.DataFrame(columns=SHOW)
        q = self.embed_query(query) if qvec is None else qvec
        sims = self.emb[rows] @ q
        top = np.argsort(-sims, kind="stable")[:k]
        out = self.chunks.iloc[rows[top]].copy()
        out["similarity"] = sims[top].round(3)
        out["rank"] = np.arange(1, len(top) + 1)
        return out[SHOW].reset_index(drop=True)


def audit(r, query="quarterly results revenue earnings per share outlook", k=5, lookback_days=30):
    """Real-data look-ahead audit using every earnings 8-K (item 2.02) that has an exhibit.

    For each such filing F (ticker T, effective_date E):
      - searching on day E must be ALLOWED to find F  -> 'hit rate on day E'
      - searching on the business day BEFORE E must NEVER find F -> 'leaks' (must be 0)
    """
    qvec = r.embed_query(query)
    ch = r.chunks
    fil = ch[(ch["doc_type"] == "exhibit") & ch["items"].fillna("").str.contains("2.02")]
    fil = fil.groupby("source_id").agg(ticker=("ticker", "first"), eff=("effective_date", "first")).reset_index()
    hit = leak = 0
    for f in fil.itertuples():
        e = pd.Timestamp(f.eff)
        on_day = r.search(query, f.ticker, e, k=k, lookback_days=lookback_days, qvec=qvec)
        day_before = r.search(query, f.ticker, e - pd.offsets.BDay(1), k=k, lookback_days=lookback_days, qvec=qvec)
        hit += int((on_day["source_id"] == f.source_id).any())
        leak += int((day_before["source_id"] == f.source_id).any())
        assert (pd.to_datetime(on_day["effective_date"]) <= e).all()
    n = len(fil)
    print(f"Audit over {n} earnings 8-Ks (top {k}, last {lookback_days} days):")
    print(f"  found on its effective day : {hit}/{n} = {hit / n:.0%}")
    print(f"  found the day BEFORE (leak): {leak}/{n}   <- must be 0")
    return hit, leak, n


def run():
    r = Retriever.load()
    print("\nExample: NVDA, earnings season (NVIDIA reported after the close on 22 May 2024)")
    for day in ("2024-05-22", "2024-05-23"):
        print(f"\n--- as_of {day} ---")
        res = r.search("earnings results revenue and guidance", "NVDA", day, k=3, lookback_days=30)
        for x in res.itertuples():
            print(f"  [{x.rank}] sim {x.similarity} | {str(x.effective_date)[:10]} | {x.doc_type} | {x.text[:110]!r}")
    print()
    audit(r)
    return r