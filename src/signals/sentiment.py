"""Phase 4.1a (signals): score the tone of every chunk with FinBERT.

FinBERT (ProsusAI/finbert) is a free, open-source language model trained on financial text.
For each chunk it gives three probabilities: positive, negative, neutral. We also keep one
summary number:   sent = p_positive - p_negative   (from -1 very negative to +1 very positive).

Each chunk is scored on its OWN text only, so a score can never contain information from other
dates. The date rule (effective_date <= t) is applied later, when scores are combined into
daily features.

Output: processed/chunk_sentiment.parquet  (chunk_id, p_pos, p_neg, p_neu, sent)
Scores are saved in blocks on Drive, so a stopped run continues where it left off.
"""
import hashlib
import os

import numpy as np
import pandas as pd

import config

MODEL_NAME = "ProsusAI/finbert"
ORDER = ("positive", "negative", "neutral")       # column order of our output
BLOCK = 10000
SETTINGS = {                                      # doc_type group -> (max tokens, batch size)
    "news": (64, 256),                            # headlines are short
    "filing": (384, 32),                          # press-release / 8-K chunks (up to ~1200 characters)
}


def label_columns(id2label):
    """Position of positive / negative / neutral in the model output (never assumed)."""
    low = {int(i): str(name).lower() for i, name in id2label.items()}
    missing = [n for n in ORDER if n not in low.values()]
    if missing:
        raise ValueError(f"Model labels {sorted(low.values())} do not contain {missing}")
    inv = {name: i for i, name in low.items()}
    return [inv[n] for n in ORDER]


class FinBert:
    def __init__(self, tokenizer=None, model=None, device=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            if self.device == "cuda":
                model = model.half()
            print(f"Loaded {MODEL_NAME} on {self.device}; labels {model.config.id2label}")
        self.tok, self.model = tokenizer, model.to(self.device).eval()
        self.cols = label_columns(self.model.config.id2label)

    def predict_proba(self, texts, max_len, batch_size):
        """(n, 3) probabilities in the order positive, negative, neutral; same order as texts."""
        order = np.argsort([len(t) for t in texts], kind="stable")      # similar lengths -> faster
        out = np.zeros((len(texts), 3), dtype=np.float32)
        torch = self.torch
        with torch.no_grad():
            for s in range(0, len(texts), batch_size):
                idx = order[s:s + batch_size]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=max_len, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float()
                p = torch.softmax(logits, dim=-1).cpu().numpy()[:, self.cols]
                out[idx] = p
        return out


def score_chunks(chunks, scorer, cache_dir, block=BLOCK, log=print):
    """chunks: DataFrame (chunk_id, doc_type, text). Returns a DataFrame with the scores."""
    os.makedirs(cache_dir, exist_ok=True)
    group = np.where(chunks["doc_type"] == "news", "news", "filing")
    result = np.zeros((len(chunks), 3), dtype=np.float32)
    for g, (max_len, bs) in SETTINGS.items():
        rows = np.flatnonzero(group == g)
        ids, texts = chunks["chunk_id"].to_numpy()[rows], chunks["text"].to_numpy()[rows]
        for start in range(0, len(rows), block):
            sl = slice(start, start + block)
            key = hashlib.md5(("|".join(ids[sl]) + MODEL_NAME).encode()).hexdigest()[:12]
            path = os.path.join(cache_dir, f"sent_{g}_{start:07d}_{key}.npy")
            if os.path.exists(path):
                part = np.load(path)
            else:
                part = scorer.predict_proba(list(texts[sl]), max_len, bs)
                np.save(path, part)
            result[rows[sl]] = part
            log(f"  {g}: {min(start + block, len(rows))}/{len(rows)} done")
    out = pd.DataFrame({"chunk_id": chunks["chunk_id"].to_numpy(), "p_pos": result[:, 0],
                        "p_neg": result[:, 1], "p_neu": result[:, 2]})
    out["sent"] = out["p_pos"] - out["p_neg"]
    return out


def run(scorer=None):
    chunks = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "chunks.parquet"))
    scorer = scorer or FinBert()
    sc = score_chunks(chunks, scorer, os.path.join(config.RAW_DIR, "sentiment"))
    assert len(sc) == len(chunks) and sc["chunk_id"].is_unique
    sc.to_parquet(os.path.join(config.PROCESSED_DIR, "chunk_sentiment.parquet"), index=False)

    m = chunks[["chunk_id", "ticker", "doc_type", "effective_date", "url", "text"]].merge(sc, on="chunk_id")
    m["label"] = m[["p_pos", "p_neg", "p_neu"]].to_numpy().argmax(1)
    m["label"] = m["label"].map({0: "positive", 1: "negative", 2: "neutral"})
    print(f"\nScored {len(sc)} chunks. Any NaN: {bool(sc[['p_pos', 'p_neg', 'p_neu', 'sent']].isna().any().any())}")
    print("\nShare of chunks by predicted label:")
    print(m.groupby("doc_type")["label"].value_counts(normalize=True).unstack(fill_value=0).round(2).to_string())
    print("\nMean sentiment (sent) by ticker and type:")
    print(m.groupby(["doc_type", "ticker"])["sent"].mean().unstack().round(3).to_string())

    news = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "news.parquet"))[["url", "tone"]].drop_duplicates("url")
    mm = m[m["doc_type"] == "news"].merge(news, on="url")
    print(f"\nCorrelation with GDELT's own tone score (news, {len(mm)} rows): "
          f"{np.corrcoef(mm['sent'], mm['tone'])[0, 1]:.3f}  (expect clearly positive)")

    nws = m[m["doc_type"] == "news"].sort_values("sent")
    print("\n5 most NEGATIVE headlines:")
    for r in nws.head(5).itertuples():
        print(f"  {r.sent:+.2f} {r.ticker}: {r.text}")
    print("\n5 most POSITIVE headlines:")
    for r in nws.tail(5).itertuples():
        print(f"  {r.sent:+.2f} {r.ticker}: {r.text}")
    return sc