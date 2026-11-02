"""Phase 3.1 (RAG): cut 8-K filing text and news headlines into small searchable "chunks".

Why chunks: a search index works best on short passages (a few paragraphs), not 100-page files.
Each chunk keeps WHO (ticker) and WHEN (effective_date) so that later retrieval can enforce the
no-look-ahead rule: for a prediction at the close of day t, only chunks with effective_date <= t
may be returned. effective_date is copied unchanged from the filing / news tables.

Chunk types (doc_type):
  exhibit : press-release text of an 8-K (EX-99.x), e.g. the earnings release
  main    : the body of the 8-K itself (cover page and signature block removed)
  news    : one news headline = one chunk
"""
import hashlib
import os
import re

import pandas as pd

import config

MAX_CHARS = 1200      # target upper size of a chunk
MIN_CHARS = 60        # smaller pieces are merged into the previous chunk

# SEC page-header lines at the very top of an exhibit ("EX-99.1", "2", "q3fy26pr.htm", ...)
_BOILER = re.compile(r"^(?:ex(?:hibit)?[-_ ]?\d+(?:\.\d+)?|\d{1,2}|document|[\w\-.]+\.html?|exhibit\s+\d+(?:\.\d+)?)$", re.I)
_ITEM = re.compile(r"^\s*item\s+\d\.\d{2}\b", re.I)
_SIGN = re.compile(r"^\s*signatures?\s*$", re.I)


def strip_exhibit_header(text, max_lines=10):
    """Remove SEC page-header lines ("EX-99.1", "2", "file.htm", "Document", "Exhibit 99.1")
    that appear among the first `max_lines` lines. Later lines are never touched."""
    lines = text.split("\n")
    head = [ln for ln in lines[:max_lines] if ln.strip() != "" and not _BOILER.match(ln.strip())]
    return "\n".join(head + lines[max_lines:]).strip()


def strip_main(text):
    """8-K body: drop the cover page (before the first 'Item x.xx') and the signature block."""
    lines = text.split("\n")
    start = next((k for k, ln in enumerate(lines) if _ITEM.match(ln)), 0)
    sig = [k for k, ln in enumerate(lines) if _SIGN.match(ln) and k > start]
    end = sig[-1] if sig else len(lines)
    return "\n".join(lines[start:end]).strip()


def _split_long(par, max_chars):
    """Split one over-long paragraph at sentence ends, then hard-cut if still too long."""
    pieces, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", par):
        while len(sent) > max_chars:
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if cur and len(cur) + 1 + len(sent) > max_chars:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_text(text, max_chars=MAX_CHARS, min_chars=MIN_CHARS):
    """Paragraph-based chunks of at most ~max_chars; nothing is dropped or repeated."""
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    units = []
    for p in paras:
        units.extend(_split_long(p, max_chars) if len(p) > max_chars else [p])
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > max_chars:
            chunks.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        chunks.append(cur)
    merged = []                                                # glue tiny pieces to the previous one
    for c in chunks:
        if merged and len(c) < min_chars:
            merged[-1] = f"{merged[-1]}\n{c}"
        else:
            merged.append(c)
    if len(merged) > 1 and len(merged[0]) < min_chars:         # tiny first piece -> glue to next
        merged[1] = f"{merged[0]}\n{merged[1]}"
        merged = merged[1:]
    return merged


def _cid(*parts):
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:16]


def filing_chunks(filings):
    """8-K rows (filing_text table) -> chunk rows."""
    rows = []
    for r in filings.itertuples(index=False):
        if isinstance(getattr(r, "error", None), str) and r.error:
            continue
        for doc_type, raw, clean in (("exhibit", r.exhibit_text, strip_exhibit_header),
                                     ("main", r.main_text, strip_main)):
            if not isinstance(raw, str) or not raw.strip():
                continue
            text = clean(raw)
            for k, c in enumerate(chunk_text(text)):
                rows.append({"chunk_id": _cid(r.accession, doc_type, k), "ticker": r.ticker,
                             "effective_date": pd.Timestamp(r.effective_date), "doc_type": doc_type,
                             "source_id": r.accession, "items": r.items, "url": r.url,
                             "chunk_index": k, "text": c})
    return pd.DataFrame(rows)


def news_chunks(news):
    """News rows -> one chunk per non-empty headline."""
    n = news[news["headline"].fillna("").str.strip() != ""]
    out = pd.DataFrame({
        "chunk_id": [_cid("news", u) for u in n["url"]],
        "ticker": n["ticker"].to_numpy(),
        "effective_date": pd.to_datetime(n["effective_date"]).to_numpy(),
        "doc_type": "news", "source_id": n["source"].to_numpy(), "items": None,
        "url": n["url"].to_numpy(), "chunk_index": 0, "text": n["headline"].str.strip().to_numpy()})
    return out.drop_duplicates("chunk_id").reset_index(drop=True)


def build(filings, news):
    """Pure function: both tables -> one chunk table."""
    allc = pd.concat([filing_chunks(filings), news_chunks(news)], ignore_index=True)
    allc["n_chars"] = allc["text"].str.len()
    return allc.sort_values(["ticker", "effective_date", "doc_type", "chunk_index"]).reset_index(drop=True)


def run():
    filings = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "filing_text.parquet"))
    news = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "news.parquet"))

    print("Exhibit header check (first 5 lines BEFORE -> AFTER cleaning) for 3 filings:")
    for r in filings[filings["exhibit_text"].fillna("") != ""].sample(3, random_state=1).itertuples():
        before = r.exhibit_text.split("\n")[:5]
        after = strip_exhibit_header(r.exhibit_text).split("\n")[:3]
        print(f"  {r.ticker} {r.accession}\n    BEFORE: {before}\n    AFTER : {after}")

    chunks = build(filings, news)
    out = os.path.join(config.PROCESSED_DIR, "chunks.parquet")
    chunks.to_parquet(out, index=False)

    print("\nChunks by type and ticker:")
    print(chunks.groupby(["doc_type", "ticker"]).size().unstack(fill_value=0).to_string())
    print("\nChunk length (characters) by type:")
    print(chunks.groupby("doc_type")["n_chars"].describe()[["count", "min", "50%", "max"]].round(0).to_string())
    print(f"\nUnique chunk ids: {chunks['chunk_id'].is_unique} | Saved {len(chunks)} chunks to {out}")
    for dt in ("exhibit", "main", "news"):
        s = chunks[chunks["doc_type"] == dt]
        if len(s):
            r = s.sample(1, random_state=3).iloc[0]
            print(f"\nSample {dt} chunk ({r['ticker']}, effective {str(r['effective_date'])[:10]}):\n{r['text'][:400]}")
    return chunks