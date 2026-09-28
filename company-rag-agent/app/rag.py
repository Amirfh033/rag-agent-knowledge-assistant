"""RAG core: load documents, chunk them, embed them, and search with hybrid scoring.

Index layout (data/index/):
  chunks.json  - chunk text + metadata
  vectors.npy  - normalized embedding matrix (n_chunks x dim)
  meta.json    - embedding model, dimensions, build time, source file hashes
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config

SUPPORTED = {".md", ".txt", ".pdf"}
_TOKEN = re.compile(r"[a-z0-9؀-ۿ]+(?:[.,][0-9]+)?", re.I)
_STOP = set("a an the and or of to in on for is are be by with at as it this that from can "
            "what how do does i we you our your my me when which who".split())


@dataclass
class Chunk:
    id: str
    source: str      # file name
    title: str       # document title (first H1 or file name)
    section: str     # heading path, e.g. "Plans > Business"
    text: str

    def embed_text(self) -> str:
        return f"{self.title} > {self.section}\n{self.text}"

    def citation(self) -> str:
        return f"{self.source} § {self.section}" if self.section else self.source


# ---------------------------------------------------------------- loading
def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- chunking
def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split markdown into (heading_path, body) sections. Returns (title, sections)."""
    title, stack, sections, buf = "", [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(h for _, h in stack), body))
        buf.clear()

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = heading
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buf.append(line)
    flush()
    return title, sections


def split_long(body: str, size: int, overlap: int) -> list[str]:
    """Split text on paragraph / sentence boundaries into pieces of ~size chars with overlap."""
    if len(body) <= size:
        return [body]
    parts = re.split(r"(?<=\n\n)|(?<=[.!?])\s+", body)
    pieces, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > size:
            pieces.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur += (" " if cur and not cur.endswith("\n") else "") + p
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def chunk_document(path: Path, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    text = read_document(path)
    title, sections = split_sections(text)
    title = title or path.stem
    if not sections:
        sections = [("", text.strip())]

    # merge tiny sections into the previous one so chunks carry enough context
    merged: list[tuple[str, str]] = []
    for sec, body in sections:
        if merged and len(body) < 120 and len(merged[-1][1]) + len(body) < size:
            psec, pbody = merged[-1]
            merged[-1] = (psec, f"{pbody}\n\n{sec.split(' > ')[-1]}:\n{body}")
        else:
            merged.append((sec, body))

    chunks = []
    for sec, body in merged:
        for j, piece in enumerate(split_long(body, size, overlap)):
            cid = hashlib.sha1(f"{path.name}|{sec}|{j}|{piece}".encode()).hexdigest()[:12]
            chunks.append(Chunk(cid, path.name, title, sec, piece))
    return chunks


# ---------------------------------------------------------------- keyword (BM25)
def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text) if t.lower() not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.tfs))
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1)))
            out[i] = s
        return out


# ---------------------------------------------------------------- embeddings helpers
def _prefixes(model: str) -> tuple[str, str]:
    # nomic-embed-text is trained with task prefixes and retrieves better with them
    return ("search_query: ", "search_document: ") if "nomic" in model else ("", "")


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


# ---------------------------------------------------------------- index
@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    vector_score: float
    keyword_score: float


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict):
        self.chunks, self.vectors, self.meta = chunks, vectors, meta
        self.bm25 = BM25([c.embed_text() for c in chunks])

    # -- build / persist ------------------------------------------------
    @classmethod
    def build(cls, llm, docs_dir: Path | None = None, log=print) -> "VectorIndex":
        docs_dir = docs_dir or config.DOCS_DIR
        files = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED)
        if not files:
            raise FileNotFoundError(f"No .md/.txt/.pdf files in {docs_dir}")
        chunks: list[Chunk] = []
        for f in files:
            cs = chunk_document(f)
            log(f"  {f.name}: {len(cs)} chunks")
            chunks.extend(cs)
        _, doc_prefix = _prefixes(llm.embed_model)
        t0 = time.time()
        vecs = np.array(llm.embed([doc_prefix + c.embed_text() for c in chunks]), dtype=np.float32)
        meta = {
            "embed_model": llm.embed_model,
            "dims": int(vecs.shape[1]),
            "n_chunks": len(chunks),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embed_seconds": round(time.time() - t0, 2),
            "files": {f.name: hashlib.sha1(f.read_bytes()).hexdigest()[:12] for f in files},
        }
        return cls(chunks, _normalize(vecs), meta)

    def save(self, index_dir: Path | None = None) -> None:
        d = index_dir or config.INDEX_DIR
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=1), "utf-8")
        np.save(d / "vectors.npy", self.vectors)
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2), "utf-8")

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "VectorIndex":
        d = index_dir or config.INDEX_DIR
        if not (d / "chunks.json").exists():
            raise FileNotFoundError(f"No index in {d}. Run: python -m app.cli ingest")
        chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text("utf-8"))]
        return cls(chunks, np.load(d / "vectors.npy"), json.loads((d / "meta.json").read_text("utf-8")))

    # -- search -----------------------------------------------------------
    def search(self, llm, query: str, k: int | None = None, min_score: float | None = None,
               alpha: float | None = None) -> list[SearchHit]:
        k = k or config.TOP_K
        min_score = config.MIN_SCORE if min_score is None else min_score
        alpha = config.HYBRID_ALPHA if alpha is None else alpha
        if llm.embed_model != self.meta.get("embed_model"):
            raise ValueError(f"Index was built with {self.meta.get('embed_model')}, "
                             f"but the client uses {llm.embed_model}. Re-run ingest.")
        q_prefix, _ = _prefixes(llm.embed_model)
        qv = _normalize(np.array(llm.embed([q_prefix + query]), dtype=np.float32))[0]
        vec = self.vectors @ qv
        kw = self.bm25.scores(query)
        kw_n = kw / kw.max() if kw.max() > 0 else kw
        final = alpha * vec + (1 - alpha) * kw_n
        order = np.argsort(-final)[:k]
        return [SearchHit(self.chunks[i], float(final[i]), float(vec[i]), float(kw_n[i]))
                for i in order if final[i] >= min_score]
