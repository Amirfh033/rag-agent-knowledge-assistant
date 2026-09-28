"""Command line.
    python -m app.cli ingest              build the document index
    python -m app.cli search "query"      show retrieved chunks (no LLM answer)
    python -m app.cli ask "question"      one question, full trace
    python -m app.cli chat                interactive chat
"""
from __future__ import annotations

import json
import sys

from .llm import OpenAICompatClient
from .rag import VectorIndex
from .service import AssistantService


def _print_result(r) -> None:
    print("\n" + r.answer + "\n")
    if r.tool_calls:
        print("Tools used:")
        for t in r.tool_calls:
            print(f"  - {t['tool']}({json.dumps(t['arguments'], ensure_ascii=False)}) -> {'ok' if t['ok'] else 'error'}")
    if r.sources:
        print("Sources:")
        for s in r.sources:
            print(f"  [{s['ref']}] {s['source']} § {s['section']}  (score {s['score']})")
    print(f"({r.steps} step(s), {r.latency_ms} ms){'  ERROR: ' + r.error if r.error else ''}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], " ".join(argv[1:]).strip()
    if cmd == "ingest":
        llm = OpenAICompatClient()
        print(f"Indexing documents with {llm.embed_model} ...")
        idx = VectorIndex.build(llm)
        idx.save()
        print(f"Done: {idx.meta['n_chunks']} chunks from {len(idx.meta['files'])} files, "
              f"{idx.meta['dims']} dims, {idx.meta['embed_seconds']} s")
        return 0
    if cmd == "search":
        llm = OpenAICompatClient()
        for h in VectorIndex.load().search(llm, rest, k=5, min_score=0):
            print(f"{h.score:.3f} (vec {h.vector_score:.3f} / kw {h.keyword_score:.3f})  {h.chunk.citation()}")
            print("   " + h.chunk.text[:160].replace("\n", " ") + "…")
        return 0
    service = AssistantService()
    if cmd == "ask":
        _print_result(service.ask(rest)[1])
        return 0
    if cmd == "chat":
        sid = None
        print("Nimbus Flow assistant — type 'exit' to quit, 'reset' for a new conversation.")
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q in ("exit", "quit"):
                break
            if q == "reset":
                sid = None
                continue
            if q:
                sid, r = service.ask(q, sid)
                _print_result(r)
        return 0
    print(f"unknown command: {cmd}\n{__doc__}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
