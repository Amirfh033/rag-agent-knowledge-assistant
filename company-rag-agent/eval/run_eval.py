"""Evaluate retrieval and end-to-end answers against eval/questions.json.

    python eval/run_eval.py --retrieval-only    # fast: only embeddings, no chat model
    python eval/run_eval.py                     # full: agent answers (needs chat model + CRM)

Writes eval/results.json and eval/results.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.llm import OpenAICompatClient  # noqa: E402
from app.rag import VectorIndex  # noqa: E402

HERE = Path(__file__).resolve().parent


def retrieval_eval(llm, index, questions, k):
    rows = []
    for q in questions:
        if not q.get("expected_source"):
            continue
        hits = index.search(llm, q["question"], k=k, min_score=0)
        sources = [h.chunk.source for h in hits]
        rank = sources.index(q["expected_source"]) + 1 if q["expected_source"] in sources else None
        rows.append({"id": q["id"], "question": q["question"], "expected": q["expected_source"],
                     "top1": sources[0] if sources else None, "rank": rank,
                     "top_score": round(hits[0].score, 3) if hits else 0})
    n = len(rows) or 1
    summary = {"questions": len(rows),
               "hit@1": round(sum(r["rank"] == 1 for r in rows) / n, 3),
               f"hit@{k}": round(sum(r["rank"] is not None for r in rows) / n, 3),
               "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3)}
    return summary, rows


def answer_eval(service, questions):
    rows = []
    for q in questions:
        _, r = service.ask(q["question"])
        tools = [t["tool"] for t in r.tool_calls]
        srcs = [s["source"] for s in r.sources]
        ans_ok = any(kw.lower() in r.answer.lower() for kw in q["must_include"])
        tool_ok = (q["expected_tool"] in tools) if q.get("expected_tool") else True
        src_ok = (q["expected_source"] in srcs) if q.get("expected_source") else True
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"], "answer": r.answer,
                     "tools": tools, "sources": srcs, "answer_ok": ans_ok, "tool_ok": tool_ok,
                     "source_ok": src_ok, "pass": ans_ok and tool_ok and src_ok and not r.error,
                     "latency_ms": r.latency_ms, "error": r.error})
        mark = "PASS" if rows[-1]["pass"] else "FAIL"
        print(f"  {mark}  {q['id']}  {r.latency_ms/1000:5.1f}s  tools={tools}  {r.answer[:70]!r}")
    n = len(rows) or 1
    lat = sorted(r["latency_ms"] for r in rows)
    summary = {"questions": len(rows), "pass_rate": round(sum(r["pass"] for r in rows) / n, 3),
               "answer_accuracy": round(sum(r["answer_ok"] for r in rows) / n, 3),
               "tool_accuracy": round(sum(r["tool_ok"] for r in rows) / n, 3),
               "citation_accuracy": round(sum(r["source_ok"] for r in rows) / n, 3),
               "median_latency_s": round(lat[len(lat) // 2] / 1000, 1) if lat else 0,
               "by_type": {t: f"{sum(r['pass'] for r in rows if r['type']==t)}/{sum(1 for r in rows if r['type']==t)}"
                           for t in dict.fromkeys(r["type"] for r in rows)}}
    return summary, rows


def write_markdown(meta, ret, ans):
    L = [f"# Evaluation results\n", f"- Date: {meta['date']}", f"- Chat model: `{meta['chat_model']}`",
         f"- Embedding model: `{meta['embed_model']}`", f"- Chunks: {meta['chunks']}\n",
         "## Retrieval\n", "| Metric | Value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in ret[0].items()]
    L += ["", "| ID | Question | Expected | Rank |", "|---|---|---|---|"]
    L += [f"| {r['id']} | {r['question']} | {r['expected']} | {r['rank'] or '✗'} |" for r in ret[1]]
    if ans:
        L += ["", "## End-to-end answers\n", "| Metric | Value |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in ans[0].items()]
        L += ["", "| ID | Type | Pass | Tools | Latency | Answer |", "|---|---|---|---|---|---|"]
        for r in ans[1]:
            a = r["answer"].replace("\n", " ").replace("|", "/")
            L.append(f"| {r['id']} | {r['type']} | {'✅' if r['pass'] else '❌'} | {', '.join(r['tools']) or '—'} "
                     f"| {r['latency_ms']/1000:.1f}s | {a[:160]}{'…' if len(a) > 160 else ''} |")
    (HERE / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--only", help="comma-separated question ids")
    args = ap.parse_args()

    questions = json.loads((HERE / "questions.json").read_text("utf-8"))
    if args.only:
        keep = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in keep]
    llm = OpenAICompatClient()
    index = VectorIndex.load()
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "chat_model": llm.chat_model,
            "embed_model": llm.embed_model, "chunks": len(index.chunks)}

    print("Retrieval evaluation ...")
    ret = retrieval_eval(llm, index, questions, args.k)
    print("  ", ret[0])
    ans = None
    if not args.retrieval_only:
        from app.service import AssistantService
        print("End-to-end evaluation ...")
        ans = answer_eval(AssistantService(llm=llm, index=index), questions)
        print("  ", ans[0])
    (HERE / "results.json").write_text(json.dumps({"meta": meta, "retrieval": ret, "answers": ans},
                                                  indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(meta, ret, ans)
    print(f"Saved {HERE / 'results.md'}")


if __name__ == "__main__":
    main()
