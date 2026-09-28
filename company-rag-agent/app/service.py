"""Wires the LLM client, index, tools and agent together, and logs every query."""
from __future__ import annotations

import json
import threading
import time
import uuid

from . import config
from .agent import Agent, AgentResult
from .llm import OpenAICompatClient
from .rag import VectorIndex
from .tools import CRMClient, Toolbox


class AssistantService:
    def __init__(self, llm=None, crm=None, index: VectorIndex | None = None):
        self.llm = llm or OpenAICompatClient()
        self.crm = crm or CRMClient()
        self.index = index or VectorIndex.load()
        self.sessions: dict[str, list[dict]] = {}
        self.lock = threading.Lock()   # one question at a time per process (toolbox has per-question state)

    def ask(self, question: str, session_id: str | None = None) -> tuple[str, AgentResult]:
        session_id = session_id or uuid.uuid4().hex[:12]
        history = self.sessions.get(session_id, [])
        with self.lock:
            agent = Agent(self.llm, Toolbox(self.llm, self.index, self.crm))
            result = agent.ask(question, history)
        if not result.error:
            history = (history + [{"role": "user", "content": question},
                                  {"role": "assistant", "content": result.answer}])[-10:]
            self.sessions[session_id] = history
        self._log(session_id, question, result)
        return session_id, result

    def reindex(self) -> dict:
        idx = VectorIndex.build(self.llm, log=lambda *_: None)
        idx.save()
        self.index = idx
        return idx.meta

    def _log(self, session_id: str, question: str, r: AgentResult) -> None:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": session_id, "question": question,
                 "answer": r.answer, "sources": [s["source"] for s in r.sources],
                 "tools": [t["tool"] for t in r.tool_calls], "steps": r.steps,
                 "latency_ms": r.latency_ms, "error": r.error}
        with open(config.LOG_DIR / "queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
