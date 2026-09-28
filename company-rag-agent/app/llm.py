"""LLM client for any OpenAI-compatible API (Ollama /v1, OpenAI, OpenRouter, Groq...).

Two operations are needed:
  * chat(messages, tools)  -> assistant message (content and/or tool_calls)
  * embed(texts)           -> list of vectors
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from . import config


class LLMError(RuntimeError):
    pass


class OpenAICompatClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 chat_model: str | None = None, embed_model: str | None = None,
                 timeout: float | None = None, retries: int = 2):
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.chat_model = chat_model or config.CHAT_MODEL
        self.embed_model = embed_model or config.EMBED_MODEL
        self.retries = retries
        self.http = httpx.Client(
            timeout=timeout or config.LLM_TIMEOUT,
            headers={"Authorization": f"Bearer {api_key or config.LLM_API_KEY}"},
        )

    def _post(self, path: str, payload: dict) -> dict:
        last = ""
        for attempt in range(self.retries + 1):
            try:
                r = self.http.post(f"{self.base_url}{path}", json=payload)
            except httpx.HTTPError as e:          # connection error / timeout -> retry
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code < 400:
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code < 500 and r.status_code != 429:
                    break                          # client error -> retrying won't help
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM request to {path} failed: {last}")

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.0) -> dict:
        payload: dict[str, Any] = {"model": self.chat_model, "messages": messages,
                                   "temperature": temperature}
        if tools:
            payload["tools"] = tools
        data = self._post("/chat/completions", payload)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError):
            raise LLMError(f"unexpected chat response: {str(data)[:300]}")
        msg["tool_calls"] = normalize_tool_calls(msg)
        return msg

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            data = self._post("/embeddings", {"model": self.embed_model, "input": batch})
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            out.extend(d["embedding"] for d in items)
        return out


def normalize_tool_calls(msg: dict) -> list[dict]:
    """Return tool calls as [{'id', 'name', 'arguments': dict}].

    Handles the standard `tool_calls` field and a common small-model habit of
    writing the call as JSON inside `content` instead.
    """
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", tc)
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name"), "arguments": args})
    if calls:
        return calls

    content = (msg.get("content") or "").strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    if content.startswith("{") and '"name"' in content:
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                args = obj.get("arguments") or obj.get("parameters") or {}
                msg["content"] = ""
                return [{"id": "call_0", "name": obj["name"], "arguments": args}]
        except json.JSONDecodeError:
            pass
    return []
