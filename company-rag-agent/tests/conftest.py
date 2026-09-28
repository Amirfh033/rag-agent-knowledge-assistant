"""Shared fixtures: a deterministic fake LLM, a built index, and a live mock CRM."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag import VectorIndex  # noqa: E402

_SYN = {"cost": "price", "costs": "price", "pricing": "price", "priced": "price", "money back": "refund",
        "stored": "store", "storage": "store", "located": "store", "reimburse": "refund", "refunds": "refund"}


class FakeLLM:
    """Embeddings: hashed bag-of-words (deterministic, lexical). Chat: scripted replies."""
    embed_model = "fake-embed"
    chat_model = "fake-chat"

    def __init__(self, script: list[dict] | None = None, dims: int = 512):
        self.dims, self.script, self.calls = dims, list(script or []), []

    def embed(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            for a, b in _SYN.items():
                t = t.replace(a, b)
            v = np.zeros(self.dims)
            for w in re.findall(r"[a-z0-9]+", t):
                if len(w) > 2:
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims] += 1
            out.append(v.tolist())
        return out

    def chat(self, messages, tools=None, temperature=0.0):
        self.calls.append({"messages": messages, "tools": tools})
        if not self.script:
            return {"role": "assistant", "content": "Default answer [S1].", "tool_calls": []}
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return {"role": "assistant", "content": step.get("content", ""), "tool_calls": step.get("tool_calls", [])}


def tool_call(name, **args):
    return {"content": "", "tool_calls": [{"id": f"c_{name}", "name": name, "arguments": args}]}


@pytest.fixture(scope="session")
def fake_llm():
    return FakeLLM()


@pytest.fixture(scope="session")
def index(fake_llm):
    return VectorIndex.build(fake_llm, ROOT / "data" / "docs", log=lambda *_: None)


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="session")
def crm_url():
    if not shutil.which("node"):
        pytest.skip("node is required for CRM integration tests")
    port = _free_port()
    db = Path(tempfile.mkdtemp()) / "db.json"
    env = {**os.environ, "PORT": str(port), "API_KEY": "test-key", "DB_FILE": str(db)}
    proc = subprocess.Popen(["node", str(ROOT / "mock-crm" / "server.js")], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    import httpx
    for _ in range(50):
        try:
            if httpx.get(url + "/health").status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    yield url
    proc.terminate()


@pytest.fixture
def crm(crm_url):
    from app.tools import CRMClient
    return CRMClient(crm_url, "test-key")
