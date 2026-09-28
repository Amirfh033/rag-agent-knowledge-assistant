from fastapi.testclient import TestClient

import app.api as api
from app.service import AssistantService
from conftest import FakeLLM, tool_call


def client(index, crm, script, tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.LOG_DIR", tmp_path)
    api._service = AssistantService(llm=FakeLLM(script), crm=crm, index=index)
    return TestClient(api.app)


def test_health_and_ui(index, crm, tmp_path, monkeypatch):
    c = client(index, crm, [], tmp_path, monkeypatch)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["documents"] == 6
    assert "Nimbus Flow Assistant" in c.get("/").text


def test_ask_roundtrip_session_and_log(index, crm, tmp_path, monkeypatch):
    c = client(index, crm, [tool_call("get_customer", customer_id="cus_1002"),
                            {"content": "Karimi Shop is on Starter (CRM) [S1]."},
                            {"content": "Follow-up answer."}], tmp_path, monkeypatch)
    r = c.post("/ask", json={"question": "What plan is Karimi Shop on?"}).json()
    assert r["answer"].startswith("Karimi Shop") and r["tool_calls"][0]["tool"] == "get_customer"
    sid = r["session_id"]
    r2 = c.post("/ask", json={"question": "And their renewal date?", "session_id": sid}).json()
    assert r2["session_id"] == sid
    assert len(api._service.sessions[sid]) == 4
    lines = (tmp_path / "queries.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2 and "get_customer" in lines[0]


def test_ask_validation(index, crm, tmp_path, monkeypatch):
    c = client(index, crm, [], tmp_path, monkeypatch)
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json={}).status_code == 422
