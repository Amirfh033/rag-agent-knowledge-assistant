import httpx
import pytest

from app.llm import LLMError, OpenAICompatClient, normalize_tool_calls


def test_standard_tool_calls_with_string_arguments():
    msg = {"tool_calls": [{"id": "a", "function": {"name": "get_customer", "arguments": '{"email": "x@y.com"}'}}]}
    assert normalize_tool_calls(msg) == [{"id": "a", "name": "get_customer", "arguments": {"email": "x@y.com"}}]


def test_dict_arguments_and_missing_id():
    msg = {"tool_calls": [{"function": {"name": "list_customers", "arguments": {"plan": "starter"}}}]}
    assert normalize_tool_calls(msg)[0] == {"id": "call_0", "name": "list_customers", "arguments": {"plan": "starter"}}


def test_tool_call_written_as_json_in_content():
    msg = {"content": '```json\n{"name": "search_company_docs", "arguments": {"query": "refund"}}\n```'}
    calls = normalize_tool_calls(msg)
    assert calls[0]["name"] == "search_company_docs" and msg["content"] == ""


def test_plain_answer_has_no_tool_calls():
    assert normalize_tool_calls({"content": "The Business plan costs €149 [S1]."}) == []


def test_broken_json_arguments_do_not_crash():
    msg = {"tool_calls": [{"id": "a", "function": {"name": "x", "arguments": "{not json"}}]}
    assert normalize_tool_calls(msg)[0]["arguments"] == {"_raw": "{not json"}


def _client(handler):
    c = OpenAICompatClient("http://llm.test/v1", "k", "chat", "emb", retries=2)
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    c.http.headers["Authorization"] = "Bearer k"
    return c


def test_retries_on_503_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    n = {"i": 0}
    def h(req):
        n["i"] += 1
        if n["i"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
    assert _client(h).chat([{"role": "user", "content": "x"}])["content"] == "hi"
    assert n["i"] == 3


def test_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    n = {"i": 0}
    def h(req):
        n["i"] += 1
        return httpx.Response(404, text="model not found")
    with pytest.raises(LLMError, match="404"):
        _client(h).chat([{"role": "user", "content": "x"}])
    assert n["i"] == 1


def test_embed_batches_and_keeps_order():
    def h(req):
        import json
        inp = json.loads(req.content)["input"]
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [len(t)]} for i, t in reversed(list(enumerate(inp)))]})
    assert _client(h).embed(["a", "bb", "ccc"], batch_size=2) == [[1], [2], [3]]
