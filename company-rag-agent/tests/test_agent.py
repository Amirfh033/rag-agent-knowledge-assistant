from app.agent import Agent
from app.llm import LLMError
from app.tools import Toolbox
from conftest import FakeLLM, tool_call


def make(index, crm, script, **kw):
    llm = FakeLLM(script)
    return Agent(llm, Toolbox(llm, index, crm), **kw), llm


def test_answers_from_retrieved_docs_with_citation(index, crm):
    agent, llm = make(index, crm, [{"content": "Annual plans are refundable within 30 days [S1]."}])
    r = agent.ask("Can I get a refund on an annual plan?")
    assert r.error is None and r.steps == 1
    assert r.sources[0]["ref"] == "S1" and r.sources[0]["cited"]
    assert r.sources[0]["source"] == "02-refund-and-cancellation.md"
    user_msg = llm.calls[0]["messages"][-1]["content"]
    assert "[S1]" in user_msg and "30 days" in user_msg   # context was injected
    assert {t["function"]["name"] for t in llm.calls[0]["tools"]} == {"search_company_docs", "get_customer", "list_customers"}


def test_customer_plus_policy_question_uses_crm_tool(index, crm):
    agent, llm = make(index, crm, [
        tool_call("get_customer", name_or_company="Acme"),
        {"content": "Acme GmbH is on Enterprise (CRM), so it has 24/7 phone support [S1]."},
    ])
    r = agent.ask("Does Acme GmbH get phone support?")
    assert [t["tool"] for t in r.tool_calls] == ["get_customer"] and r.tool_calls[0]["ok"]
    assert r.steps == 2
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "enterprise" in tool_msg["content"]


def test_second_search_adds_new_refs(index, crm):
    agent, _ = make(index, crm, [
        tool_call("search_company_docs", query="API rate limit per minute"),
        {"content": "Business: 100 requests per minute [S5]."},
    ])
    r = agent.ask("How many calls can our integration make?")
    assert r.tool_calls[0]["tool"] == "search_company_docs"
    assert any(s["source"] == "06-integrations-and-limits.md" for s in r.sources)


def test_tool_errors_are_passed_back_to_model(index, crm):
    agent, llm = make(index, crm, [tool_call("get_customer", email="broken"), {"content": "Please give a valid email."}])
    r = agent.ask("Look up customer broken")
    assert r.tool_calls[0]["ok"] is False
    assert "email is not valid" in llm.calls[1]["messages"][-1]["content"]


def test_step_budget_forces_final_answer(index, crm):
    loop = [tool_call("list_customers") for _ in range(3)]
    agent, llm = make(index, crm, loop + [{"content": "Final answer."}], max_steps=3)
    r = agent.ask("Which customers exist?")
    assert r.answer == "Final answer." and len(r.tool_calls) == 3
    assert llm.calls[-1]["tools"] is None     # last call without tools


def test_llm_failure_returns_friendly_error(index, crm):
    agent, _ = make(index, crm, [LLMError("connection refused")])
    r = agent.ask("What is the SLA?")
    assert r.error and "temporarily unavailable" in r.answer


def test_input_validation(index, crm):
    agent, _ = make(index, crm, [])
    assert agent.ask("   ").error == "empty_question"
    assert agent.ask("x" * 2001).error == "too_long"


def test_history_is_included_but_limited(index, crm):
    agent, llm = make(index, crm, [{"content": "ok"}])
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(20)]
    agent.ask("And the Enterprise plan?", history)
    roles = [m["role"] for m in llm.calls[0]["messages"]]
    assert roles.count("user") + roles.count("assistant") == 7   # 6 history + current question
