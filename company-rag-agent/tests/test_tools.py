from app.tools import CRMClient, Toolbox


def tb(fake_llm, index, crm):
    return Toolbox(fake_llm, index, crm)


def test_search_docs_assigns_stable_refs(fake_llm, index, crm):
    t = tb(fake_llm, index, crm)
    r1 = t.search_company_docs("refund annual plan")
    r2 = t.search_company_docs("refund annual plan")
    assert r1["results"][0]["ref"] == "S1"
    assert [x["ref"] for x in r1["results"]] == [x["ref"] for x in r2["results"]]
    assert len(t.last_hits) == len(r1["results"])


def test_search_docs_rejects_tiny_query(fake_llm, index, crm):
    assert "error" in tb(fake_llm, index, crm).search_company_docs("a")


def test_get_customer_by_email(fake_llm, index, crm):
    r = tb(fake_llm, index, crm).get_customer(email="SARA@acme-example.com")
    assert r["found"] and r["customer"]["plan"] == "enterprise"
    assert "phone" not in r["customer"]  # privacy: phone numbers are never shown to the model


def test_get_customer_by_company_and_id(fake_llm, index, crm):
    t = tb(fake_llm, index, crm)
    assert t.get_customer(name_or_company="karimi")["customer"]["id"] == "cus_1002"
    assert t.get_customer(customer_id="cus_1005")["customer"]["status"] == "past_due"


def test_get_customer_validation_and_not_found(fake_llm, index, crm):
    t = tb(fake_llm, index, crm)
    assert "error" in t.get_customer()
    assert "error" in t.get_customer(email="a@b.com", name_or_company="x")
    assert "error" in t.get_customer(email="not-an-email")
    assert "error" in t.get_customer(customer_id="12; DROP TABLE")
    assert t.get_customer(email="nobody@nowhere.com")["found"] is False
    assert t.get_customer(customer_id="cus_9999")["found"] is False


def test_ambiguous_match_returns_candidates(fake_llm, index, crm):
    r = tb(fake_llm, index, crm).get_customer(name_or_company="example")
    assert r["ambiguous"] and len(r["matches"]) >= 2


def test_list_customers_filters(fake_llm, index, crm):
    t = tb(fake_llm, index, crm)
    assert [c["id"] for c in t.list_customers(status="past_due")["customers"]] == ["cus_1005"]
    assert t.list_customers(plan="business")["count"] == 3
    assert "error" in t.list_customers(plan="gold")


def test_unknown_tool_and_bad_args(fake_llm, index, crm):
    t = tb(fake_llm, index, crm)
    assert "unknown tool" in t.run("delete_everything", {})["error"]
    assert "bad arguments" in t.run("get_customer", {"password": "x"})["error"]


def test_crm_down_returns_error_not_exception(fake_llm, index):
    t = Toolbox(fake_llm, index, CRMClient("http://127.0.0.1:1", "k", timeout=0.5))
    assert "unreachable" in t.get_customer(email="sara@acme-example.com")["error"]
