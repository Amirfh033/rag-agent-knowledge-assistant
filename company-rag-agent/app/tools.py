"""Tools the agent can call. Each tool validates its arguments and returns a
compact, JSON-serializable result (errors are returned, not raised, so the
model can react to them)."""
from __future__ import annotations

import re
from typing import Any, Callable

import httpx

from . import config
from .rag import VectorIndex

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$")
ID_RE = re.compile(r"^cus_[A-Za-z0-9]+$")

# Only these customer fields are shown to the model (no phone numbers).
CUSTOMER_FIELDS = ["id", "full_name", "email", "company", "company_size", "plan", "workspace_id",
                   "status", "renewal_date", "mrr_eur", "open_tickets", "last_invoice", "notes"]


class CRMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 10):
        self.base = (base_url or config.CRM_BASE_URL).rstrip("/")
        self.http = httpx.Client(timeout=timeout,
                                 headers={"Authorization": f"Bearer {api_key or config.CRM_API_KEY}"})

    def _get(self, path: str, params: dict | None = None) -> tuple[int, Any]:
        last = None
        for attempt in range(3):
            try:
                r = self.http.get(self.base + path, params=params)
                if r.status_code >= 500:
                    last = f"CRM returned {r.status_code}"
                    continue
                return r.status_code, r.json()
            except httpx.HTTPError as e:
                last = f"CRM unreachable: {type(e).__name__}"
        return 503, {"error": last}

    def by_id(self, cid: str):
        return self._get(f"/api/customers/{cid}")

    def search(self, **params):
        return self._get("/api/customers", {k: v for k, v in params.items() if v})


def _compact(c: dict) -> dict:
    return {k: c.get(k) for k in CUSTOMER_FIELDS if c.get(k) not in (None, "", [])}


class Toolbox:
    def __init__(self, llm, index: VectorIndex, crm: CRMClient | None = None):
        self.llm, self.index, self.crm = llm, index, crm or CRMClient()
        self.reset()
        self.handlers: dict[str, Callable[..., dict]] = {
            "search_company_docs": self.search_company_docs,
            "get_customer": self.get_customer,
            "list_customers": self.list_customers,
        }

    def reset(self) -> None:
        """Clear per-question state (retrieved chunks and their [S#] labels)."""
        self.last_hits = []
        self.refs: dict[str, str] = {}

    # ------------------------------------------------------------ schemas
    @property
    def schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": "search_company_docs",
                "description": "Search Nimbus Flow's internal documents (pricing, refunds, support SLA, "
                               "security, onboarding, integrations). Use for any question about company "
                               "policies, prices, limits or procedures.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "Search query in English"}},
                    "required": ["query"]}}},
            {"type": "function", "function": {
                "name": "get_customer",
                "description": "Look up ONE customer in the CRM. Provide exactly one of: email, "
                               "customer_id (cus_...), or name_or_company. Returns plan, status, "
                               "renewal date, open tickets and last invoice.",
                "parameters": {"type": "object", "properties": {
                    "email": {"type": "string"},
                    "customer_id": {"type": "string"},
                    "name_or_company": {"type": "string"}}}}},
            {"type": "function", "function": {
                "name": "list_customers",
                "description": "List customers filtered by plan (starter, business, enterprise) and/or "
                               "status (active, trial, past_due).",
                "parameters": {"type": "object", "properties": {
                    "plan": {"type": "string", "enum": ["starter", "business", "enterprise"]},
                    "status": {"type": "string", "enum": ["active", "trial", "past_due"]}}}}},
        ]

    def run(self, name: str, args: dict) -> dict:
        fn = self.handlers.get(name)
        if not fn:
            return {"error": f"unknown tool '{name}'. Available: {', '.join(self.handlers)}"}
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}
        try:
            return fn(**{k: v for k, v in args.items() if not k.startswith("_")})
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}

    # ------------------------------------------------------------ tools
    def search_company_docs(self, query: str = "") -> dict:
        query = str(query).strip()
        if len(query) < 3:
            return {"error": "query must be at least 3 characters"}
        hits = self.index.search(self.llm, query)
        for h in hits:
            if h.chunk.id not in self.refs:
                self.refs[h.chunk.id] = f"S{len(self.refs) + 1}"
                self.last_hits.append(h)
        if not hits:
            return {"results": [], "note": "No relevant document found."}
        return {"results": [{"ref": self.refs[h.chunk.id], "source": h.chunk.citation(),
                             "score": round(h.score, 3), "text": h.chunk.text} for h in hits]}

    def get_customer(self, email: str = "", customer_id: str = "", name_or_company: str = "") -> dict:
        email, customer_id, name_or_company = (str(x or "").strip() for x in (email, customer_id, name_or_company))
        given = [x for x in (email, customer_id, name_or_company) if x]
        if len(given) != 1:
            return {"error": "provide exactly one of email, customer_id, name_or_company"}
        if customer_id:
            if not ID_RE.match(customer_id):
                return {"error": "customer_id must look like cus_1234"}
            code, data = self.crm.by_id(customer_id)
            if code == 404:
                return {"found": False, "message": f"No customer with id {customer_id}"}
            return {"found": True, "customer": _compact(data)} if code == 200 else {"error": data.get("error")}
        if email:
            if not EMAIL_RE.match(email):
                return {"error": "email is not valid"}
            code, data = self.crm.search(email=email.lower())
        else:
            if len(name_or_company) < 2:
                return {"error": "name_or_company too short"}
            code, data = self.crm.search(q=name_or_company)
        if code != 200:
            return {"error": data.get("error", f"CRM error {code}")}
        rows = data.get("data", [])
        if not rows:
            return {"found": False, "message": "No matching customer"}
        if len(rows) > 1:
            return {"found": True, "ambiguous": True,
                    "matches": [{"id": r["id"], "full_name": r.get("full_name"), "company": r.get("company")} for r in rows[:5]]}
        return {"found": True, "customer": _compact(rows[0])}

    def list_customers(self, plan: str = "", status: str = "") -> dict:
        if plan and plan not in {"starter", "business", "enterprise"}:
            return {"error": "plan must be starter, business or enterprise"}
        if status and status not in {"active", "trial", "past_due"}:
            return {"error": "status must be active, trial or past_due"}
        code, data = self.crm.search(plan=plan, status=status)
        if code != 200:
            return {"error": data.get("error", f"CRM error {code}")}
        return {"count": data.get("count", 0), "customers": [
            {k: r.get(k) for k in ("id", "full_name", "company", "plan", "status", "renewal_date")}
            for r in data.get("data", [])[:20]]}
