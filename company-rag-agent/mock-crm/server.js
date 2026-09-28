// Mock CRM API — zero dependencies (Node 18+).
// Run:  node server.js        (defaults: PORT=4000, API_KEY=dev-secret-key)
// Simulate an unreliable CRM to test n8n retries:  FAIL_RATE=0.3 node server.js

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const PORT = Number(process.env.PORT || 4000);
const API_KEY = process.env.API_KEY || 'dev-secret-key';
const FAIL_RATE = Number(process.env.FAIL_RATE || 0);
const DB_FILE = process.env.DB_FILE || path.join(__dirname, 'data', 'db.json');

const CATEGORIES = ['sales_lead', 'support_request', 'partnership', 'general_inquiry', 'spam'];
const PRIORITIES = ['high', 'medium', 'low'];

// ---------- tiny JSON "database" ----------
function loadDb() {
  try { return JSON.parse(fs.readFileSync(DB_FILE, 'utf8')); }
  catch { return { customers: [], logs: [], notifications: [] }; }
}
function saveDb(db) {
  fs.mkdirSync(path.dirname(DB_FILE), { recursive: true });
  fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2));
}
let db = loadDb();
// Seed demo customers on first start (Project 2 needs data to look up)
const SEED_FILE = process.env.SEED_FILE || path.join(__dirname, 'seed.json');
if (db.customers.length === 0 && fs.existsSync(SEED_FILE)) {
  db.customers = JSON.parse(fs.readFileSync(SEED_FILE, 'utf8'));
  saveDb(db);
  console.log(`Seeded ${db.customers.length} customers from ${SEED_FILE}`);
}

// ---------- helpers ----------
function send(res, status, data) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Allow-Methods': 'GET, POST, PATCH, OPTIONS',
  });
  res.end(data === undefined ? '' : JSON.stringify(data));
}
function readBody(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', (c) => { raw += c; if (raw.length > 1e6) req.destroy(); });
    req.on('end', () => {
      if (!raw) return resolve({});
      try { resolve(JSON.parse(raw)); } catch { reject(new Error('invalid JSON')); }
    });
  });
}
const authorized = (req) => req.headers.authorization === `Bearer ${API_KEY}`;
const newId = (p) => `${p}_${crypto.randomBytes(5).toString('hex')}`;

function validateCustomer(b) {
  const e = [];
  if (!b.full_name || typeof b.full_name !== 'string') e.push('full_name is required');
  if (!b.email || !/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(b.email)) e.push('valid email is required');
  if (!CATEGORIES.includes(b.category)) e.push(`category must be one of ${CATEGORIES.join(', ')}`);
  if (!PRIORITIES.includes(b.priority)) e.push(`priority must be one of ${PRIORITIES.join(', ')}`);
  if (b.summary && b.summary.length > 200) e.push('summary max 200 chars');
  return e;
}

// ---------- routes ----------
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const p = url.pathname.replace(/\/+$/, '') || '/';
  const started = Date.now();
  res.on('finish', () =>
    console.log(`${new Date().toISOString()} ${req.method} ${p} -> ${res.statusCode} (${Date.now() - started}ms)`));

  if (req.method === 'OPTIONS') return send(res, 204);
  if (p === '/health') return send(res, 200, { status: 'ok', customers: db.customers.length });

  if (!p.startsWith('/api/')) return send(res, 404, { error: 'not_found' });
  if (!authorized(req)) return send(res, 401, { error: 'unauthorized', message: 'Use Authorization: Bearer <API_KEY>' });

  // Simulated outage (only for write operations on customers)
  if (FAIL_RATE > 0 && req.method === 'POST' && p === '/api/customers' && Math.random() < FAIL_RATE)
    return send(res, 503, { error: 'service_unavailable', message: 'Simulated CRM outage — retry later' });

  let body = {};
  if (['POST', 'PATCH'].includes(req.method)) {
    try { body = await readBody(req); } catch { return send(res, 400, { error: 'invalid_json' }); }
  }

  // POST /api/customers — create (idempotent on request_id, conflict on email)
  if (req.method === 'POST' && p === '/api/customers') {
    const errors = validateCustomer(body);
    if (errors.length) return send(res, 422, { error: 'validation_failed', details: errors });

    if (body.request_id) {
      const dup = db.customers.find((c) => c.request_id === body.request_id);
      if (dup) return send(res, 200, { ...dup, idempotent_replay: true });
    }
    const email = body.email.toLowerCase();
    const existing = db.customers.find((c) => c.email === email);
    if (existing) return send(res, 409, { error: 'duplicate_email', existing_id: existing.id });

    const now = new Date().toISOString();
    const customer = {
      id: newId('cus'),
      request_id: body.request_id || null,
      full_name: body.full_name,
      email,
      phone: body.phone || null,
      company: body.company || null,
      company_size: body.company_size || null,
      source: body.source || 'api',
      message: body.message || null,
      category: body.category,
      priority: body.priority,
      summary: body.summary || null,
      ai_confidence: body.ai_confidence ?? null,
      status: body.status || 'new',
      created_at: now,
      updated_at: now,
    };
    db.customers.push(customer);
    saveDb(db);
    return send(res, 201, customer);
  }

  // GET /api/customers?category=&priority=&email=
  if (req.method === 'GET' && p === '/api/customers') {
    let list = db.customers;
    const q = (url.searchParams.get('q') || '').toLowerCase().trim();
    if (q) list = list.filter((c) =>
      [c.full_name, c.company, c.email, c.workspace_id].some((v) => (v || '').toLowerCase().includes(q)));
    for (const k of ['category', 'priority', 'email', 'status', 'plan']) {
      const v = url.searchParams.get(k);
      if (v) list = list.filter((c) => c[k] === v.toLowerCase());
    }
    return send(res, 200, { count: list.length, data: list });
  }

  // GET / PATCH /api/customers/:id
  const m = p.match(/^\/api\/customers\/([\w-]+)$/);
  if (m) {
    const c = db.customers.find((x) => x.id === m[1]);
    if (!c) return send(res, 404, { error: 'customer_not_found' });
    if (req.method === 'GET') return send(res, 200, c);
    if (req.method === 'PATCH') {
      for (const k of ['status', 'priority', 'category', 'summary', 'phone', 'company']) {
        if (body[k] !== undefined) c[k] = body[k];
      }
      const errs = validateCustomer(c);
      if (errs.length) { db = loadDb(); return send(res, 422, { error: 'validation_failed', details: errs }); }
      c.updated_at = new Date().toISOString();
      saveDb(db);
      return send(res, 200, c);
    }
  }

  // Execution log store
  if (p === '/api/logs') {
    if (req.method === 'POST') {
      const entry = { id: newId('log'), logged_at: new Date().toISOString(), ...body };
      db.logs.push(entry); saveDb(db);
      return send(res, 201, entry);
    }
    if (req.method === 'GET') {
      const status = url.searchParams.get('status');
      const list = status ? db.logs.filter((l) => l.status === status) : db.logs;
      return send(res, 200, { count: list.length, data: list.slice(-200).reverse() });
    }
  }

  // Mock notification channel (stands in for Slack / Telegram / email)
  if (p === '/api/notifications') {
    if (req.method === 'POST') {
      const n = { id: newId('ntf'), sent_at: new Date().toISOString(), ...body };
      db.notifications.push(n); saveDb(db);
      console.log(`🔔 NOTIFICATION [${body.level || 'info'}] ${body.text || ''}`);
      return send(res, 201, n);
    }
    if (req.method === 'GET') return send(res, 200, { count: db.notifications.length, data: db.notifications.slice(-100).reverse() });
  }

  return send(res, 404, { error: 'not_found' });
});

if (require.main === module) {
  server.listen(PORT, () => {
    console.log(`Mock CRM listening on http://localhost:${PORT}`);
    console.log(`API key: ${API_KEY}  |  FAIL_RATE: ${FAIL_RATE}  |  DB: ${DB_FILE}`);
  });
}
module.exports = server;
