"use strict";
// FPL Manager dashboard. All upstream text (player names, news) is inserted with textContent.

const POS_ORDER = { GKP: 0, DEF: 1, MID: 2, FWD: 3 };
const CHIP_LABEL = { "3xc": "Triple Captain", bboost: "Bench Boost", freehit: "Free Hit", wildcard: "Wildcard" };
const PAGE = 25;

const state = { data: null, byId: new Map(), sort: { key: "total", dir: -1 }, pos: "", q: "", maxPrice: null, mine: false, limit: PAGE };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const money = (tenths) => `£${(tenths / 10).toFixed(1)}m`;

function banner(kind, text) {
  const b = $("banner");
  b.hidden = !text;
  b.className = `banner ${kind}`;
  b.textContent = text || "";
}

async function load() {
  const res = await fetch("/api/v1/dashboard");
  if (!res.ok) throw new Error(`dashboard request failed (${res.status})`);
  state.data = await res.json();
  state.byId = new Map(state.data.players.map((p) => [p.player_id, p]));
  render();
}

function render() {
  renderHeader();
  renderPlan();
  renderAutopilot();
  renderTableHead();
  renderTable();
  renderJobs();
}

// ---------------------------------------------------------------- header + countdown
function renderHeader() {
  const { team, prediction_run: run } = state.data;
  $("team-line").textContent = team
    ? `${team.entry_name} · rank ${Number(team.summary_overall_rank).toLocaleString()} · ${team.summary_overall_points} pts`
    : "Set FPL_TEAM_ID to see your team";
  $("model-meta").textContent = run ? `xP ${run.model_version} · run ${new Date(run.run_timestamp).toLocaleString()}` : "no xP run yet";
  tick();
}

function tick() {
  const gw = state.data && state.data.next_gameweek;
  const box = $("deadline");
  if (!gw) { box.textContent = "No upcoming deadline"; return; }
  const ms = new Date(gw.deadline_time) - Date.now();
  box.replaceChildren();
  if (ms <= 0) { box.append(`${gw.name} deadline passed`); return; }
  const h = Math.floor(ms / 3.6e6), m = Math.floor((ms % 3.6e6) / 6e4), s = Math.floor((ms % 6e4) / 1e3);
  const d = Math.floor(h / 24);
  const text = d >= 2 ? `${d}d ${h % 24}h` : `${h}h ${String(m).padStart(2, "0")}m ${String(s).padStart(2, "0")}s`;
  box.append(`${gw.name} deadline in `, el("strong", { text }), el("br"), new Date(gw.deadline_time).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit", day: "numeric", month: "short" }));
}
setInterval(tick, 1000);

// ---------------------------------------------------------------- plan, pitch, chips
function playerCard(id, gw, { isIn, captain, vice } = {}) {
  const p = state.byId.get(id);
  const name = p ? p.name : `#${id}`;
  const opp = p && p.opp[gw] && p.opp[gw].length ? p.opp[gw].join(", ") : "blank";
  const xp = p ? p.xp[gw] : null;
  const flagged = p && p.status !== "a";
  return el("div", { class: `player${isIn ? " is-in" : ""}${flagged ? " flag" : ""}`, title: p && p.news ? p.news : null },
    isIn ? el("span", { class: "tag in", text: "IN" }) : null,
    captain ? el("span", { class: "badge c", text: "C" }) : vice ? el("span", { class: "badge v", text: "V" }) : null,
    el("div", { class: "name", text: name }),
    el("div", { class: "meta", text: `${p ? p.team : ""} · ${opp}` }),
    el("div", { class: "xp", text: fmt(captain ? xp * 2 : xp, 1) }));
}

function renderPlan() {
  const opt = state.data.optimization;
  const pitch = $("pitch"), bench = $("bench"), tl = $("timeline"), kpis = $("kpis"), chips = $("chips");
  [pitch, bench, tl, kpis, chips].forEach((n) => n.replaceChildren());
  if (!opt) {
    pitch.append(el("div", { class: "empty", text: "No plan yet — press “Run dry-run”." }));
    $("plan-meta").textContent = "";
    return;
  }
  const g = opt.plan[0];
  const gw = g.gameweek;
  const incoming = new Set(g.transfers_in);
  $("pitch-title").textContent = `GW${gw} lineup${g.chip ? ` · ${CHIP_LABEL[g.chip] || g.chip}` : ""}`;

  const byPos = { GKP: [], DEF: [], MID: [], FWD: [] };
  for (const id of g.starting_xi) {
    const p = state.byId.get(id);
    (byPos[p ? p.position : "MID"] || byPos.MID).push(id);
  }
  for (const pos of ["GKP", "DEF", "MID", "FWD"]) {
    const ids = byPos[pos].sort((a, b) => (state.byId.get(b)?.xp[gw] ?? 0) - (state.byId.get(a)?.xp[gw] ?? 0));
    pitch.append(el("div", { class: "row" }, ids.map((id) => playerCard(id, gw, { isIn: incoming.has(id), captain: id === g.captain, vice: id === g.vice_captain }))));
  }
  g.bench.forEach((id, k) => bench.append(el("div", {}, el("div", { class: "slot", text: k === 0 ? "GK" : `${k}.` }), playerCard(id, gw, { isIn: incoming.has(id) }))));

  const gain = opt.objective - opt.baseline_objective;
  const ts = opt.team_state;
  kpis.append(
    el("div", { class: "kpi" }, el("div", { class: "label", text: `xP GW${gw}` }), el("div", { class: "value", text: fmt(g.xp, 1) })),
    el("div", { class: "kpi" }, el("div", { class: "label", text: "Gain vs no moves" }), el("div", { class: "value good", text: `+${fmt(gain, 1)}` })),
    el("div", { class: "kpi" }, el("div", { class: "label", text: "Free transfers" }), el("div", { class: "value", text: String(ts.free_transfers) })),
  );
  $("plan-meta").textContent = `bank ${money(ts.bank)} · ${opt.solver.status.toLowerCase()} · ${new Date(opt.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;

  for (const step of opt.plan) {
    const byPos = (a, b) => (POS_ORDER[state.byId.get(a)?.position] - POS_ORDER[state.byId.get(b)?.position]) || a - b;
    const outs = [...step.transfers_out].sort(byPos);
    const ins = [...step.transfers_in].sort(byPos);
    const moves = outs.length
      ? outs.map((o, i) => el("div", { class: "move" },
          el("span", { class: "tag out", text: "OUT" }), state.byId.get(o)?.name ?? o, el("span", { class: "arrow", text: "→" }),
          el("span", { class: "tag in", text: "IN" }), state.byId.get(ins[i])?.name ?? ins[i]))
      : [el("div", { class: "move muted", text: step.chip ? "" : "Roll transfer" })];
    const chip = step.chip ? el("div", { class: "move" }, el("span", { class: "pill use", text: CHIP_LABEL[step.chip] || step.chip })) : null;
    tl.append(el("li", {},
      el("div", { class: "gw" }, `GW${step.gameweek}`, el("small", { text: `${step.free_transfers} FT${step.hits ? ` · −${step.hits * 4}` : ""}` })),
      el("div", { class: "moves" }, chip, moves),
      el("div", { class: "right" }, "C ", el("strong", { text: state.byId.get(step.captain)?.name ?? "" }), el("br"), `xP `, el("strong", { text: fmt(step.xp, 1) }))));
  }

  const evals = opt.chip_evaluation || [];
  $("chip-meta").textContent = opt.chip_recommended ? `play ${CHIP_LABEL[opt.chip_recommended]}` : evals.length ? "holding all" : "";
  if (!evals.length) chips.append(el("div", { class: "empty", text: "No chips available this gameweek." }));
  const maxGain = Math.max(1, ...evals.map((e) => e.gain));
  for (const e of evals) {
    chips.append(el("div", { class: "chip-row" },
      el("div", {}, el("div", { class: "chip-name", text: CHIP_LABEL[e.chip] || e.chip }), el("div", { class: "muted num", text: `${e.gain >= 0 ? "+" : ""}${fmt(e.gain, 1)} xP` })),
      el("div", {}, el("div", { class: "chip-reason", text: e.reason }), el("div", { class: "gainbar" }, el("span", { style: `width:${Math.max(0, (e.gain / maxGain) * 100)}%` }))),
      el("span", { class: `pill ${e.triggered ? "use" : "hold"}`, text: e.triggered ? "Use" : "Hold" })));
  }
}

// ---------------------------------------------------------------- autopilot (Phase 5/6)
function renderAutopilot() {
  const box = $("autopilot");
  box.replaceChildren();
  const c = state.data.control;
  if (!c) { box.append(el("div", { class: "empty", text: "Set FPL_TEAM_ID to enable autopilot." })); return; }
  const on = c.config.auto_execute_enabled;
  const auth = c.auth || {};
  const authPill = !auth.has_tokens ? el("span", { class: "pill off", text: "no tokens" })
    : auth.auth_last_check_ok === false ? el("span", { class: "pill bad", text: "failing", title: auth.auth_last_error || "" })
    : auth.auth_last_check_ok ? el("span", { class: "pill use", text: "ok" }) : el("span", { class: "pill warn", text: "unchecked" });
  const steps = el("div", { class: "steps" }, [["dry_run", "T−24h"], ["final_lock", "T−30m"], ["commit", "T−15m"]].map(([k, label]) => {
    const a = c.deadline_actions[k];
    const cls = !a ? "off" : a.status === "SUCCESS" ? "use" : a.status === "SKIPPED" ? "warn" : a.status === "RUNNING" ? "warn" : "bad";
    return el("span", { class: `pill ${cls}`, title: a ? a.status : "pending", text: label });
  }));
  const last = c.last_execution;
  $("auto-meta").textContent = `max ${c.config.max_hits_per_gw} hit/GW · chips ${c.config.chip_aggression_profile.toLowerCase()}`;
  box.append(...[
    el("div", { class: "ap-row" }, el("span", { class: "k", text: "Auto-execute" }), el("span", { class: `pill ${on ? "use" : "off"}`, text: on ? "ON" : "OFF" })),
    el("div", { class: "ap-row" }, el("span", { class: "k", text: "FPL login" }), authPill),
    auth.session_expires_at ? el("div", { class: "ap-row" }, el("span", { class: "k", text: "Sign in again by" }),
      el("span", { class: "v", text: new Date(auth.session_expires_at).toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" }) })) : null,
    el("div", { class: "ap-row" }, el("span", { class: "k", text: "Deadline actions" }), steps),
    el("div", { class: "ap-row" }, el("span", { class: "k", text: "Last execution" }),
      el("span", { class: "v", text: last ? `GW${last.gameweek} ${last.execution_type.replace("_", " ").toLowerCase()} · ${last.status.toLowerCase()}` : "none" })),
    el("div", { class: "ap-actions" },
      on ? el("button", { class: "btn danger", type: "button", onclick: () => setAuto(false), text: "Kill switch" })
         : el("button", { class: "btn", type: "button", onclick: () => setAuto(true), disabled: !(auth.has_tokens && auth.auth_last_check_ok), text: "Enable auto-execute",
               title: auth.has_tokens && auth.auth_last_check_ok ? null : "FPL login must pass a live check first" })),
  ].filter(Boolean));  // optional rows are null; Node.append would print them as the text "null"
}

async function setAuto(enabled) {
  let token = null;
  try { token = sessionStorage.getItem("fplApiToken"); } catch (_) { /* storage unavailable */ }
  if (!token) token = window.prompt("API token (FPL_API_TOKEN):");
  if (!token) return;
  const res = await fetch(`/api/v1/config/${state.data.team.entry_id}`, {
    method: "PATCH", headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
    body: JSON.stringify({ auto_execute_enabled: enabled }),
  });
  if (res.status === 401) { try { sessionStorage.removeItem("fplApiToken"); } catch (_) { /* ignore */ } banner("error", "Wrong API token."); return; }
  if (!res.ok) { const b = await res.json().catch(() => ({})); banner("error", b.detail || `HTTP ${res.status}`); return; }
  try { sessionStorage.setItem("fplApiToken", token); } catch (_) { /* ignore */ }
  banner("ok", enabled ? "Auto-execute enabled. Guardrails still apply." : "Kill switch engaged: auto-execute is off.");
  await load();
}

// ---------------------------------------------------------------- projections table
function columns() {
  const gws = state.data.gameweeks;
  return [
    { key: "name", label: "Player", cls: "l", get: (p) => p.name },
    { key: "team", label: "Team", cls: "l", get: (p) => p.team },
    { key: "position", label: "Pos", cls: "l", get: (p) => POS_ORDER[p.position] },
    { key: "price", label: "£", get: (p) => p.price },
    { key: "owned", label: "Own %", get: (p) => p.owned ?? 0 },
    { key: "pplay", label: "P(play)", get: (p) => p.p_play[gws[0]] ?? 0 },
    ...gws.map((gw) => ({ key: `gw${gw}`, label: `GW${gw}`, gw, get: (p) => p.xp[gw] ?? 0 })),
    { key: "total", label: "Total", get: (p) => gws.reduce((s, gw) => s + (p.xp[gw] ?? 0), 0) },
  ];
}

function renderTableHead() {
  const head = $("thead");
  head.replaceChildren();
  for (const c of columns()) {
    const sorted = state.sort.key === c.key ? (state.sort.dir < 0 ? "descending" : "ascending") : null;
    head.append(el("th", { class: c.cls, "aria-sort": sorted, scope: "col", tabindex: "0",
      onclick: () => setSort(c.key), onkeydown: (ev) => { if (ev.key === "Enter") setSort(c.key); } }, c.label));
  }
}

function setSort(key) {
  state.sort = state.sort.key === key ? { key, dir: -state.sort.dir } : { key, dir: ["name", "team", "position"].includes(key) ? 1 : -1 };
  renderTableHead();
  renderTable();
}

function mySquad() {
  const opt = state.data.optimization;
  return new Set(opt ? opt.team_state.squad : []);
}

function renderTable() {
  const cols = columns();
  const col = cols.find((c) => c.key === state.sort.key) || cols[cols.length - 1];
  const mine = mySquad();
  const q = state.q.trim().toLowerCase();
  const rows = state.data.players.filter((p) =>
    (!state.pos || p.position === state.pos) &&
    (!q || p.name.toLowerCase().includes(q) || p.team.toLowerCase().includes(q)) &&
    (state.maxPrice == null || p.price <= state.maxPrice * 10) &&
    (!state.mine || mine.has(p.player_id)));
  rows.sort((a, b) => {
    const va = col.get(a), vb = col.get(b);
    return (typeof va === "string" ? va.localeCompare(vb) : va - vb) * state.sort.dir;
  });
  const maxXp = Math.max(1, ...state.data.players.flatMap((p) => Object.values(p.xp)));
  const body = $("tbody");
  body.replaceChildren();
  for (const p of rows.slice(0, state.limit)) {
    body.append(el("tr", { class: mine.has(p.player_id) ? "mine" : null }, cols.map((c) => {
      if (c.key === "name") return el("td", { class: "l", title: p.news || null }, p.name, p.status !== "a" ? el("span", { class: `status-dot ${p.status}`, "aria-label": `status ${p.status}` }) : null);
      if (c.key === "team") return el("td", { class: "l", text: p.team });
      if (c.key === "position") return el("td", { class: "l", text: p.position });
      if (c.key === "price") return el("td", { class: "num", text: (p.price / 10).toFixed(1) });
      if (c.key === "owned") return el("td", { class: "num", text: fmt(p.owned, 1) });
      if (c.key === "pplay") return el("td", { class: "num", text: `${Math.round((p.p_play[state.data.gameweeks[0]] ?? 0) * 100)}%` });
      if (c.gw) {
        const v = p.xp[c.gw] ?? 0;
        const alpha = Math.min(0.3, Math.max(0, (v - 1) / (maxXp - 1)) * 0.3);
        return el("td", { class: "heat", style: `background: rgba(var(--heat), ${alpha.toFixed(3)})` },
          fmt(v, 2), el("span", { class: "opp", text: (p.opp[c.gw] || []).join(", ") || "blank" }));
      }
      return el("td", { class: "num", text: fmt(c.get(p), 1) });
    })));
  }
  if (!rows.length) body.append(el("tr", {}, el("td", { colspan: String(cols.length), class: "empty", text: "No players match." })));
  $("more").hidden = rows.length <= state.limit;
}

// ---------------------------------------------------------------- pipeline
function renderJobs() {
  const box = $("jobs");
  box.replaceChildren();
  const ago = (t) => {
    if (!t) return "";
    const mins = Math.round((Date.now() - new Date(t)) / 6e4);
    return mins < 60 ? `${mins} min ago` : mins < 2880 ? `${Math.round(mins / 60)} h ago` : `${Math.round(mins / 1440)} d ago`;
  };
  for (const j of state.data.jobs) {
    box.append(el("div", { class: "job" },
      el("div", { class: "jn" }, j.job, el("span", { class: `st ${j.status}`, text: j.status.toLowerCase() })),
      el("div", { class: "when", text: ago(j.finished_at || j.started_at) }),
      j.error_message ? el("div", { class: "err", text: j.error_message }) : null));
  }
}

// ---------------------------------------------------------------- controls
document.querySelectorAll("#pos-filter button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#pos-filter button").forEach((x) => x.classList.toggle("on", x === b));
  state.pos = b.dataset.pos; state.limit = PAGE; renderTable();
}));
$("search").addEventListener("input", (e) => { state.q = e.target.value; state.limit = PAGE; renderTable(); });
$("max-price").addEventListener("input", (e) => { state.maxPrice = e.target.value === "" ? null : Number(e.target.value); renderTable(); });
$("mine").addEventListener("change", (e) => { state.mine = e.target.checked; renderTable(); });
$("more").addEventListener("click", () => { state.limit += PAGE; renderTable(); });

$("run-btn").addEventListener("click", async () => {
  const btn = $("run-btn");
  btn.disabled = true;
  btn.textContent = "Solving…";
  banner("info", "Running the optimiser on the latest projections (about 30 seconds)…");
  try {
    const res = await fetch("/api/v1/optimize/dry-run", { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    await load();
    banner("ok", "Plan updated. Nothing was submitted to FPL.");
  } catch (err) {
    banner("error", `Dry-run failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run dry-run";
  }
});

load().catch((err) => banner("error", `Could not load dashboard: ${err.message}`));
