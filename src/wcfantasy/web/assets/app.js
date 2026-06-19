// WC Fantasy Advisor — single-page app (vanilla JS, no build step).
// Talks to the local FastAPI service; slow commands run as background jobs
// that we poll for progress. Nothing here writes to the FIFA account.

import { t, setLang, getLang, LANG_NAMES, roleLabel } from "./i18n.js";

// ----------------------------------------------------------------- helpers
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

const fmtPct = (x) => `${Math.round((x ?? 0) * 100)}%`;
const fmtMoney = (x) => `$${Number(x ?? 0).toFixed(1)}m`;

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j && j.detail) detail = j.detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}
const apiGet = (p) => api(p);
const apiPost = (p, body) =>
  api(p, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

// Start a job (startFn -> {job_id}) and poll until done; onProgress(snapshot).
async function runJob(startFn, onProgress) {
  const { job_id } = await startFn();
  while (true) {
    const snap = await apiGet(`/api/jobs/${job_id}`);
    onProgress(snap);
    if (snap.status === "done") return snap.result;
    if (snap.status === "error") throw new Error(snap.error || "job failed");
    await sleep(600);
  }
}

function jobPanel() {
  const log = el("div", { class: "log" });
  const title = el("span", { text: t("common.loading") });
  const node = el("div", { class: "job-panel" }, [
    el("div", { class: "job-head" }, [el("div", { class: "spinner" }), title]),
    log,
  ]);
  return {
    node,
    update(snap) {
      log.textContent = (snap.logs || []).join("\n");
      log.scrollTop = log.scrollHeight;
    },
    fail(msg) {
      node.querySelector(".spinner")?.remove();
      title.textContent = t("common.error");
      node.append(el("div", { class: "callout danger", text: msg }));
    },
  };
}

// Run a job into `host`, showing the progress panel, then render the result.
async function runJobView(host, startFn, renderResult) {
  const panel = jobPanel();
  host.replaceChildren(panel.node);
  try {
    const result = await runJob(startFn, (s) => panel.update(s));
    host.replaceChildren();
    renderResult(result, host);
  } catch (e) {
    panel.fail(e.message);
  }
}

function toast(msg, kind = "") {
  const node = el("div", { class: `toast ${kind}`, text: msg });
  document.getElementById("toast-root").append(node);
  setTimeout(() => node.remove(), 4200);
}

function confirmModal({ title, body, confirmLabel, danger }) {
  return new Promise((resolve) => {
    const close = (val) => { root.replaceChildren(); resolve(val); };
    const root = document.getElementById("modal-root");
    root.replaceChildren(
      el("div", { class: "modal-backdrop", onclick: (e) => { if (e.target === e.currentTarget) close(false); } }, [
        el("div", { class: "modal" }, [
          el("h3", { text: title }),
          el("p", { text: body }),
          el("div", { class: "actions" }, [
            el("button", { class: "btn ghost", text: t("common.cancel"), onclick: () => close(false) }),
            el("button", { class: `btn ${danger ? "danger" : ""}`, text: confirmLabel || t("common.confirm"), onclick: () => close(true) }),
          ]),
        ]),
      ])
    );
  });
}

// ----------------------------------------------------------------- shared bits
function posChip(pos) { return el("span", { class: `chip pos ${pos}`, text: pos }); }

function playerLine(p) {
  const tags = [];
  if (p.is_captain || p.role === "CAPTAIN") tags.push(el("span", { class: "tag-c", text: "C" }));
  if (p.is_vice || p.role === "VICE") tags.push(el("span", { class: "tag-v", text: "V" }));
  return el("div", { class: "actions", style: "gap:8px;margin:6px 0" }, [
    posChip(p.position),
    el("strong", { text: p.name }),
    ...tags,
    el("span", { class: "team-chip", text: `· ${p.team}` }),
    el("span", { class: "header-spacer" }),
    el("span", { class: "note", text: roleLabel(p.role) }),
  ]);
}

function squadTable(rows) {
  const head = el("tr", {}, [
    el("th", { text: t("common.pos") }),
    el("th", { text: t("common.player") }),
    el("th", { text: t("common.team") }),
    el("th", { class: "num", text: t("common.price") }),
    el("th", { class: "num", text: t("common.ep") }),
    el("th", { text: t("common.play") }),
    el("th", { text: t("common.role") }),
  ]);
  const body = rows.map((r) =>
    el("tr", { class: r.role === "bench" || r.role === "bench GK" || r.role.startsWith("sub") ? "" : "is-start" }, [
      el("td", {}, [posChip(r.position)]),
      el("td", { text: r.name }),
      el("td", { class: "team-chip", text: r.team }),
      el("td", { class: "num", text: fmtMoney(r.price) }),
      el("td", { class: "num", text: r.ep_match.toFixed(2) }),
      el("td", {}, [
        el("span", { class: "playbar" }, [el("span", { style: `width:${Math.round(r.p_play * 100)}%` })]),
        `${Math.round(r.p_play * 100)}%`,
      ]),
      el("td", {}, [el("span", { class: "chip", text: roleLabel(r.role) })]),
    ])
  );
  return el("table", { class: "data" }, [el("thead", {}, [head]), el("tbody", {}, body)]);
}

function linesCard(title, lines, opts = {}) {
  if (!lines || !lines.length) return null;
  return el("div", { class: "card" }, [
    el("h3", { text: title }),
    el("ul", { class: "lines" }, lines.map((l) => el("li", { text: l }))),
  ]);
}

// ----------------------------------------------------------------- views
const view = () => document.getElementById("view");

async function renderDashboard() {
  const host = view();
  host.replaceChildren(pageHead("dashboard.title", null));
  let data;
  try { data = await apiGet("/api/team"); } catch (e) { host.append(errorCard(e.message)); return; }

  if (!data.has_team) { host.append(emptyState(t("dashboard.empty"))); return; }
  if (!data.feeds_available) { host.append(emptyState(t("dashboard.feedsMissing"))); return; }

  host.append(
    el("div", { class: "card" }, [
      el("div", { class: "stat-row" }, [
        stat("dashboard.round", data.round_no),
        stat("dashboard.bank", fmtMoney(data.bank)),
        stat("dashboard.boostersUsed", data.boosters_used.length ? data.boosters_used.join(", ") : t("common.none")),
        data.transfers_carried ? stat("dashboard.carried", data.transfers_carried) : null,
      ]),
    ])
  );

  const starters = data.players.filter((p) => p.is_starter);
  const bench = data.players.filter((p) => !p.is_starter);
  host.append(
    el("div", { class: "grid-2" }, [
      el("div", { class: "card" }, [el("h3", { text: t("dashboard.startingXI") }), ...starters.map(playerLine)]),
      el("div", { class: "card" }, [el("h3", { text: t("dashboard.bench") }), ...bench.map(playerLine)]),
    ])
  );
}

function stageLabel(round) {
  return t(`stage.${round}`);
}

function roundHero(plan) {
  return el("div", { class: "round-hero" }, [
    el("div", { class: "round-hero-badge", text: plan.round }),
    el("div", {}, [
      el("div", { class: "round-hero-kicker", text: t("plan.planningFor") }),
      el("div", { class: "round-hero-title", text: stageLabel(plan.round) }),
      el("div", { class: "round-hero-sub", text: `${fmtMoney(plan.budget)} · ${t("plan.maxPerNation", { n: plan.country_cap })}` }),
    ]),
  ]);
}

function moveRow(badgeClass, badgeText, label, sub) {
  return el("div", { class: "move-row" }, [
    el("span", { class: `badge-ic ${badgeClass}`, text: badgeText }),
    el("div", {}, [
      label ? el("span", { class: "label", text: label }) : null,
      sub ? el("div", { class: "sub", text: sub }) : null,
    ]),
  ]);
}

// Structured, iconified "moves to execute" card: red ↓ for OUT, green ↑ for
// IN, a gold armband for the captain, silver for the vice, a roll icon for the
// timing note, and a numbered bench strip.
function movesCardNode(plan) {
  const tr = plan.transfers;
  const rows = el("div", { class: "moves" });

  if (tr) {
    if (tr.out.length || tr.in.length) {
      tr.out.forEach((name) => rows.append(moveRow("out", "↓", `${t("plan.transfersOut")}: ${name}`)));
      tr.in.forEach((name) => rows.append(moveRow("in", "↑", `${t("plan.transfersIn")}: ${name}`)));
    } else if (plan.moves.length) {
      rows.append(el("div", { class: "note", text: plan.moves[0] }));
    }
    for (const note of [tr.forced_note, tr.voluntary_note, tr.carry_note]) {
      if (note) rows.append(el("div", { class: "callout move-callout", text: note }));
    }
  } else {
    for (const m of plan.moves) rows.append(el("div", { class: "note", text: m }));
  }

  rows.append(moveRow("cap", "C", `${t("plan.captain")}: ${plan.captain.name}`));
  rows.append(moveRow("vice", "V", `${t("plan.vice")}: ${plan.vice.name}`));
  if (plan.captain_timing_note) {
    rows.append(moveRow("timing", "↻", null, plan.captain_timing_note));
  }

  const strip = el("div", { class: "bench-strip" });
  plan.bench_order.forEach((b, i) =>
    strip.append(el("span", { class: "bench-pill" }, [el("span", { class: "badge-ic num sm", text: i + 1 }), b.name]))
  );
  strip.append(el("span", { class: "bench-pill" }, [el("span", { class: "badge-ic gk sm", text: "GK" }), plan.bench_gk.name]));

  return el("div", { class: "card" }, [
    el("h3", { text: t("plan.moves") }),
    rows,
    el("div", { style: "margin-top:12px" }, [
      el("div", { class: "note", style: "margin-bottom:2px", text: t("plan.benchOrder") }),
      strip,
    ]),
  ]);
}

// A card whose header carries an icon badge and whose lines are listed plainly
// (used by the live captain / subs / outlook sections).
function iconLinesCard(badgeClass, badgeText, title, lines) {
  if (!lines || !lines.length) return null;
  return el("div", { class: "card" }, [
    el("h3", {}, [el("span", { class: `badge-ic ${badgeClass} sm`, text: badgeText }), title]),
    el("ul", { class: "lines" }, lines.map((l) => el("li", { text: l }))),
  ]);
}

function planResultNode(plan, { applyable }) {
  const s = plan.status || {};
  const frag = document.createDocumentFragment();

  // big, clear "which round are we planning for" banner, first thing shown
  frag.append(roundHero(plan));

  // status / model line
  const statusLines = [s.freshness, s.model_line, s.country_rule].filter(Boolean);
  if (s.header) statusLines.unshift(s.header);
  if (s.transfer_window) statusLines.push(s.transfer_window);
  frag.append(el("div", { class: "card" }, [el("ul", { class: "lines note" }, statusLines.map((l) => el("li", { text: l })))]));

  // squad
  frag.append(el("div", { class: "card" }, [
    el("h3", {}, [t("plan.squad"), " ", el("span", { class: "muted", text: `· ${t("plan.cost")} ${fmtMoney(plan.cost)} · ${t("plan.bankLeft")} ${fmtMoney(plan.bank)}` })]),
    squadTable(plan.squad_rows),
  ]));

  // moves + captain (iconified, structured)
  frag.append(movesCardNode(plan));

  // availability watch
  if (plan.availability_watch && plan.availability_watch.length) {
    frag.append(el("div", { class: "card" }, [
      el("h3", { text: t("plan.availabilityWatch") }),
      el("ul", { class: "lines" }, plan.availability_watch.map((r) =>
        el("li", { text: `${r.name} (${r.team}): ${fmtPct(r.p_play)} ${t("plan.likelyToPlay")}` }))),
    ]));
  }
  // unverified starters
  if (plan.unverified_starters && plan.unverified_starters.length) {
    frag.append(linesCard(t("plan.unverified"), plan.unverified_starters));
  }
  // comparison (from-scratch)
  if (plan.comparison && plan.comparison.length) {
    frag.append(linesCard(t("ideal.comparison"), plan.comparison));
  }
  // booster + schedule
  if (plan.booster_advice && plan.booster_advice.length) frag.append(linesCard(t("plan.boosterAdvice"), plan.booster_advice));
  if (plan.schedule && plan.schedule.length) frag.append(linesCard(t("plan.schedule"), plan.schedule));

  // apply
  if (applyable && plan.applyable) {
    const applyBtn = el("button", { class: "btn", text: t("plan.apply") });
    applyBtn.addEventListener("click", () => doApply(false));
    frag.append(el("div", { class: "card" }, [
      el("p", { class: "note", text: t("plan.applyHint") }),
      el("div", { class: "actions" }, [applyBtn]),
    ]));
  }
  return frag;
}

async function doApply(force) {
  let res;
  try { res = await apiPost("/api/apply", { force }); } catch (e) { toast(e.message, "err"); return; }
  if (res.status === "blocked") {
    const ok = await confirmModal({
      title: t("apply.blockedTitle"), body: res.message,
      confirmLabel: t("apply.blockedForce"), danger: true,
    });
    if (ok) return doApply(true);
    return;
  }
  toast(res.message, res.status === "applied" ? "ok" : "err");
  if (res.status === "applied") refreshStatusStrip();
}

async function renderPlan() {
  const host = view();
  host.replaceChildren(pageHead("plan.title", "plan.intro"));
  const results = el("div", {});
  const btn = el("button", { class: "btn", text: t("plan.run") });
  btn.addEventListener("click", () => {
    btn.disabled = true;
    runJobView(results, () => apiPost("/api/advise"), (plan, h) => {
      btn.disabled = false;
      h.append(planResultNode(plan, { applyable: true }));
    }).finally(() => (btn.disabled = false));
  });
  host.append(el("div", { class: "card" }, [el("div", { class: "actions" }, [btn])]), results);
}

async function renderIdeal() {
  const host = view();
  host.replaceChildren(pageHead("ideal.title", "ideal.intro"));
  const results = el("div", {});
  const btn = el("button", { class: "btn secondary", text: t("ideal.run") });
  btn.addEventListener("click", () => {
    btn.disabled = true;
    runJobView(results, () => apiPost("/api/advise/from-scratch"), (plan, h) => {
      btn.disabled = false;
      h.append(planResultNode(plan, { applyable: false }));
    }).finally(() => (btn.disabled = false));
  });
  host.append(el("div", { class: "card" }, [el("div", { class: "actions" }, [btn])]), results);
}

function liveResultNode(state) {
  const frag = document.createDocumentFragment();
  if (state.status === "no_team") { frag.append(emptyState(t("live.noTeam"))); return frag; }
  if (state.status === "no_live_round") { frag.append(emptyState(t("live.noRound"))); return frag; }
  if (state.lineup_warning) frag.append(el("div", { class: "card" }, [el("div", { class: "callout", text: state.lineup_warning })]));
  if (state.status === "record_error") {
    frag.append(el("div", { class: "card" }, [el("div", { class: "callout danger", text: state.recorded_error })]));
    return frag;
  }
  if (state.recorded && state.recorded.length) frag.append(linesCard(t("live.recorded"), state.recorded, {}));
  if (state.change_cost_note) frag.append(el("div", { class: "card" }, [el("div", { class: "callout", text: state.change_cost_note })]));
  frag.append(iconLinesCard("cap", "C", t("live.captain"), state.captain_lines));
  frag.append(iconLinesCard("swap", "⇄", t("live.subs"), state.sub_lines));
  frag.append(iconLinesCard("play", "⏱", t("live.outlook"), state.outlook_lines));
  return frag;
}

function recordForm(onDone) {
  const f = {
    captain: el("input", { type: "text" }),
    sub_out: el("input", { type: "text" }),
    sub_in: el("input", { type: "text" }),
    vice: el("input", { type: "text" }),
    mark: el("input", { type: "checkbox" }),
  };
  const submit = el("button", { class: "btn", text: t("live.record") });
  submit.addEventListener("click", async () => {
    const body = {
      captain: f.captain.value.trim() || null,
      sub_out: f.sub_out.value.trim() || null,
      sub_in: f.sub_in.value.trim() || null,
      vice: f.vice.value.trim() || null,
      mark_changed: f.mark.checked,
    };
    submit.disabled = true;
    let res;
    try { res = await apiPost("/api/live/record", body); }
    catch (e) { toast(e.message, "err"); submit.disabled = false; return; }
    submit.disabled = false;
    if (res.status === "ok") {
      toast(res.messages && res.messages.length ? res.messages.join(" ") : t("live.recordOk"), "ok");
      onDone();
    } else {
      toast(res.message, "err");
    }
  });
  return el("div", { class: "card" }, [
    el("h3", { text: t("live.recordTitle") }),
    el("p", { class: "note", text: t("live.recordIntro") }),
    el("div", { class: "grid-2" }, [
      field("live.fieldCaptain", f.captain),
      field("live.fieldVice", f.vice),
      field("live.fieldSubOut", f.sub_out),
      field("live.fieldSubIn", f.sub_in),
    ]),
    el("label", { class: "check" }, [f.mark, el("span", { text: t("live.fieldMarkChanged") })]),
    el("div", { class: "actions", style: "margin-top:12px" }, [submit]),
  ]);
}

async function renderLive() {
  const host = view();
  host.replaceChildren(pageHead("live.title", "live.intro"));
  const results = el("div", {});
  const btn = el("button", { class: "btn", text: t("live.run") });
  const go = () => {
    btn.disabled = true;
    runJobView(results, () => apiGet("/api/live"), (state, h) => {
      btn.disabled = false;
      h.append(liveResultNode(state));
      h.append(recordForm(go));
    }).finally(() => (btn.disabled = false));
  };
  btn.addEventListener("click", go);
  host.append(el("div", { class: "card" }, [el("div", { class: "actions" }, [btn])]), results);
}

async function renderOdds() {
  const host = view();
  host.replaceChildren(pageHead("odds.title", "odds.intro"));
  const results = el("div", {});
  const btn = el("button", { class: "btn", text: t("odds.run") });
  btn.addEventListener("click", () => {
    btn.disabled = true;
    runJobView(results, () => apiGet("/api/odds"), (res, h) => {
      btn.disabled = false;
      if (!res.available) { h.append(el("div", { class: "card" }, [el("div", { class: "callout danger", text: res.message })])); return; }
      const head = el("tr", {}, [
        el("th", { text: t("common.team") }),
        el("th", { class: "num", text: t("odds.pwin") }),
        el("th", { class: "num", text: t("odds.fair") }),
        el("th", { class: "num", text: t("odds.market") }),
      ]);
      const body = res.rows.map((r) => el("tr", {}, [
        el("td", { text: r.team }),
        el("td", { class: "num", text: `${(r.p_win * 100).toFixed(1)}%` }),
        el("td", { class: "num", text: r.fair_odds.toFixed(1) }),
        el("td", { class: "num", text: r.market ? `${(r.market * 100).toFixed(1)}%` : "—" }),
      ]));
      h.append(el("div", { class: "card" }, [
        el("h3", {}, [t("odds.title"), " ", el("span", { class: "muted", text: `· ${t("odds.model")}: ${res.model}` })]),
        el("table", { class: "data" }, [el("thead", {}, [head]), el("tbody", {}, body)]),
        res.market_on ? null : el("p", { class: "note", text: t("odds.marketOff") }),
      ]));
    }).finally(() => (btn.disabled = false));
  });
  host.append(el("div", { class: "card" }, [el("div", { class: "actions" }, [btn])]), results);
}

async function renderBoosters() {
  const host = view();
  host.replaceChildren(pageHead("boosters.title", "boosters.intro"));
  let data;
  try { data = await apiGet("/api/team"); } catch (e) { host.append(errorCard(e.message)); return; }
  const list = el("div", { class: "grid-2" });
  for (const b of data.boosters) {
    const actions = el("div", { class: "actions", style: "margin-top:10px" });
    if (b.used) {
      actions.append(el("span", { class: "chip", text: `✓ ${t("boosters.used")}` }));
    } else {
      const btn = el("button", { class: "btn sm secondary", text: t("boosters.markUsed") });
      btn.addEventListener("click", async () => {
        const ok = await confirmModal({
          title: t("boosters.confirmTitle"),
          body: b.irreversible ? t("boosters.confirmWildcard") : t("boosters.confirmGeneric"),
          confirmLabel: t("boosters.markUsed"),
          danger: b.irreversible,
        });
        if (!ok) return;
        try {
          const res = await apiPost("/api/booster", { name: b.key });
          toast(res.message, res.status === "recorded" ? "ok" : "err");
          renderBoosters();
        } catch (e) { toast(e.message, "err"); }
      });
      actions.append(btn);
    }
    list.append(el("div", { class: "card" }, [
      el("h3", {}, [b.key.replace(/_/g, " "), b.irreversible ? el("span", { class: "tag-c", text: "!" }) : null]),
      el("p", { class: "note", text: b.description }),
      actions,
    ]));
  }
  host.append(list);
}

async function renderCalibrate() {
  const host = view();
  host.replaceChildren(pageHead("calibrate.title", "calibrate.intro"));
  const results = el("div", {});
  const preview = el("button", { class: "btn secondary", text: t("calibrate.preview") });
  const write = el("button", { class: "btn", text: t("calibrate.write") });
  const run = (doWrite) => {
    preview.disabled = write.disabled = true;
    runJobView(results, () => apiPost("/api/calibrate", { write: doWrite }), (res, h) => {
      preview.disabled = write.disabled = false;
      h.append(calibrateResultNode(res));
    }).finally(() => { preview.disabled = write.disabled = false; });
  };
  preview.addEventListener("click", () => run(false));
  write.addEventListener("click", async () => {
    const ok = await confirmModal({ title: t("calibrate.write"), body: t("calibrate.confirmWrite"), confirmLabel: t("calibrate.write") });
    if (ok) run(true);
  });
  host.append(el("div", { class: "card" }, [el("div", { class: "actions" }, [preview, write])]), results);
}

function calibrateResultNode(res) {
  const lines = [];
  lines.push(`${t("calibrate.nplayed")}: ${res.n_played}`);
  if (res.default_price_metrics) lines.push(`${t("calibrate.beforeMetrics")}: ${JSON.stringify(res.default_price_metrics)}`);
  if (res.refit_price_metrics) lines.push(`${t("calibrate.refitMetrics")}: ${JSON.stringify(res.refit_price_metrics)}`);
  if (res.club_blend) lines.push(`${t("calibrate.clubBlend")}: w=${res.club_blend.w} (${res.club_blend.n_club}) ${JSON.stringify(res.club_blend.metrics)}`);
  lines.push(`${t("calibrate.proposed")}: ${Object.keys(res.proposed_weights).length ? JSON.stringify(res.proposed_weights) : "(none)"}`);

  const status = res.written
    ? el("div", { class: "callout ok", text: t("calibrate.written") })
    : el("div", { class: "callout", text: `${t("calibrate.notWritten")}: ${res.refusal_reason || ""}` });

  return el("div", { class: "card" }, [
    el("ul", { class: "lines" }, lines.map((l) => el("li", { text: l }))),
    status,
  ]);
}

async function renderSettings() {
  const host = view();
  host.replaceChildren(pageHead("settings.title", null));
  let status;
  try { status = await apiGet("/api/status"); } catch (e) { host.append(errorCard(e.message)); return; }

  // language
  const langBtns = el("div", { class: "lang-select" });
  for (const code of status.supported_languages) {
    const b = el("button", { class: getLang() === code ? "active" : "", text: LANG_NAMES[code] || code });
    b.addEventListener("click", () => setLanguage(code));
    langBtns.append(b);
  }

  // sources
  const yesNo = (on, onTxt, offTxt) => el("span", { class: `chip`, text: on ? (onTxt || t("settings.on")) : (offTxt || t("settings.off")) });
  const fetchBtn = el("button", { class: "btn secondary", text: t("settings.fetch") });
  const fetchResult = el("div", {});
  fetchBtn.addEventListener("click", () => {
    fetchBtn.disabled = true;
    runJobView(fetchResult, () => apiPost("/api/fetch"), (res, h) => {
      fetchBtn.disabled = false;
      h.append(el("div", { class: "callout ok", text: `${t("settings.fetchDone")} ${JSON.stringify(res.feeds)}` }));
      refreshStatusStrip();
    }).finally(() => (fetchBtn.disabled = false));
  });

  host.append(
    el("div", { class: "card" }, [el("h3", { text: t("settings.language") }), langBtns]),
    el("div", { class: "card" }, [
      el("h3", { text: t("settings.sources") }),
      el("ul", { class: "lines" }, [
        el("li", {}, [`${t("settings.feeds")}: `, yesNo(status.feeds_available, t("settings.available"), t("settings.missing"))]),
        status.freshness ? el("li", { text: `${t("settings.freshness")}: ${status.freshness}` }) : null,
        el("li", {}, [`${t("settings.calibration")}: `, yesNo(status.calibration_on)]),
        el("li", {}, [`${t("settings.oddsKey")}: `, yesNo(status.odds_key_set)]),
      ]),
      el("div", { class: "actions", style: "margin-top:12px" }, [fetchBtn]),
      fetchResult,
    ])
  );
}

// ----------------------------------------------------------------- small builders
function pageHead(titleKey, introKey) {
  return el("div", { class: "page-head" }, [
    el("h2", { text: t(titleKey) }),
    introKey ? el("p", { text: t(introKey) }) : null,
  ]);
}
function stat(key, value) {
  return el("div", { class: "stat" }, [el("span", { class: "k", text: t(key) }), el("span", { class: "v", text: value })]);
}
function field(labelKey, input) {
  return el("div", { class: "field" }, [el("label", { text: t(labelKey) }), input]);
}
function emptyState(msg) {
  return el("div", { class: "empty" }, [el("div", { class: "big", text: t("app.title") }), el("div", { text: msg })]);
}
function errorCard(msg) {
  return el("div", { class: "card" }, [el("div", { class: "callout danger", text: `${t("common.error")}: ${msg}` })]);
}

// ----------------------------------------------------------------- shell
const TABS = [
  ["dashboard", renderDashboard],
  ["plan", renderPlan],
  ["ideal", renderIdeal],
  ["live", renderLive],
  ["odds", renderOdds],
  ["boosters", renderBoosters],
  ["calibrate", renderCalibrate],
  ["settings", renderSettings],
];
let currentTab = "dashboard";

function renderTabs() {
  const nav = document.getElementById("tabs");
  nav.replaceChildren(
    ...TABS.map(([id]) => {
      const b = el("button", { class: id === currentTab ? "active" : "", text: t(`nav.${id}`) });
      b.addEventListener("click", () => switchTab(id));
      return b;
    })
  );
}

function switchTab(id) {
  currentTab = id;
  renderTabs();
  const entry = TABS.find(([tid]) => tid === id);
  if (entry) entry[1]();
}

function renderLangSelect() {
  const wrap = document.getElementById("lang-select");
  wrap.replaceChildren(
    ...Object.keys(LANG_NAMES).map((code) => {
      const b = el("button", { class: getLang() === code ? "active" : "", text: code.toUpperCase() });
      b.addEventListener("click", () => setLanguage(code));
      return b;
    })
  );
}

function applyStaticI18n() {
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.getAttribute("data-i18n"));
  });
  document.title = t("app.title");
}

async function refreshStatusStrip() {
  const strip = document.getElementById("status-strip");
  let s;
  try { s = await apiGet("/api/status"); } catch { strip.replaceChildren(); return; }
  strip.replaceChildren(
    el("span", { class: `dot ${s.feeds_available ? "on" : "off"}`, text: `${t("settings.feeds")}: ${s.feeds_available ? t("settings.available") : t("settings.missing")}` }),
    s.freshness ? el("span", { class: "dot on", text: s.freshness }) : null,
    el("span", { class: `dot ${s.calibration_on ? "on" : "off"}`, text: `${t("settings.calibration")}: ${s.calibration_on ? t("settings.on") : t("settings.off")}` }),
    el("span", { class: `dot ${s.odds_key_set ? "on" : "off"}`, text: `${t("settings.oddsKey")}: ${s.odds_key_set ? t("settings.on") : t("settings.off")}` }),
  );
}

async function setLanguage(code) {
  setLang(code);
  try { await apiPost("/api/settings", { language: code }); } catch {}
  applyStaticI18n();
  renderLangSelect();
  renderTabs();
  refreshStatusStrip();
  switchTab(currentTab);
}

async function init() {
  // restore persisted language
  try {
    const s = await apiGet("/api/settings");
    if (s.language) setLang(s.language);
  } catch {}
  applyStaticI18n();
  renderLangSelect();
  renderTabs();
  refreshStatusStrip();
  switchTab("dashboard");
}

init();
