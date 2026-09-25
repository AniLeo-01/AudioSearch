// AudioSearch UI - vanilla JS, no build step. All untrusted text is inserted with textContent.
"use strict";

const $ = (sel) => document.querySelector(sel);
const EXAMPLES = [
  "cesium",
  '"firing room one"',
  "why do clocks tick faster on Mars?",
  "people who get lost in their own neighborhood",
  "Zubaire",
  "Guppy role:guest",
  "mentors who shaped their career",
];
const CHANNEL_LABEL = { lexical: "keyword", dense: "semantic", rerank: "rerank" };

const state = { mode: "hybrid", files: new Map(), transcript: [], currentFile: null, activeIdx: -1, playingRank: null };

function fmt(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec - m * 60).toFixed(1).padStart(4, "0");
  return `${String(m).padStart(2, "0")}:${s}`;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function speakerLabel(role, name, label) {
  const roleName = role === "host" ? "Host" : role === "guest" ? "Guest" : label;
  return name ? `${roleName} · ${name}` : roleName;
}

function highlighted(text, spans) {
  const frag = document.createDocumentFragment();
  let pos = 0;
  for (const [a, b] of spans || []) {
    if (a > pos) frag.append(text.slice(pos, a));
    frag.append(el("mark", {}, text.slice(a, b)));
    pos = b;
  }
  frag.append(text.slice(pos));
  return frag;
}

// ------------------------------------------------------------------------------------------ search
async function search(pushHistory = true) {
  const q = $("#q").value.trim();
  if (!q) return;
  const params = new URLSearchParams({ q, mode: state.mode, k: "10" });
  if ($("#role").value) params.set("role", $("#role").value);
  if ($("#file").value) params.append("file_id", $("#file").value);
  if ($("#rerank").checked) params.set("rerank", "true");
  if (pushHistory) history.pushState(null, "", `/?${params}`);
  $("#status").textContent = "Searching…";
  let data;
  try {
    const res = await fetch(`/api/search?${params}`);
    data = await res.json();
    if (!res.ok) throw new Error(data.detail?.[0]?.msg || data.detail || res.statusText);
  } catch (err) {
    $("#status").textContent = `Error: ${err.message}`;
    $("#results").replaceChildren();
    return;
  }
  renderResults(data);
}

function renderResults(data) {
  const status = $("#status");
  status.replaceChildren(
    `${data.hits.length} moment${data.hits.length === 1 ? "" : "s"} · ${Math.round(data.timings_ms.total)} ms · intent `,
    el("code", {}, data.intent),
    ` · mode `,
    el("code", {}, data.mode),
  );
  if (data.expansions.length) {
    const parts = data.expansions.map((e) => `“${e.source}” → “${e.term}”`).join(", ");
    status.append(el("div", { class: "soundslike" }, `Sounds-like: also matched ${parts}`));
  }
  $("#empty").hidden = data.hits.length > 0;
  $("#results").replaceChildren(...data.hits.map(renderHit));
}

function renderHit(h) {
  const why = Object.entries(h.channels).map(([ch, r]) => el("span", { class: `badge ${ch}` }, `${CHANNEL_LABEL[ch] || ch} #${r}`));
  const card = el(
    "li",
    { class: "hit", "data-rank": h.rank },
    el(
      "div",
      { class: "hit-head" },
      el("span", { class: "rank" }, `#${h.rank}`),
      el("button", { class: "play", type: "button", title: "Play from this moment", onclick: () => play(h) }, `▶ ${h.timestamp}`),
      el("span", { class: "file" }, h.file_title, el("small", {}, `${h.file_id}.mp3`)),
      el("span", { class: `speaker ${h.speaker_role}` }, speakerLabel(h.speaker_role, h.speaker_name, h.speaker)),
    ),
    el("p", { class: "quote" }, highlighted(h.text, h.highlights)),
    el(
      "details",
      {},
      el("summary", {}, `Context ${fmt(h.passage_start)}–${fmt(h.passage_end)}`),
      el("p", {}, h.passage_text),
    ),
    el("div", { class: "why" }, ...why, el("span", { class: "badge" }, `match @ ${fmt(h.match_time)}`)),
  );
  return card;
}

// ------------------------------------------------------------------------------------------ playback
async function loadTranscript(fileId) {
  const res = await fetch(`/api/files/${encodeURIComponent(fileId)}/transcript`);
  state.transcript = res.ok ? await res.json() : [];
  const meta = state.files.get(fileId);
  const roles = new Map((meta?.speakers || []).map((s) => [s.label, s]));
  $("#transcript").replaceChildren(
    ...state.transcript.map((u, i) => {
      const s = roles.get(u.speaker);
      return el(
        "li",
        { "data-i": i, onclick: () => seek(u.start) },
        el("span", { class: "t" }, fmt(u.start)),
        el("span", {}, el("span", { class: `who ${s?.role || ""}` }, s?.display_name || s?.role || u.speaker), u.text),
      );
    }),
  );
  state.activeIdx = -1;
}

function seek(t) {
  const audio = $("#audio");
  audio.currentTime = Math.max(0, t - 0.3);
  audio.play().catch(() => {});
}

async function play(h) {
  const audio = $("#audio");
  document.querySelectorAll(".hit.playing").forEach((n) => n.classList.remove("playing"));
  document.querySelector(`.hit[data-rank="${h.rank}"]`)?.classList.add("playing");
  $("#np-title").textContent = h.file_title;
  $("#np-meta").textContent = `${fmt(h.start)} · ${speakerLabel(h.speaker_role, h.speaker_name, h.speaker)}`;
  if (state.currentFile !== h.file_id) {
    state.currentFile = h.file_id;
    const ready = new Promise((resolve) => audio.addEventListener("loadedmetadata", resolve, { once: true }));
    audio.preload = "metadata";  // with preload="none" the element would never fetch metadata
    audio.src = h.audio_url;
    audio.load();
    await Promise.all([ready, loadTranscript(h.file_id)]);
  }
  seek(h.start);
  syncTranscript();
}

function syncTranscript() {
  const t = $("#audio").currentTime;
  const tr = state.transcript;
  let lo = 0, hi = tr.length - 1, idx = -1;
  while (lo <= hi) {  // last utterance starting at or before t
    const mid = (lo + hi) >> 1;
    if (tr[mid].start <= t + 0.05) { idx = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (idx === state.activeIdx) return;
  const list = $("#transcript");
  list.querySelector("li.active")?.classList.remove("active");
  const node = list.querySelector(`li[data-i="${idx}"]`);
  if (node) {
    node.classList.add("active");
    if ($("#follow").checked) {  // scroll only the transcript pane, never the page
      list.scrollTo({ top: node.offsetTop - list.clientHeight / 2 + node.clientHeight / 2, behavior: "smooth" });
    }
  }
  state.activeIdx = idx;
}

// ------------------------------------------------------------------------------------------ init
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".segmented button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
}

async function init() {
  $("#search-form").addEventListener("submit", (e) => { e.preventDefault(); search(); });
  document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => { setMode(b.dataset.mode); search(); }));
  for (const id of ["#role", "#file", "#rerank"]) $(id).addEventListener("change", () => search());
  $("#audio").addEventListener("timeupdate", syncTranscript);
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== $("#q")) { e.preventDefault(); $("#q").focus(); }
  });
  $(".examples").replaceChildren(
    ...EXAMPLES.map((q) => el("button", { type: "button", onclick: () => { $("#q").value = q; search(); } }, q)),
  );
  try {
    const files = await (await fetch("/api/files")).json();
    for (const f of files) {
      state.files.set(f.file_id, f);
      $("#file").append(el("option", { value: f.file_id }, `${f.title} (${fmt(f.duration_sec)})`));
    }
  } catch { /* the search box still works without the file list */ }
  window.addEventListener("popstate", () => fromUrl(false));
  fromUrl(false);
}

function fromUrl(push) {
  const p = new URLSearchParams(location.search);
  if (!p.get("q")) return;
  $("#q").value = p.get("q");
  setMode(p.get("mode") || "hybrid");
  $("#role").value = p.get("role") || "";
  $("#file").value = p.get("file_id") || "";
  $("#rerank").checked = p.get("rerank") === "true";
  search(push);
}

document.addEventListener("DOMContentLoaded", init);
