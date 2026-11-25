"use strict";

const DATASET_MEAN = 5.82;
const TOKEN_KEY = "wqp_token";
const USER_KEY = "wqp_user";

// name, label, unit, min, max, step
const GROUPS = {
  "Acidity": [
    ["fixed_acidity", "Fixed acidity", "g/L", 3, 16, 0.1],
    ["volatile_acidity", "Volatile acidity", "g/L", 0, 2, 0.01],
    ["citric_acid", "Citric acid", "g/L", 0, 2, 0.01],
    ["pH", "pH", "", 2.5, 4.5, 0.01],
  ],
  "Sugar & minerals": [
    ["residual_sugar", "Residual sugar", "g/L", 0, 70, 0.1],
    ["chlorides", "Chlorides", "g/L", 0, 1, 0.001],
    ["sulphates", "Sulphates", "g/L", 0, 2.5, 0.01],
    ["density", "Density", "g/cm³", 0.98, 1.05, 0.0001],
  ],
  "Sulfur dioxide & body": [
    ["free_sulfur_dioxide", "Free SO₂", "mg/L", 0, 300, 1],
    ["total_sulfur_dioxide", "Total SO₂", "mg/L", 0, 450, 1],
    ["alcohol", "Alcohol", "% vol", 7, 16, 0.1],
  ],
};

const PRESETS = {
  "Everyday red": { fixed_acidity: 7.4, volatile_acidity: 0.70, citric_acid: 0.00, residual_sugar: 1.9, chlorides: 0.076,
    free_sulfur_dioxide: 11, total_sulfur_dioxide: 34, density: 0.9978, pH: 3.51, sulphates: 0.56, alcohol: 9.4, is_red: 1 },
  "Structured red": { fixed_acidity: 8.5, volatile_acidity: 0.32, citric_acid: 0.42, residual_sugar: 2.3, chlorides: 0.065,
    free_sulfur_dioxide: 12, total_sulfur_dioxide: 30, density: 0.9960, pH: 3.30, sulphates: 0.78, alcohol: 12.6, is_red: 1 },
  "Crisp white": { fixed_acidity: 6.8, volatile_acidity: 0.24, citric_acid: 0.34, residual_sugar: 3.0, chlorides: 0.036,
    free_sulfur_dioxide: 35, total_sulfur_dioxide: 120, density: 0.9915, pH: 3.20, sulphates: 0.48, alcohol: 12.0, is_red: 0 },
  "Faulty wine (vinegar notes)": { fixed_acidity: 7.8, volatile_acidity: 1.10, citric_acid: 0.02, residual_sugar: 2.0, chlorides: 0.09,
    free_sulfur_dioxide: 8, total_sulfur_dioxide: 25, density: 0.9975, pH: 3.45, sulphates: 0.48, alcohol: 9.2, is_red: 1 },
};

const LABELS = {
  Poor: ["#B23A48", "Noticeable defects. Unlikely to please most tasters."],
  Average: ["#C07A2C", "Sound but unremarkable, in line with most wines in the dataset."],
  Good: ["#5E8C4A", "Above average. Balanced profile with few weaknesses."],
  Excellent: ["#2F6B4F", "Top of the range. Few wines in the dataset score this high."],
};

const $ = (id) => document.getElementById(id);
const FEATURES = Object.values(GROUPS).flat();
let token = localStorage.getItem(TOKEN_KEY);
let authMode = "login";

// ------------------------------------------------------------------ API

async function api(path, { method = "GET", body } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  let resp;
  try {
    resp = await fetch(path, { method, headers, body: body ? JSON.stringify(body) : undefined });
  } catch {
    throw { status: 0, message: "API unreachable" };
  }
  const data = resp.status === 204 ? null : await resp.json().catch(() => null);
  if (!resp.ok) {
    const detail = data && data.detail;
    let message = typeof detail === "string" ? detail : "Request failed";
    let fields = [];
    if (Array.isArray(detail)) {
      fields = detail.map((e) => e.loc[e.loc.length - 1]);
      message = detail.map((e) => {
        const field = e.loc.filter((p) => p !== "body").join(".");
        return field ? `${field}: ${e.msg}` : e.msg;
      }).join(" · ");
    }
    throw { status: resp.status, message, fields };
  }
  return data;
}

// ------------------------------------------------------------------ form

function buildForm() {
  const container = $("fields");
  for (const [group, specs] of Object.entries(GROUPS)) {
    const title = document.createElement("div");
    title.className = "group-title";
    title.textContent = group;
    const grid = document.createElement("div");
    grid.className = "grid";
    for (const [name, label, unit, min, max, step] of specs) {
      grid.insertAdjacentHTML("beforeend", `
        <div class="field">
          <label for="f-${name}">${label}<span class="unit">${unit}</span></label>
          <input id="f-${name}" name="${name}" type="number" inputmode="decimal"
                 min="${min}" max="${max}" step="${step}" required>
        </div>`);
    }
    container.append(title, grid);
  }
  const select = $("preset");
  for (const name of Object.keys(PRESETS)) select.add(new Option(name, name));
  select.addEventListener("change", () => applyPreset(select.value));
  applyPreset(select.value);
}

function applyPreset(name) {
  const values = PRESETS[name];
  for (const [field] of FEATURES) $(`f-${field}`).value = values[field];
  $(values.is_red ? "type-red" : "type-white").checked = true;
  clearErrors();
}

function readForm() {
  const features = {};
  const invalid = [];
  for (const [name, label, , min, max] of FEATURES) {
    const input = $(`f-${name}`);
    const value = input.value === "" ? NaN : Number(input.value);
    if (!Number.isFinite(value) || value < min || value > max) invalid.push([name, `${label} must be between ${min} and ${max}`]);
    features[name] = value;
  }
  features.is_red = Number(document.querySelector("input[name=is_red]:checked").value);
  const wineName = $("wine-name").value.trim();
  if (wineName) features.wine_name = wineName;
  if (features.free_sulfur_dioxide > features.total_sulfur_dioxide) {
    invalid.push(["free_sulfur_dioxide", "Free SO₂ cannot exceed total SO₂"]);
  }
  return { features, invalid };
}

function clearErrors() {
  $("form-error").hidden = true;
  document.querySelectorAll("input.invalid").forEach((el) => el.classList.remove("invalid"));
}

function showErrors(message, fields = []) {
  $("form-error").textContent = message;
  $("form-error").hidden = false;
  fields.forEach((f) => $(`f-${f}`)?.classList.add("invalid"));
}

async function onPredict(event) {
  event.preventDefault();
  clearErrors();
  const { features, invalid } = readForm();
  if (invalid.length) {
    showErrors(invalid.map(([, msg]) => msg).join(" · "), invalid.map(([name]) => name));
    return;
  }
  const button = $("predict-btn");
  button.disabled = true;
  button.textContent = "Tasting…";
  try {
    renderResult(await api("/predict", { method: "POST", body: features }));
    if (token) loadHistory();
  } catch (err) {
    showErrors(`Prediction failed: ${err.message}`, err.fields);
  } finally {
    button.disabled = false;
    button.textContent = "Predict quality";
  }
}

// ------------------------------------------------------------------ result

function renderResult(r) {
  const [color, comment] = LABELS[r.label] || ["#7B1E3A", ""];
  $("result-empty").hidden = true;
  $("result").hidden = false;
  $("result-name").textContent = r.wine_name || "";
  $("result-name").hidden = !r.wine_name;
  $("score").textContent = r.quality.toFixed(1);
  $("label").textContent = r.label;
  $("label").style.background = color;
  $("comment").textContent = comment;
  const bar = $("gauge-bar");
  bar.style.stroke = color;
  requestAnimationFrame(() => { bar.style.strokeDasharray = `${r.quality * 10} 100`; });
  const delta = r.quality - DATASET_MEAN;
  $("delta").textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`;
  $("delta").className = `stat ${delta >= 0 ? "delta-up" : "delta-down"}`;
  $("result-version").textContent = r.model_version;
  $("saved-note").textContent = r.prediction_id ? "✓ Saved to your history." : "Sign in to save your analyses.";
}

function placeMeanMarker() {
  // Gold tick on the gauge at the dataset mean.
  const angle = Math.PI * (1 - DATASET_MEAN / 10);
  const [cx, cy, r1, r2] = [100, 100, 72, 98];
  const line = $("gauge-mean");
  line.setAttribute("x1", cx + r1 * Math.cos(angle));
  line.setAttribute("y1", cy - r1 * Math.sin(angle));
  line.setAttribute("x2", cx + r2 * Math.cos(angle));
  line.setAttribute("y2", cy - r2 * Math.sin(angle));
}

// ------------------------------------------------------------------ model info

async function loadModelInfo() {
  const dot = $("model-dot");
  try {
    const info = await api("/model-info");
    if (info.ready) {
      const v = /^\d+$/.test(info.version) ? `v${info.version}` : info.version;
      $("model-status").textContent = `Model ${info.model_name} ${v} · ${info.alias}`;
      dot.className = "dot ok";
      $("predict-btn").disabled = false;
    } else {
      $("model-status").textContent = "Model not ready";
      dot.className = "dot";
      $("predict-btn").disabled = true;
    }
  } catch {
    $("model-status").textContent = "API offline";
    dot.className = "dot down";
    $("predict-btn").disabled = true;
  }
}

// ------------------------------------------------------------------ auth

function setSession(newToken, username) {
  token = newToken;
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, username);
  } else {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  }
  renderAccount();
}

function renderAccount() {
  const signedIn = Boolean(token);
  $("open-auth").hidden = signedIn;
  $("signed-in").hidden = !signedIn;
  $("username").textContent = localStorage.getItem(USER_KEY) || "";
  $("history-locked").hidden = signedIn;
  $("history-content").hidden = !signedIn;
  $("clear-history").hidden = !signedIn;
  if (signedIn) loadHistory();
}

function setAuthMode(mode) {
  authMode = mode;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
  $("auth-submit").textContent = mode === "login" ? "Sign in" : "Create account";
  $("auth-hint").textContent = mode === "login"
    ? "Welcome back."
    : "3–32 characters (letters, digits, . _ -), password of at least 8 characters.";
  $("auth-password").autocomplete = mode === "login" ? "current-password" : "new-password";
  $("auth-error").hidden = true;
}

async function onAuth(event) {
  event.preventDefault();
  const username = $("auth-username").value.trim();
  const password = $("auth-password").value;
  if (!username || !password) {
    $("auth-error").textContent = "Enter a username and a password.";
    $("auth-error").hidden = false;
    return;
  }
  try {
    const path = authMode === "login" ? "/auth/login" : "/auth/signup";
    const result = await api(path, { method: "POST", body: { username, password } });
    setSession(result.access_token, result.username);
    $("auth-dialog").close();
    $("auth-form").reset();
  } catch (err) {
    $("auth-error").textContent = err.message;
    $("auth-error").hidden = false;
  }
}

async function onLogout() {
  try { await api("/auth/logout", { method: "POST" }); } catch { /* already invalid server side */ }
  setSession(null);
}

// ------------------------------------------------------------------ history

async function loadHistory() {
  let items;
  try {
    items = await api("/history?limit=100");
  } catch (err) {
    if (err.status === 401) setSession(null);
    return;
  }
  const scores = items.map((i) => i.quality);
  $("kpi-count").textContent = items.length;
  $("kpi-avg").textContent = items.length ? (scores.reduce((a, b) => a + b, 0) / items.length).toFixed(2) : "–";
  $("kpi-best").textContent = items.length ? Math.max(...scores).toFixed(2) : "–";
  drawChart([...items].reverse());

  const rows = $("history-rows");
  rows.replaceChildren();
  if (!items.length) {
    rows.innerHTML = `<tr><td colspan="8" class="muted">No analyses yet. Your predictions will appear here.</td></tr>`;
    return;
  }
  for (const item of items) {
    const tr = document.createElement("tr");
    const date = new Date(item.created_at).toLocaleString();
    const cells = [
      item.wine_name || "—",
      date,
      item.inputs.is_red ? "Red" : "White",
      null,
      item.label,
      `${item.inputs.alcohol.toFixed(1)} %`,
      item.inputs.volatile_acidity.toFixed(2),
      item.model_version,
    ];
    cells.forEach((value, idx) => {
      const td = document.createElement("td");
      if (idx === 3) {
        td.innerHTML = `<span class="bar" style="width:${item.quality * 8}px"></span>${item.quality.toFixed(2)}`;
      } else {
        td.textContent = value;
      }
      tr.append(td);
    });
    rows.append(tr);
  }
}

function drawChart(items) {
  const svg = $("history-chart");
  const [w, h, pad] = [svg.clientWidth || 600, 160, 14];
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const y = (v) => h - pad - (v / 10) * (h - 2 * pad);
  const x = (i) => (items.length < 2 ? w / 2 : pad + (i * (w - 2 * pad)) / (items.length - 1));
  const meanY = y(DATASET_MEAN);
  const points = items.map((it, i) => `${x(i)},${y(it.quality)}`).join(" ");
  svg.innerHTML = `
    <line x1="0" x2="${w}" y1="${meanY}" y2="${meanY}" stroke="#C9BCC0" stroke-dasharray="4 4"/>
    <text x="${w - 6}" y="${meanY - 5}" text-anchor="end" font-size="11" fill="#9A8A8F">dataset mean</text>
    <polyline points="${points}" fill="none" stroke="#7B1E3A" stroke-width="2.5" stroke-linejoin="round"/>
    ${items.map((it, i) => `<circle cx="${x(i)}" cy="${y(it.quality)}" r="4.5" fill="#C9A24B" stroke="#7B1E3A" stroke-width="1.5"/>`).join("")}`;
}

async function onClearHistory() {
  if (!confirm("Delete your whole history?")) return;
  await api("/history", { method: "DELETE" }).catch(() => {});
  loadHistory();
}

// ------------------------------------------------------------------ init

document.addEventListener("DOMContentLoaded", () => {
  buildForm();
  placeMeanMarker();
  $("predict-form").addEventListener("submit", onPredict);
  $("open-auth").addEventListener("click", () => { setAuthMode("login"); $("auth-dialog").showModal(); });
  $("auth-cancel").addEventListener("click", () => $("auth-dialog").close());
  $("auth-form").addEventListener("submit", onAuth);
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setAuthMode(t.dataset.mode)));
  $("logout").addEventListener("click", onLogout);
  $("clear-history").addEventListener("click", onClearHistory);
  renderAccount();
  loadModelInfo();
});
