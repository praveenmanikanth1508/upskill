const els = {
  modeNote: document.getElementById("app-mode-note"),
  apiBaseUrl: document.getElementById("api-base-url"),
  apiConnectionStatus: document.getElementById("api-connection-status"),
  connectApiBtn: document.getElementById("connect-api-btn"),
  clearApiBtn: document.getElementById("clear-api-btn"),
  loadMode: document.getElementById("load-mode"),
  loadWindow: document.getElementById("load-window"),
  loadPartition: document.getElementById("load-partition"),
  runLoadBtn: document.getElementById("run-load-btn"),
  syncBtn: document.getElementById("sync-btn"),
  loadStatus: document.getElementById("load-status"),
  company: document.getElementById("filter-company"),
  tag: document.getElementById("filter-tag"),
  keyword: document.getElementById("filter-keyword"),
  limit: document.getElementById("filter-limit"),
  applyFiltersBtn: document.getElementById("apply-filters-btn"),
  resetFiltersBtn: document.getElementById("reset-filters-btn"),
  stats: document.getElementById("stats"),
  resultCount: document.getElementById("result-count"),
  items: document.getElementById("items"),
};

const appState = {
  mode: "api",
  staticBank: null,
  apiBase: "",
  activeJobId: null,
};

let keywordDebounceTimer = null;
const API_BASE_STORAGE_KEY = "deInterviewApiBase";
const JOB_POLL_INTERVAL_MS = 2000;
const JOB_POLL_TIMEOUT_MS = 12 * 60 * 1000;

function storageGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch (_error) {
    return null;
  }
}

function storageSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch (_error) {
    // no-op for environments where localStorage is blocked
  }
}

function storageRemove(key) {
  try {
    window.localStorage.removeItem(key);
  } catch (_error) {
    // no-op
  }
}

function normalizeApiBase(rawValue) {
  const raw = String(rawValue || "").trim();
  if (!raw) {
    return "";
  }
  try {
    const parsed = new URL(raw);
    const basePath = (parsed.pathname || "").replace(/\/+$/, "");
    return `${parsed.origin}${basePath}`;
  } catch (_error) {
    return "";
  }
}

function resolveInitialApiBase() {
  const query = new URLSearchParams(window.location.search);
  const fromQuery = normalizeApiBase(query.get("api") || "");
  if (fromQuery) {
    storageSet(API_BASE_STORAGE_KEY, fromQuery);
    return fromQuery;
  }

  const fromStorage = normalizeApiBase(storageGet(API_BASE_STORAGE_KEY) || "");
  return fromStorage;
}

appState.apiBase = resolveInitialApiBase();

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function buildApiUrl(path) {
  if (!appState.apiBase) {
    return path;
  }
  return `${appState.apiBase}${path}`;
}

async function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(buildApiUrl(path), {
    ...options,
    headers,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.error || `Request failed: ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

function setStatus(message, isError = false) {
  els.loadStatus.textContent = message;
  els.loadStatus.classList.toggle("error", Boolean(isError));
}

function setConnectionStatus(text) {
  if (els.apiConnectionStatus) {
    els.apiConnectionStatus.value = text;
  }
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function formatDate(iso) {
  if (!iso) return "n/a";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

function renderStats(stats) {
  const lastRun = stats.last_run || {};
  const cards = [
    { label: "Total Items", value: stats.total_items == null ? 0 : stats.total_items },
    { label: "Companies", value: stats.distinct_companies == null ? 0 : stats.distinct_companies },
    { label: "Latest Collected", value: formatDate(stats.latest_collected_at) },
    { label: "Latest Published", value: formatDate(stats.latest_published_at) },
    { label: "Last Run Mode", value: lastRun.mode || "n/a" },
    { label: "Last Run Window", value: lastRun.window || "n/a" },
  ];

  els.stats.innerHTML = cards
    .map(
      (card) => `
      <article class="stat-card">
        <p class="stat-label">${escapeHtml(card.label)}</p>
        <p class="stat-value">${escapeHtml(card.value)}</p>
      </article>
    `
    )
    .join("");
}

function renderItems(items, total) {
  els.resultCount.textContent = `Showing ${items.length} of ${total} matching item(s)`;
  if (!items.length) {
    els.items.innerHTML = `<p class="subtle">No results for the selected filters.</p>`;
    return;
  }

  els.items.innerHTML = items
    .map((item) => {
      const tags = Array.isArray(item.tags) ? item.tags : [];
      const source = item.source || {};
      const sourceUrl = source.entry_url || source.source_url || "#";
      return `
        <article class="item-card">
          <h3>${escapeHtml(item.question)}</h3>
          <p class="hint">${escapeHtml(item.answer_hint || "")}</p>
          <div class="chip-row">
            <span class="chip">${escapeHtml(item.company || "unknown")}</span>
            <span class="chip muted">${escapeHtml(item.role_level || "")}</span>
            <span class="chip muted">${escapeHtml(item.region || "")}</span>
          </div>
          <div class="chip-row">
            ${tags.map((tag) => `<span class="chip tag">${escapeHtml(tag)}</span>`).join("")}
          </div>
          <p class="meta">
            Source:
            <a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">
              ${escapeHtml(source.entry_title || source.source_name || sourceUrl)}
            </a>
          </p>
        </article>
      `;
    })
    .join("");
}

function getSourcePublished(item) {
  if (!item || !item.source) {
    return "";
  }
  return item.source.published_at || "";
}

function lastValue(values) {
  if (!values.length) {
    return null;
  }
  return values[values.length - 1];
}

function setSelectOptions(selectEl, rows) {
  selectEl.innerHTML =
    `<option value="all">all</option>` +
    rows
      .map((entry) => `<option value="${escapeHtml(entry.value)}">${escapeHtml(entry.value)} (${entry.count})</option>`)
      .join("");
}

function sortItemsByRecency(items) {
  const valueFor = (item) =>
    String(getSourcePublished(item) || item.collected_at || "");
  return [...items].sort((left, right) => valueFor(right).localeCompare(valueFor(left)));
}

function filterStaticItems(items) {
  const company = (els.company.value || "all").toLowerCase();
  const tag = (els.tag.value || "all").toLowerCase();
  const keyword = (els.keyword.value || "").trim().toLowerCase();
  const limit = Number.parseInt(els.limit.value || "25", 10);

  const filtered = sortItemsByRecency(items).filter((item) => {
    if (company !== "all" && (item.company || "").toLowerCase() !== company) {
      return false;
    }
    if (tag !== "all") {
      const tags = Array.isArray(item.tags) ? item.tags.map((v) => String(v).toLowerCase()) : [];
      if (!tags.includes(tag)) {
        return false;
      }
    }
    if (keyword) {
      const haystack = `${item.question || ""} ${item.answer_hint || ""}`.toLowerCase();
      if (!haystack.includes(keyword)) {
        return false;
      }
    }
    return true;
  });

  return {
    total: filtered.length,
    items: filtered.slice(0, Number.isNaN(limit) ? 25 : limit),
  };
}

function buildStaticFilters(items) {
  const companyCounts = new Map();
  const tagCounts = new Map();

  for (const item of items) {
    const company = String(item.company || "unknown").toLowerCase();
    companyCounts.set(company, (companyCounts.get(company) || 0) + 1);

    const tags = Array.isArray(item.tags) ? item.tags : [];
    for (const rawTag of tags) {
      const tag = String(rawTag || "").trim().toLowerCase();
      if (!tag) continue;
      tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
    }
  }

  const companies = [...companyCounts.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => (b.count - a.count) || a.value.localeCompare(b.value));
  const tags = [...tagCounts.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => (b.count - a.count) || a.value.localeCompare(b.value))
    .slice(0, 50);

  return { companies, tags };
}

function computeStaticStats(bank) {
  const items = Array.isArray(bank.items) ? bank.items : [];
  const metadata = bank.metadata || {};
  const companies = new Set(items.map((item) => String(item.company || "unknown").toLowerCase()));
  const latestCollectedAt = items
    .map((item) => item.collected_at)
    .filter(Boolean)
    .sort();
  const latestPublishedAt = items
    .map((item) => getSourcePublished(item))
    .filter(Boolean)
    .sort();

  return {
    total_items: items.length,
    distinct_companies: companies.size,
    latest_collected_at: lastValue(latestCollectedAt),
    latest_published_at: lastValue(latestPublishedAt),
    last_run: {
      mode: `static-data (${metadata.load_mode || "unknown"})`,
      window: metadata.window || (metadata.days_back ? `${metadata.days_back}d` : "n/a"),
    },
  };
}

async function detectMode() {
  try {
    await apiFetch("/api/health");
    appState.mode = "api";
  } catch (_error) {
    appState.mode = "static";
  }
}

function applyModeUI() {
  if (appState.mode === "api") {
    if (els.modeNote) {
      const originInfo = appState.apiBase || "same-origin";
      els.modeNote.textContent = `Mode: API backend connected (${originInfo}). You can run load jobs from this UI.`;
    }
    els.runLoadBtn.disabled = false;
    els.syncBtn.disabled = false;
    els.loadMode.disabled = false;
    els.loadWindow.disabled = false;
    els.loadPartition.disabled = false;
    setConnectionStatus(`Connected (${appState.apiBase || "same-origin"})`);
    return;
  }

  if (els.modeNote) {
    if (appState.apiBase) {
      els.modeNote.textContent =
        "Mode: static fallback. Backend API URL did not respond; using published JSON locally.";
    } else {
      els.modeNote.textContent =
        "Mode: GitHub Pages static mode. Backend load/sync is unavailable here; filters run client-side on published JSON.";
    }
  }
  els.runLoadBtn.disabled = true;
  els.syncBtn.disabled = true;
  els.loadMode.disabled = true;
  els.loadWindow.disabled = true;
  els.loadPartition.disabled = true;
  els.runLoadBtn.title = "Disabled in static mode";
  els.syncBtn.title = "Disabled in static mode";
  setConnectionStatus(appState.apiBase ? `Not connected (${appState.apiBase})` : "Static mode (no API)");
}

async function ensureStaticBankLoaded() {
  if (appState.staticBank) {
    return appState.staticBank;
  }

  const origin = window.location.origin;
  const pathname = window.location.pathname;
  const segments = pathname.split("/").filter(Boolean);
  const candidates = new Set();

  // Handles normal paths like /upskill/ and /upskill/index.html.
  candidates.add(new URL("./data/interview_bank.json", window.location.href).toString());
  candidates.add(new URL("data/interview_bank.json", window.location.href).toString());

  // Handles GitHub Pages project URLs opened as /repo (without trailing slash).
  if (window.location.hostname.endsWith(".github.io") && segments.length >= 1) {
    candidates.add(`${origin}/${segments[0]}/data/interview_bank.json`);
  }

  // Root fallback (mostly for user/organization Pages sites).
  candidates.add(`${origin}/data/interview_bank.json`);

  const errors = [];
  for (const url of candidates) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        errors.push(`${url} -> HTTP ${response.status}`);
        continue;
      }
      const payload = await response.json();
      if (!payload || typeof payload !== "object") {
        errors.push(`${url} -> invalid JSON object`);
        continue;
      }
      appState.staticBank = payload;
      return payload;
    } catch (error) {
      errors.push(`${url} -> ${error.message}`);
    }
  }

  throw new Error(`Could not load static data bundle. Tried: ${errors.join(" | ")}`);
}

async function loadFilters() {
  if (appState.mode === "api") {
    const payload = await apiFetch("/api/filters");
    setSelectOptions(els.company, payload.companies || []);
    setSelectOptions(els.tag, payload.tags || []);
    return;
  }

  const bank = await ensureStaticBankLoaded();
  const items = Array.isArray(bank.items) ? bank.items : [];
  const filters = buildStaticFilters(items);
  setSelectOptions(els.company, filters.companies);
  setSelectOptions(els.tag, filters.tags);
}

async function loadItems() {
  if (appState.mode === "api") {
    const query = new URLSearchParams();
    if (els.company.value && els.company.value !== "all") {
      query.set("company", els.company.value);
    }
    if (els.tag.value && els.tag.value !== "all") {
      query.set("tag", els.tag.value);
    }
    if (els.keyword.value.trim()) {
      query.set("keyword", els.keyword.value.trim());
    }
    query.set("limit", els.limit.value || "25");

    const payload = await apiFetch(`/api/items?${query.toString()}`);
    renderItems(payload.items || [], payload.total == null ? 0 : payload.total);
    return;
  }

  const bank = await ensureStaticBankLoaded();
  const items = Array.isArray(bank.items) ? bank.items : [];
  const filtered = filterStaticItems(items);
  renderItems(filtered.items, filtered.total);
}

async function loadStats() {
  if (appState.mode === "api") {
    const payload = await apiFetch("/api/stats");
    renderStats(payload);
    return;
  }

  const bank = await ensureStaticBankLoaded();
  renderStats(computeStaticStats(bank));
}

async function pollLoadJob(jobId) {
  const startedAt = Date.now();
  while (true) {
    const job = await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    const status = String(job.status || "");
    if (status === "queued" || status === "running") {
      const elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000);
      setStatus(`Load job ${jobId} is ${status}... (${elapsedSeconds}s elapsed)`);
      if (Date.now() - startedAt > JOB_POLL_TIMEOUT_MS) {
        throw new Error(`Load job ${jobId} timed out after ${Math.floor(JOB_POLL_TIMEOUT_MS / 1000)}s`);
      }
      await sleep(JOB_POLL_INTERVAL_MS);
      continue;
    }
    return job;
  }
}

async function runLoad() {
  if (appState.mode !== "api") {
    setStatus("Load unavailable in static mode. Use GitHub Action or local backend.", true);
    return;
  }

  setStatus("Running backend load, this may take a while...");
  const body = {
    mode: els.loadMode.value,
    window: els.loadWindow.value,
    incremental_partition: els.loadPartition.value,
  };
  try {
    const payload = await apiFetch("/api/load/async", {
      method: "POST",
      body: JSON.stringify(body),
    });
    const jobId = payload.job_id;
    if (!jobId) {
      throw new Error("No job_id returned from backend.");
    }
    appState.activeJobId = jobId;
    setStatus(`Load job submitted: ${jobId}. Waiting for completion...`);

    const job = await pollLoadJob(jobId);
    if (job.status !== "succeeded") {
      const errorMessage = job.error || "Unknown job failure.";
      const stderr = job.load_result && job.load_result.stderr ? `\n${job.load_result.stderr}` : "";
      setStatus(`Load job failed: ${errorMessage}${stderr}`, true);
      return;
    }

    const load = job.load_result || {};
    const sync = job.sync_result || {};
    setStatus(
      [
        `Load job ${jobId} completed.`,
        `Return code: ${load.return_code}`,
        `Synced items: ${sync.item_count == null ? 0 : sync.item_count} (inserted=${sync.inserted_count == null ? 0 : sync.inserted_count}, updated=${sync.updated_count == null ? 0 : sync.updated_count})`,
      ].join("\n")
    );
    await Promise.all([loadFilters(), loadStats(), loadItems()]);
  } catch (error) {
    setStatus(`Load failed: ${error.message}`, true);
  }
}

async function runSyncOnly() {
  if (appState.mode !== "api") {
    setStatus("Sync unavailable in static mode.", true);
    return;
  }

  setStatus("Syncing existing JSON into SQLite...");
  try {
    const payload = await apiFetch("/api/sync", { method: "POST" });
    const sync = payload.sync_result || {};
    setStatus(
      `Sync complete. items=${sync.item_count == null ? 0 : sync.item_count}, inserted=${sync.inserted_count == null ? 0 : sync.inserted_count}, updated=${sync.updated_count == null ? 0 : sync.updated_count}`
    );
    await Promise.all([loadFilters(), loadStats(), loadItems()]);
  } catch (error) {
    setStatus(`Sync failed: ${error.message}`, true);
  }
}

function resetFilters() {
  els.company.value = "all";
  els.tag.value = "all";
  els.keyword.value = "";
  els.limit.value = "25";
  loadItems().catch((error) => setStatus(error.message, true));
}

async function refreshAppData() {
  try {
    await detectMode();
    applyModeUI();
    await Promise.all([loadFilters(), loadStats()]);
    await loadItems();
    if (appState.mode === "static") {
      if (appState.apiBase) {
        setStatus("Backend URL unreachable, running in static fallback mode with published JSON.", true);
      } else {
        setStatus("Static mode ready. Use filters and click Apply (or type keyword). Backend load actions are disabled on Pages.");
      }
    } else {
      setStatus("Ready.");
    }
  } catch (error) {
    setStatus(`Initialization failed: ${error.message}`, true);
  }
}

function setApiBaseInputValue() {
  if (els.apiBaseUrl) {
    els.apiBaseUrl.value = appState.apiBase;
  }
}

async function connectApiBase() {
  if (!els.apiBaseUrl) {
    return;
  }
  const normalized = normalizeApiBase(els.apiBaseUrl.value);
  if (!normalized) {
    setStatus("Please enter a valid API base URL (example: https://my-backend.onrender.com).", true);
    return;
  }

  appState.apiBase = normalized;
  storageSet(API_BASE_STORAGE_KEY, normalized);
  appState.staticBank = null;
  setApiBaseInputValue();
  await refreshAppData();
}

async function clearApiBase() {
  appState.apiBase = "";
  storageRemove(API_BASE_STORAGE_KEY);
  appState.staticBank = null;
  setApiBaseInputValue();
  await refreshAppData();
}

function applyFiltersWithStatus() {
  loadItems().catch((error) => setStatus(error.message, true));
}

async function init() {
  setApiBaseInputValue();
  await refreshAppData();
}

els.runLoadBtn.addEventListener("click", () => runLoad());
els.syncBtn.addEventListener("click", () => runSyncOnly());
els.applyFiltersBtn.addEventListener("click", applyFiltersWithStatus);
els.resetFiltersBtn.addEventListener("click", resetFilters);
if (els.connectApiBtn) {
  els.connectApiBtn.addEventListener("click", () => {
    connectApiBase().catch((error) => setStatus(error.message, true));
  });
}
if (els.clearApiBtn) {
  els.clearApiBtn.addEventListener("click", () => {
    clearApiBase().catch((error) => setStatus(error.message, true));
  });
}
if (els.apiBaseUrl) {
  els.apiBaseUrl.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      connectApiBase().catch((error) => setStatus(error.message, true));
    }
  });
}
els.company.addEventListener("change", applyFiltersWithStatus);
els.tag.addEventListener("change", applyFiltersWithStatus);
els.limit.addEventListener("change", applyFiltersWithStatus);
els.keyword.addEventListener("input", () => {
  if (keywordDebounceTimer) {
    window.clearTimeout(keywordDebounceTimer);
  }
  keywordDebounceTimer = window.setTimeout(() => {
    applyFiltersWithStatus();
  }, 220);
});
els.keyword.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    if (keywordDebounceTimer) {
      window.clearTimeout(keywordDebounceTimer);
      keywordDebounceTimer = null;
    }
    applyFiltersWithStatus();
  }
});

init();
