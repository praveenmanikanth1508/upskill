const els = {
  modeNote: document.getElementById("app-mode-note"),
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
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, {
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

function formatDate(iso) {
  if (!iso) return "n/a";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

function renderStats(stats) {
  const lastRun = stats.last_run || {};
  const cards = [
    { label: "Total Items", value: stats.total_items ?? 0 },
    { label: "Companies", value: stats.distinct_companies ?? 0 },
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

function setSelectOptions(selectEl, rows) {
  selectEl.innerHTML =
    `<option value="all">all</option>` +
    rows
      .map((entry) => `<option value="${escapeHtml(entry.value)}">${escapeHtml(entry.value)} (${entry.count})</option>`)
      .join("");
}

function sortItemsByRecency(items) {
  const valueFor = (item) =>
    String(item?.source?.published_at || item?.collected_at || "");
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
    .sort()
    .at(-1) || null;
  const latestPublishedAt = items
    .map((item) => item?.source?.published_at)
    .filter(Boolean)
    .sort()
    .at(-1) || null;

  return {
    total_items: items.length,
    distinct_companies: companies.size,
    latest_collected_at: latestCollectedAt,
    latest_published_at: latestPublishedAt,
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
      els.modeNote.textContent = "Mode: API backend connected. You can run load jobs from this UI.";
    }
    return;
  }

  if (els.modeNote) {
    els.modeNote.textContent =
      "Mode: GitHub Pages static mode. Backend load/sync is unavailable here; filters run client-side on published JSON.";
  }
  els.runLoadBtn.disabled = true;
  els.syncBtn.disabled = true;
  els.runLoadBtn.title = "Disabled in static mode";
  els.syncBtn.title = "Disabled in static mode";
}

async function ensureStaticBankLoaded() {
  if (appState.staticBank) {
    return appState.staticBank;
  }
  const bankUrl = new URL("./data/interview_bank.json", window.location.href).toString();
  const response = await fetch(bankUrl, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not load static data bundle (data/interview_bank.json).");
  }
  const payload = await response.json();
  if (!payload || typeof payload !== "object") {
    throw new Error("Static bank payload is invalid.");
  }
  appState.staticBank = payload;
  return payload;
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
    renderItems(payload.items || [], payload.total ?? 0);
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
    const payload = await apiFetch("/api/load", {
      method: "POST",
      body: JSON.stringify(body),
    });
    const load = payload.load_result || {};
    const sync = payload.sync_result || {};
    setStatus(
      [
        payload.message || "Load completed.",
        `Return code: ${load.return_code}`,
        `Synced items: ${sync.item_count ?? 0} (inserted=${sync.inserted_count ?? 0}, updated=${sync.updated_count ?? 0})`,
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
      `Sync complete. items=${sync.item_count ?? 0}, inserted=${sync.inserted_count ?? 0}, updated=${sync.updated_count ?? 0}`
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

async function init() {
  try {
    await detectMode();
    applyModeUI();
    await Promise.all([loadFilters(), loadStats()]);
    await loadItems();
    setStatus("Ready.");
  } catch (error) {
    setStatus(`Initialization failed: ${error.message}`, true);
  }
}

els.runLoadBtn.addEventListener("click", () => runLoad());
els.syncBtn.addEventListener("click", () => runSyncOnly());
els.applyFiltersBtn.addEventListener("click", () => {
  loadItems().catch((error) => setStatus(error.message, true));
});
els.resetFiltersBtn.addEventListener("click", resetFilters);

init();
