const els = {
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

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
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

async function loadFilters() {
  const payload = await apiFetch("/api/filters");
  const companies = payload.companies || [];
  const tags = payload.tags || [];

  els.company.innerHTML = `<option value="all">all</option>` +
    companies
      .map((entry) => `<option value="${escapeHtml(entry.value)}">${escapeHtml(entry.value)} (${entry.count})</option>`)
      .join("");

  els.tag.innerHTML = `<option value="all">all</option>` +
    tags
      .map((entry) => `<option value="${escapeHtml(entry.value)}">${escapeHtml(entry.value)} (${entry.count})</option>`)
      .join("");
}

async function loadItems() {
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
}

async function loadStats() {
  const payload = await apiFetch("/api/stats");
  renderStats(payload);
}

async function runLoad() {
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
