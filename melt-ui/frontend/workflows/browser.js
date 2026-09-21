(function () {
  function nowIso() {
    return new Date().toISOString();
  }

  function uid() {
    return `wf_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  }

  function safeJsonParse(raw, fallback) {
    try {
      return JSON.parse(raw);
    } catch {
      return fallback;
    }
  }

  function dateLabel(iso) {
    if (!iso) return "";
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return iso;
    }
  }

  class WorkflowBrowser {
    constructor(options) {
      this.graph = options.graph;
      this.canvas = options.canvas;
      this.setStatus = options.setStatus || (() => {});
      this.saveWorkingGraph = options.saveWorkingGraph || (() => {});
      this.autosaveGraphKey = options.autosaveGraphKey || "lg:graph:v1";
      this.autosaveViewKey = options.autosaveViewKey || "lg:view:v1";
      this.templatesManifestUrl = options.templatesManifestUrl;
      this.templatesBaseUrl = options.templatesBaseUrl;

      this._templates = [];
      this._templatesLoaded = false;
      this._savedIndex = [];
      this._savedDirectory = "";
      this._activeTab = "templates";
      this._search = "";
      this._selected = null;

      this._ensureUi();
      this._bindUi();
    }

    _ensureUi() {
      if (document.getElementById("wfOverlay")) return;

      const root = document.createElement("div");
      root.innerHTML = `
        <div id="wfOverlay" aria-hidden="true">
          <div id="wfModal" role="dialog" aria-label="Workflow Browser" aria-modal="true">
            <div id="wfHeader">
              <h3>Workflow Browser</h3>
              <span class="wf-muted">Templates + your saved workflows</span>
              <div class="wf-spacer"></div>
              <button class="wf-btn" id="wfRefreshBtn">Refresh</button>
              <button class="wf-btn" id="wfCloseBtn">Close</button>
            </div>
            <div id="wfControls">
              <button class="wf-tab active" id="wfTabTemplates">Templates</button>
              <button class="wf-tab" id="wfTabSaved">My Workflows</button>
              <input id="wfSearch" type="text" placeholder="Search templates or saved workflows" />
              <button class="wf-btn primary" id="wfSaveAsBtn">Save Current As</button>
            </div>
            <div id="wfBody">
              <div id="wfList"></div>
              <div id="wfDetail"></div>
            </div>
          </div>
        </div>

        <div id="wfConfirm" aria-hidden="true">
          <div id="wfConfirmCard">
            <h4 style="margin:0 0 6px">Unsaved graph changes</h4>
            <div class="wf-muted">Your current canvas differs from the last autosaved state.</div>
            <div id="wfConfirmActions">
              <button class="wf-btn" id="wfConfirmCancel">Cancel</button>
              <button class="wf-btn" id="wfConfirmSaveAs">Save As New</button>
              <button class="wf-btn primary" id="wfConfirmReplace">Replace Without Saving</button>
            </div>
          </div>
        </div>
      `;
      document.body.appendChild(root);

      this.overlay = document.getElementById("wfOverlay");
      this.listEl = document.getElementById("wfList");
      this.detailEl = document.getElementById("wfDetail");
      this.searchEl = document.getElementById("wfSearch");
      this.tabTemplates = document.getElementById("wfTabTemplates");
      this.tabSaved = document.getElementById("wfTabSaved");
      this.confirmOverlay = document.getElementById("wfConfirm");
    }

    _bindUi() {
      document
        .getElementById("wfCloseBtn")
        ?.addEventListener("click", () => this.close());
      document
        .getElementById("wfRefreshBtn")
        ?.addEventListener("click", async () => {
          this._templatesLoaded = false;
          await this._loadTemplates();
          await this._loadSavedIndex();
          this.render();
        });
      document
        .getElementById("wfSaveAsBtn")
        ?.addEventListener("click", async () => {
          await this.promptSaveAs();
        });

      this.overlay?.addEventListener("click", (e) => {
        if (e.target === this.overlay) this.close();
      });

      this.searchEl?.addEventListener("input", () => {
        this._search = (this.searchEl.value || "").trim().toLowerCase();
        this.render();
      });

      this.tabTemplates?.addEventListener("click", () => {
        this._activeTab = "templates";
        this._selected = null;
        this._setTabClasses();
        this.render();
      });

      this.tabSaved?.addEventListener("click", () => {
        this._activeTab = "saved";
        this._selected = null;
        this._setTabClasses();
        this.render();
      });

      document.addEventListener("keydown", (e) => {
        if (!this.isOpen()) return;
        if (e.key === "Escape") {
          e.preventDefault();
          if (this.confirmOverlay?.classList.contains("open")) {
            this._resolveConfirm("cancel");
          } else {
            this.close();
          }
        }
      });
    }

    _setTabClasses() {
      this.tabTemplates?.classList.toggle(
        "active",
        this._activeTab === "templates",
      );
      this.tabSaved?.classList.toggle("active", this._activeTab === "saved");
    }

    async _loadTemplates() {
      if (this._templatesLoaded) return this._templates;
      try {
        const res = await fetch(this.templatesManifestUrl, {
          cache: "no-cache",
        });
        if (!res.ok)
          throw new Error(`Template manifest request failed: ${res.status}`);
        const json = await res.json();
        const templates = Array.isArray(json) ? json : json.templates;
        this._templates = Array.isArray(templates) ? templates : [];
      } catch (err) {
        console.warn("Failed to load workflow templates manifest", err);
        this._templates = [];
      } finally {
        this._templatesLoaded = true;
      }
      return this._templates;
    }

    async _requestJson(url, options = {}) {
      const response = await fetch(url, options);
      let payload = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }
      if (!response.ok) {
        const msg =
          payload?.error || `${response.status} ${response.statusText}`;
        throw new Error(msg);
      }
      return payload;
    }

    async _loadSavedIndex() {
      try {
        const payload = await this._requestJson("/workflows/user", {
          cache: "no-cache",
        });
        this._savedIndex = Array.isArray(payload?.items) ? payload.items : [];
        this._savedDirectory = payload?.directory || "";
      } catch (err) {
        console.warn("Failed to load saved workflows", err);
        this._savedIndex = [];
      }
      return this._savedIndex;
    }

    _getSavedIndex() {
      if (!Array.isArray(this._savedIndex)) return [];
      return [...this._savedIndex].sort((a, b) =>
        (a.updatedAt || "") < (b.updatedAt || "") ? 1 : -1,
      );
    }

    async _getSavedRecord(id) {
      if (!id) return null;
      try {
        return await this._requestJson(
          `/workflows/user/${encodeURIComponent(String(id))}`,
          {
            cache: "no-cache",
          },
        );
      } catch (err) {
        console.warn("Failed to fetch saved workflow", err);
        return null;
      }
    }

    isOpen() {
      return this.overlay?.classList.contains("open");
    }

    async open() {
      await this._loadTemplates();
      await this._loadSavedIndex();
      this.overlay?.classList.add("open");
      this.overlay?.setAttribute("aria-hidden", "false");
      this._setTabClasses();
      this.render();
      this.searchEl?.focus();
    }

    close() {
      this.overlay?.classList.remove("open");
      this.overlay?.setAttribute("aria-hidden", "true");
    }

    _activeCollection() {
      if (this._activeTab === "templates") {
        return this._templates.filter((t) =>
          this._matchesSearch(t.name, t.description, t.category, t.tags),
        );
      }
      const saved = this._getSavedIndex();
      return saved.filter((s) =>
        this._matchesSearch(s.name, s.description, s.tags),
      );
    }

    _matchesSearch(...parts) {
      if (!this._search) return true;
      const hay = parts
        .flatMap((p) => (Array.isArray(p) ? p : [p]))
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return hay.includes(this._search);
    }

    render() {
      if (!this.listEl || !this.detailEl) return;
      const items = this._activeCollection();
      if (!items.length) {
        this.listEl.innerHTML = `
          <div class="wf-empty">
            ${
              this._activeTab === "templates"
                ? "No templates found. Add entries to /static/workflows/index.json and files under /static/workflows/templates/."
                : `No saved workflows yet. Use Save As to create one from your current graph.<br><br>Saved files folder:<br>${this._esc(this._savedDirectory || "/frontend/workflows/user_workflow")}`
            }
          </div>
        `;
        this.detailEl.innerHTML = `<div class="wf-muted">Select an item to view details.</div>`;
        return;
      }

      if (this._activeTab === "templates") {
        this._renderTemplates(items);
      } else {
        this._renderSaved(items);
      }
    }

    _renderTemplates(items) {
      this.listEl.innerHTML = `<div class="wf-grid">${items
        .map((entry) => {
          const selectedClass =
            this._selected?.kind === "template" &&
            this._selected.id === entry.id
              ? "selected"
              : "";
          return `
            <article class="wf-card ${selectedClass}" data-kind="template" data-id="${entry.id || ""}">
              <h4 class="wf-title">${this._esc(entry.name || "Unnamed Template")}</h4>
              <p class="wf-desc">${this._esc(entry.description || "No description")}</p>
              <div class="wf-tags">
                ${entry.category ? `<span class="wf-tag">${this._esc(entry.category)}</span>` : ""}
                ${(entry.tags || [])
                  .slice(0, 5)
                  .map((tag) => `<span class="wf-tag">${this._esc(tag)}</span>`)
                  .join("")}
              </div>
            </article>
          `;
        })
        .join("")}</div>`;

      this.listEl.querySelectorAll(".wf-card").forEach((card) => {
        card.addEventListener("click", () => {
          const id = card.getAttribute("data-id");
          const item = items.find((it) => String(it.id) === String(id));
          if (!item) return;
          this._selected = { kind: "template", id: item.id || id };
          this._renderTemplateDetail(item);
          this._renderTemplates(items);
        });
        card.addEventListener("dblclick", async () => {
          const id = card.getAttribute("data-id");
          const item = items.find((it) => String(it.id) === String(id));
          if (!item) return;
          await this._loadTemplate(item);
        });
      });

      if (this._selected?.kind === "template") {
        const selected = items.find(
          (it) => String(it.id) === String(this._selected.id),
        );
        if (selected) {
          this._renderTemplateDetail(selected);
        } else {
          this.detailEl.innerHTML = `<div class="wf-muted">Select a template to preview details.</div>`;
        }
      } else {
        this.detailEl.innerHTML = `<div class="wf-muted">Select a template to preview details.</div>`;
      }
    }

    _renderTemplateDetail(item) {
      this.detailEl.innerHTML = `
        <h4>${this._esc(item.name || "Unnamed Template")}</h4>
        <div class="wf-muted" style="margin-bottom:8px">${this._esc(item.category || "Uncategorized")}</div>
        <p style="margin-top:0">${this._esc(item.description || "No description provided.")}</p>
        <div class="wf-tags" style="margin-bottom:12px">
          ${(item.tags || []).map((t) => `<span class="wf-tag">${this._esc(t)}</span>`).join("")}
        </div>
        <button class="wf-btn primary" id="wfLoadTemplateBtn">Load Template</button>
      `;

      document
        .getElementById("wfLoadTemplateBtn")
        ?.addEventListener("click", async () => {
          await this._loadTemplate(item);
        });
    }

    _renderSaved(items) {
      this.listEl.innerHTML = `<div class="wf-list">${items
        .map((entry) => {
          const selected =
            this._selected?.kind === "saved" && this._selected.id === entry.id;
          return `
            <div class="wf-row ${selected ? "selected" : ""}" data-kind="saved" data-id="${entry.id}">
              <div>
                <h4 class="wf-title">${this._esc(entry.name || "Unnamed Workflow")}</h4>
                <div class="wf-muted">Updated ${this._esc(dateLabel(entry.updatedAt))}</div>
                ${entry.description ? `<p class="wf-desc">${this._esc(entry.description)}</p>` : ""}
              </div>
              <div class="wf-actions">
                <button class="wf-btn" data-action="load" data-id="${entry.id}">Load</button>
                <button class="wf-btn" data-action="overwrite" data-id="${entry.id}">Overwrite</button>
                <button class="wf-btn" data-action="rename" data-id="${entry.id}">Rename</button>
                <button class="wf-btn" data-action="export" data-id="${entry.id}">Export</button>
                <button class="wf-btn" data-action="delete" data-id="${entry.id}">Delete</button>
              </div>
            </div>
          `;
        })
        .join("")}</div>`;

      this.listEl.querySelectorAll(".wf-row").forEach((row) => {
        row.addEventListener("click", (e) => {
          if (e.target instanceof HTMLElement && e.target.closest("button"))
            return;
          const id = row.getAttribute("data-id");
          this._selected = { kind: "saved", id };
          const selected = items.find((x) => String(x.id) === String(id));
          this._renderSavedDetail(selected);
          this._renderSaved(items);
        });
      });

      this.listEl.querySelectorAll("button[data-action]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const id = btn.getAttribute("data-id");
          const action = btn.getAttribute("data-action");
          if (!id || !action) return;
          if (action === "load") return this._loadSavedById(id);
          if (action === "overwrite") return this._overwriteSavedById(id);
          if (action === "rename") return this._renameSavedById(id);
          if (action === "export") return this._exportSavedById(id);
          if (action === "delete") return this._deleteSavedById(id);
        });
      });

      const selected =
        this._selected?.kind === "saved"
          ? items.find((x) => String(x.id) === String(this._selected.id))
          : null;
      if (selected) {
        this._renderSavedDetail(selected);
      } else {
        this.detailEl.innerHTML = `
          <h4>My Workflows</h4>
          <p class="wf-muted">Select one of your saved workflows to inspect details and load it.</p>
          <div style="margin-top:10px">
            <button class="wf-btn primary" id="wfSaveAsSide">Save Current As...</button>
          </div>
        `;
        document
          .getElementById("wfSaveAsSide")
          ?.addEventListener("click", async () => {
            await this.promptSaveAs();
            this.render();
          });
      }
    }

    _renderSavedDetail(entry) {
      if (!entry) {
        this.detailEl.innerHTML = `<div class="wf-muted">Select an item to view details.</div>`;
        return;
      }
      this.detailEl.innerHTML = `
        <h4>${this._esc(entry.name || "Unnamed Workflow")}</h4>
        <div class="wf-muted">Created ${this._esc(dateLabel(entry.createdAt))}</div>
        <div class="wf-muted" style="margin-bottom:8px">Updated ${this._esc(dateLabel(entry.updatedAt))}</div>
        <p style="margin-top:0">${this._esc(entry.description || "No description")}</p>
        <div class="wf-actions" style="justify-content:flex-start">
          <button class="wf-btn primary" id="wfLoadSavedBtn">Load</button>
          <button class="wf-btn" id="wfOverwriteSavedBtn">Overwrite</button>
          <button class="wf-btn" id="wfRenameSavedBtn">Rename</button>
          <button class="wf-btn" id="wfExportSavedBtn">Export</button>
          <button class="wf-btn" id="wfDeleteSavedBtn">Delete</button>
        </div>
      `;
      document
        .getElementById("wfLoadSavedBtn")
        ?.addEventListener("click", async () => this._loadSavedById(entry.id));
      document
        .getElementById("wfOverwriteSavedBtn")
        ?.addEventListener("click", async () =>
          this._overwriteSavedById(entry.id),
        );
      document
        .getElementById("wfRenameSavedBtn")
        ?.addEventListener("click", async () =>
          this._renameSavedById(entry.id),
        );
      document
        .getElementById("wfExportSavedBtn")
        ?.addEventListener("click", async () =>
          this._exportSavedById(entry.id),
        );
      document
        .getElementById("wfDeleteSavedBtn")
        ?.addEventListener("click", async () =>
          this._deleteSavedById(entry.id),
        );
    }

    _esc(val) {
      return String(val ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    _captureCurrentGraphRecord() {
      const graphJson = this.graph.serialize();
      const view = {
        scale: this.canvas.ds?.scale ?? 1,
        offset: this.canvas.ds?.offset ?? [0, 0],
      };
      return { graph: graphJson, view };
    }

    async _confirmReplaceIfDirty() {
      const current = JSON.stringify(this.graph.serialize());
      const autosaved = localStorage.getItem(this.autosaveGraphKey);
      const dirty = autosaved && autosaved !== current;
      if (!dirty) return "replace";
      return this._openConfirm();
    }

    _openConfirm() {
      return new Promise((resolve) => {
        this._confirmResolve = resolve;
        this.confirmOverlay?.classList.add("open");
        this.confirmOverlay?.setAttribute("aria-hidden", "false");

        const onCancel = () => this._resolveConfirm("cancel");
        const onSaveAs = () => this._resolveConfirm("saveAs");
        const onReplace = () => this._resolveConfirm("replace");

        document
          .getElementById("wfConfirmCancel")
          ?.addEventListener("click", onCancel, { once: true });
        document
          .getElementById("wfConfirmSaveAs")
          ?.addEventListener("click", onSaveAs, { once: true });
        document
          .getElementById("wfConfirmReplace")
          ?.addEventListener("click", onReplace, { once: true });
      });
    }

    _resolveConfirm(choice) {
      this.confirmOverlay?.classList.remove("open");
      this.confirmOverlay?.setAttribute("aria-hidden", "true");
      if (this._confirmResolve) {
        const r = this._confirmResolve;
        this._confirmResolve = null;
        r(choice);
      }
    }

    async _loadGraphPayload(payload, label) {
      const choice = await this._confirmReplaceIfDirty();
      if (choice === "cancel") return false;
      if (choice === "saveAs") {
        await this.promptSaveAs();
      }

      try {
        this.graph.stop();
        this.graph.clear();
        this.graph.configure(payload.graph);
        if (window.syncGraphWidgetsFromProperties) {
          window.syncGraphWidgetsFromProperties(this.graph);
        }
        if (payload.view && this.canvas.ds) {
          this.canvas.ds.scale = payload.view.scale ?? 1;
          this.canvas.ds.offset = payload.view.offset ?? [0, 0];
        }
        this.saveWorkingGraph();
        this.setStatus(`loaded ${label}`);
        return true;
      } catch (err) {
        console.error("Workflow load failed", err);
        this.setStatus("workflow load error");
        return false;
      }
    }

    async _loadTemplate(template) {
      if (!template?.file) {
        this.setStatus("template missing file");
        return;
      }

      const resolveCandidateUrls = (templateFile) => {
        const raw = String(templateFile || "").trim();
        const urls = [];
        const add = (u) => {
          if (!u) return;
          if (!urls.includes(u)) urls.push(u);
        };

        const toAbsolute = (value, fallbackBase) => {
          try {
            return new URL(
              value,
              fallbackBase || window.location.href,
            ).toString();
          } catch {
            return null;
          }
        };

        const absManifestUrl = toAbsolute(this.templatesManifestUrl);
        const absTemplatesBaseUrl = toAbsolute(this.templatesBaseUrl);

        // 1) Absolute URL/path as-is
        if (/^https?:\/\//i.test(raw) || raw.startsWith("/")) {
          add(toAbsolute(raw));
        }

        // 2) Relative to configured templates base
        if (absTemplatesBaseUrl) {
          try {
            add(new URL(raw, absTemplatesBaseUrl).toString());
          } catch {}
        }

        // 3) If file is already prefixed with templates/ but base also ends with /templates/
        // try with prefix stripped to avoid /templates/templates/... issues.
        if (
          raw.startsWith("templates/") &&
          /\/templates\/?$/.test(String(absTemplatesBaseUrl || ""))
        ) {
          const stripped = raw.replace(/^templates\//, "");
          if (absTemplatesBaseUrl) {
            try {
              add(new URL(stripped, absTemplatesBaseUrl).toString());
            } catch {}
          }
        }

        // 4) Relative to manifest directory (usually /static/workflows/)
        if (absManifestUrl) {
          try {
            const manifestBase = new URL(".", absManifestUrl).toString();
            add(new URL(raw, manifestBase).toString());
          } catch {}
        }

        // 5) Relative to current page as final fallback
        add(toAbsolute(raw));

        return urls;
      };

      try {
        const candidates = resolveCandidateUrls(template.file);
        if (!candidates.length) {
          throw new Error("No template URL candidates could be resolved");
        }

        let graphJson = null;
        let lastError = null;
        let lastTriedUrl = null;
        for (const url of candidates) {
          try {
            lastTriedUrl = url;
            const res = await fetch(url, { cache: "no-cache" });
            if (!res.ok) {
              lastError = new Error(
                `template fetch failed (${res.status}) for ${url}`,
              );
              continue;
            }
            graphJson = await res.json();
            break;
          } catch (err) {
            lastError = err;
          }
        }

        if (!graphJson) {
          const detail = candidates.join(" | ");
          throw (
            lastError ||
            new Error(
              `Unable to fetch template JSON. tried=${detail || lastTriedUrl || "none"}`,
            )
          );
        }

        const wrappedGraph =
          graphJson &&
          typeof graphJson === "object" &&
          graphJson.graph &&
          typeof graphJson.graph === "object" &&
          Array.isArray(graphJson.graph.nodes);

        const payload = wrappedGraph
          ? {
              graph: graphJson.graph,
              view:
                graphJson.view && typeof graphJson.view === "object"
                  ? graphJson.view
                  : undefined,
            }
          : { graph: graphJson };

        await this._loadGraphPayload(
          payload,
          `template: ${template.name || template.file}`,
        );
      } catch (err) {
        console.error("Failed to load template", err);
        const msg = String(err?.message || err || "unknown error");
        this.setStatus(`template load error: ${msg.slice(0, 220)}`);
      }
    }

    async _loadSavedById(id) {
      const record = await this._getSavedRecord(id);
      if (!record) {
        this.setStatus("saved workflow not found");
        return;
      }
      await this._loadGraphPayload(
        record,
        `workflow: ${record.name || "unnamed"}`,
      );
    }

    async _overwriteSavedById(id) {
      const record = await this._getSavedRecord(id);
      if (!record) return;
      if (!confirm(`Overwrite "${record.name}" with your current graph?`))
        return;
      const snap = this._captureCurrentGraphRecord();
      try {
        await this._requestJson("/workflows/user/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id,
            name: record.name,
            description: record.description || "",
            tags: Array.isArray(record.tags) ? record.tags : [],
            graph: snap.graph,
            view: snap.view,
            overwrite: true,
          }),
        });
        await this._loadSavedIndex();
        this.setStatus(`overwrote workflow: ${record.name}`);
        this.render();
      } catch (err) {
        console.error("Overwrite failed", err);
        this.setStatus("overwrite failed");
      }
    }

    async _renameSavedById(id) {
      const record = await this._getSavedRecord(id);
      if (!record) return;
      const name = prompt("New workflow name:", record.name || "");
      if (!name) return;
      const trimmed = name.trim();
      if (!trimmed) return;

      const desc =
        prompt("Description (optional):", record.description || "") ??
        record.description;

      try {
        await this._requestJson(
          `/workflows/user/${encodeURIComponent(String(id))}/rename`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: trimmed,
              description: desc,
            }),
          },
        );
        await this._loadSavedIndex();
        this.setStatus(`renamed workflow: ${trimmed}`);
        this.render();
      } catch (err) {
        console.error("Rename failed", err);
        this.setStatus("rename failed");
      }
    }

    async _deleteSavedById(id) {
      const record = await this._getSavedRecord(id);
      if (!record) return;
      if (!confirm(`Delete workflow "${record.name}"? This cannot be undone.`))
        return;
      try {
        await this._requestJson(
          `/workflows/user/${encodeURIComponent(String(id))}`,
          {
            method: "DELETE",
          },
        );
        await this._loadSavedIndex();
        this.setStatus(`deleted workflow: ${record.name}`);
        this._selected = null;
        this.render();
      } catch (err) {
        console.error("Delete failed", err);
        this.setStatus("delete failed");
      }
    }

    async _exportSavedById(id) {
      const record = await this._getSavedRecord(id);
      if (!record) return;
      const blob = new Blob([JSON.stringify(record.graph, null, 2)], {
        type: "application/json",
      });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${(record.name || "workflow").replace(/\s+/g, "-").toLowerCase()}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
      this.setStatus(`exported workflow: ${record.name}`);
    }

    _nameExists(name) {
      const n = (name || "").trim().toLowerCase();
      return this._getSavedIndex().some(
        (it) => (it.name || "").trim().toLowerCase() === n,
      );
    }

    async promptSaveAs() {
      const nameRaw = prompt("Workflow name:", "");
      if (!nameRaw) return false;
      const name = nameRaw.trim();
      if (!name) return false;

      const description = prompt("Description (optional):", "") || "";

      if (this._nameExists(name)) {
        const overwrite = confirm(
          `A workflow named "${name}" already exists. Overwrite it?`,
        );
        if (!overwrite) return false;
        const existing = this._getSavedIndex().find(
          (it) => (it.name || "").trim().toLowerCase() === name.toLowerCase(),
        );
        if (existing) {
          const snap = this._captureCurrentGraphRecord();
          try {
            await this._requestJson("/workflows/user/save", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                id: existing.id,
                name,
                description,
                tags: Array.isArray(existing.tags) ? existing.tags : [],
                graph: snap.graph,
                view: snap.view,
                overwrite: true,
              }),
            });
            await this._loadSavedIndex();
            this.setStatus(`overwrote workflow: ${name}`);
            this._activeTab = "saved";
            this._selected = { kind: "saved", id: existing.id };
            this._setTabClasses();
            this.render();
            return true;
          } catch (err) {
            console.error("Save failed", err);
            this.setStatus("save failed");
            return false;
          }
        }
      }

      const snap = this._captureCurrentGraphRecord();
      try {
        const result = await this._requestJson("/workflows/user/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id: uid(),
            name,
            description,
            tags: [],
            graph: snap.graph,
            view: snap.view,
            overwrite: false,
          }),
        });

        await this._loadSavedIndex();
        this.setStatus(`saved workflow: ${name}`);
        this._activeTab = "saved";
        this._selected = { kind: "saved", id: result?.id || null };
        this._setTabClasses();
        this.render();
        return true;
      } catch (err) {
        console.error("Save failed", err);
        this.setStatus("save failed");
        return false;
      }
    }
  }

  window.WorkflowBrowser = WorkflowBrowser;
})();
