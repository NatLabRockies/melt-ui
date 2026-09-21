class RegressionDataNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Make Regression Data"); // sets this.title

    // Outputs: x array, y array, shape object
    this.addOutput("x", "array");
    this.addOutput("y", "array");
    this.addOutput("shape", "object");

    // Properties with sensible defaults
    this.addProperty("n_samples", 1000, "number");
    this.addProperty("n_features", 10, "number");
    this.addProperty("n_informative", 5, "number");
    this.addProperty("n_targets", 3, "number");
    this.addProperty("noise", 1.0, "number");
    this.addProperty("random_state", 42, "number");
    this.addProperty("endpoint", "/regression_data", "string");

    // Add widgets for properties (same as before)
    this.addIntPropertyWidget("N Samples", "n_samples", {
      min: 0,
      default: 1000,
    });
    this.addIntPropertyWidget("N Features", "n_features", {
      min: 0,
      default: 10,
    });
    this.addIntPropertyWidget("N Informative", "n_informative", {
      min: 0,
      default: 5,
    });
    this.addIntPropertyWidget("N Targets", "n_targets", {
      min: 0,
      default: 3,
    });
    this.addFloatPropertyWidget("Noise", "noise", {
      min: 0,
      default: 1.0,
      step: 1,
      precision: 2,
    });
    this.addIntPropertyWidget("Seed", "random_state", {
      min: 0,
      default: 42,
    });

    this.size = [200, 225];
  }

  // The actual fetch — base class + NodeRunner will call this.
  async fetch() {
    const endpoint = this.properties && this.properties.endpoint;
    if (!endpoint) {
      // Let NodeRunner cache this error object
      return {
        x: [],
        y: [],
        shape: { x: [], y: [] },
        error: "No endpoint configured",
      };
    }

    const payload = {
      n_samples: parseInt(this.properties.n_samples),
      n_features: Number(this.properties.n_features),
      n_informative: Number(this.properties.n_informative),
      n_targets: Number(this.properties.n_targets),
      noise: Number(this.properties.noise),
      random_state: parseInt(this.properties.random_state),
    };

    const response = await window.MeltApi.fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (err) {
      json = { __raw_text: text };
    }

    if (!response.ok) {
      const errMsg = `HTTP ${response.status} ${
        response.statusText
      } - ${JSON.stringify(json)}`;
      throw new Error(errMsg);
    }

    return json;
  }

  // Map full JSON -> each output
  get extractors() {
    return [
      (j) => (j && (j.x || j.X)) || [], // x
      (j) => (j && (j.y || j.Y)) || [], // y
      (j) => (j && j.shape) || { x: [], y: [] }, // shape
    ];
  }

  // Defaults when upstream resolves to empty / errors are handled separately
  get defaults() {
    return [[], [], { x: [], y: [] }];
  }

  // Optional: keep old _pX/_pY/_pShape names if something else uses them
  onExecute() {
    // Let the base class handle NodeRunner.runMulti + outputs
    super.onExecute();

    // After base onExecute, NodeRunner.runMulti will have set:
    //   this._p0, this._p1, this._p2  (promises for each output)
    this._pX = this._p0;
    this._pY = this._p1;
    this._pShape = this._p2;
  }
}

// Keep LiteGraph metadata consistent with your old node
RegressionDataNode.title = "Make Regression Data";
RegressionDataNode.desc = "Generate surrogate regression data";

LiteGraph.registerNodeType("MELT/Data/RegressionDataNode", RegressionDataNode);

class DataNormalizerNode extends AsyncNodeBase {
  constructor() {
    // primary output index = 0 ("scaled_data")
    super("Normalize Data", 0);

    // Inputs
    this.addInput("input_data", "array");

    // Outputs
    this.addOutput("scaled_data", "array");
    // if you later add a scaler object, just add another output here

    // Properties with sensible defaults
    this.addProperty("norm_typ", "standard", "string");
    this.addProperty("endpoint", "/normalize_data", "string");

    // Dropdown for normalization type
    this.addWidget(
      "combo",
      "Normalization Type",
      this.properties.norm_typ,
      (v) => (this.properties.norm_typ = v),
      {
        values: ["standard", "minmax", "robust", "power", "quantile", "none"],
      },
    );

    // Optional: node size
    this.size = [250, 75];
  }

  // This is what NodeRunner will call whenever it needs to (re)fetch
  async fetch(resolvedInput) {
    const endpoint = this.properties && this.properties.endpoint;
    if (!endpoint) {
      // Let NodeRunner cache this error object
      return { scaled_data: [], error: "No endpoint configured" };
    }

    // If there's no upstream data, just return empty
    // (NodeRunner will often short-circuit this before calling fetch,
    // but this keeps the contract explicit and safe.)
    if (resolvedInput === undefined || resolvedInput === null) {
      return { scaled_data: [] };
    }

    const payload = {
      input_data: resolvedInput,
      norm_typ: this.properties.norm_typ,
    };

    const response = await window.MeltApi.fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (err) {
      json = { __raw_text: text };
    }

    if (!response.ok) {
      const errMsg = `HTTP ${response.status} ${
        response.statusText
      } - ${JSON.stringify(json)}`;
      throw new Error(errMsg);
    }

    return json;
  }

  // Map the backend JSON -> the single output
  extractPrimary(json) {
    return (json && json.scaled_data) || [];
  }

  // Default when we have no data / short-circuit
  get defaultPrimary() {
    return [];
  }
}

// Keep LiteGraph metadata & registration
DataNormalizerNode.title = "Normalize Data";
DataNormalizerNode.desc = "Normalize data using various scaling techniques";

LiteGraph.registerNodeType("MELT/Data/DataNormalizerNode", DataNormalizerNode);

class ExcelXYDataNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Load Excel/CSV Data");

    // Inputs (optional)
    this.addInput("model_metadata", "object");

    // Outputs
    this.addOutput("x_data", "array");
    this.addOutput("y_data", "array");
    this.addOutput("shape", "object");
    this.addOutput("schema", "object");

    // Properties
    this.addProperty("inspect_endpoint", "/excel_inspect", "string");
    this.addProperty(
      "inspect_by_id_endpoint",
      "/excel_inspect_by_id",
      "string",
    );
    this.addProperty("build_endpoint", "/excel_build_xy", "string");
    this.addProperty("dataset_id", "", "string");
    this.addProperty("sheet_name", "", "string");
    this.addProperty("target_col", "", "string");
    this.addProperty("preview_rows", 50, "number");
    this.addProperty("x_cols", [], "array");
    this.addProperty("y_cols", [], "array");
    this.addProperty("selected_col", "", "string");
    this.addProperty("auto_select_from_metadata", true, "boolean");
    this.addProperty("metadata_fill_empty_only", true, "boolean");

    // Internal state from inspect
    this._file_id = null;
    this._columns = [];
    this._sheet_names = [];
    this._dtypes = {};
    this._preview = [];
    this._x_cols = [];
    this._y_cols = [];
    this._uiState = "idle";
    this._uiMessage = "Step 1: Upload a CSV or Excel file.";
    this._lastError = "";
    this._lastHint = "";

    // Interactive column selector scroll + hit-area state
    this._colScrollOffset = 0;
    this._maxColScroll = 0;
    this._isDraggingColScroll = false;
    this._dragStartY = 0;
    this._scrollStart = 0;
    this._colSelectorArea = null;
    this._colRowAreas = [];
    this._chipRemoveAreas = [];

    // Upload button
    this.addWidget("button", "1) Upload Excel/CSV", "", () =>
      this._openFileDialog(),
    );

    // Sheet selector (populated after inspect)
    this._sheetWidget = this.addWidget(
      "combo",
      "2) Worksheet",
      this.properties.sheet_name,
      async (v) => {
        this.properties.sheet_name = v;
        await this._inspectSheetById(v);
      },
      { values: [] },
    );

    this._colWidget = this.addWidget(
      "combo",
      "3) Choose Column",
      this.properties.selected_col,
      (v) => (this.properties.selected_col = v),
      { values: [] },
    );

    this.addWidget("button", "Add to Features (X)", "", () => this._addToX());
    this.addWidget("button", "Add to Targets (y)", "", () => this._addToY());

    this.addWidget("button", "Undo Last Feature", "", () =>
      this._removeLastX(),
    );
    this.addWidget("button", "Undo Last Target", "", () => this._removeLastY());

    this.addWidget("button", "Clear Features", "", () => this._clearX());
    this.addWidget("button", "Clear Targets", "", () => this._clearY());
    this.addBooleanPropertyWidget(
      "Auto-select from Metadata",
      "auto_select_from_metadata",
      { default: true },
    );
    this.addBooleanPropertyWidget(
      "Metadata Fill Empty Only",
      "metadata_fill_empty_only",
      { default: true },
    );
    this.addWidget("button", "Apply Metadata Now", "", () =>
      this._applyMetadataNow(),
    );

    this.size = [400, 700];
    this.configureWidgetLayout({
      top: this._getWidgetTopOffset(),
      gap: 8,
      bottom: 14,
      minHeight: 640,
      autoSize: true,
    });
  }

  _colSelectorViewportHeight() {
    return 160;
  }

  _colRowHeight() {
    return 22;
  }

  // Assign a column directly by name (used by the in-panel +X / +y buttons)
  _assignColToX(col) {
    if (!col) return;
    if (this._y_cols.includes(col)) {
      this._y_cols = this._y_cols.filter((k) => k !== col);
      this._setUiState(
        "selecting",
        `Moved '${col}' from Targets to Features.`,
        "A column can only be in one role at a time.",
      );
    }
    if (!this._x_cols.includes(col)) {
      this._x_cols.push(col);
    }
    this._syncColProps();
    this._syncReadyHint();
  }

  _assignColToY(col) {
    if (!col) return;
    if (this._x_cols.includes(col)) {
      this._x_cols = this._x_cols.filter((k) => k !== col);
      this._setUiState(
        "selecting",
        `Moved '${col}' from Features to Targets.`,
        "A column can only be in one role at a time.",
      );
    }
    if (!this._y_cols.includes(col)) {
      this._y_cols.push(col);
    }
    this._syncColProps();
    this._syncReadyHint();
  }

  _chipPanelHeaderHeight() {
    return 18;
  }

  _chipHeight() {
    return 20;
  }

  _chipGap() {
    return 3;
  }

  _chipPanelSectionGap() {
    return 10;
  }

  _measureChipPanelHeight(items) {
    const safeItems = Array.isArray(items) ? items : [];
    const count = Math.max(1, safeItems.length);
    return (
      this._chipPanelHeaderHeight() +
      4 +
      count * (this._chipHeight() + this._chipGap())
    );
  }

  _drawSingleChip(
    ctx,
    text,
    borderColor,
    fillColor,
    x,
    y,
    width,
    height,
    showRemove,
  ) {
    const r = 4;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.arcTo(x + width, y, x + width, y + r, r);
    ctx.lineTo(x + width, y + height - r);
    ctx.arcTo(x + width, y + height, x + width - r, y + height, r);
    ctx.lineTo(x + r, y + height);
    ctx.arcTo(x, y + height, x, y + height - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 0.8;
    ctx.stroke();

    const removeBtnW = showRemove ? 18 : 0;
    // Truncate text to fit pill width (approx 6.5 px per char at 11px Arial)
    const maxTextWidth = width - 14 - removeBtnW;
    const avgCharW = 6.5;
    const maxChars = Math.max(3, Math.floor(maxTextWidth / avgCharW));
    const displayText =
      text.length > maxChars ? `${text.slice(0, maxChars - 1)}\u2026` : text;

    ctx.font = "11px Arial";
    ctx.fillStyle = "#e0e0e0";
    ctx.fillText(displayText, x + 7, y + height - 5);

    if (!showRemove) return null;

    // Red × remove button on right edge of chip
    const xBtnX = x + width - removeBtnW;
    const xBtnY = y;
    const xBtnW = removeBtnW;
    const xBtnH = height;

    ctx.fillStyle = "rgba(170,45,45,0.85)";
    ctx.beginPath();
    ctx.moveTo(xBtnX, xBtnY);
    ctx.lineTo(xBtnX + xBtnW - r, xBtnY);
    ctx.arcTo(xBtnX + xBtnW, xBtnY, xBtnX + xBtnW, xBtnY + r, r);
    ctx.lineTo(xBtnX + xBtnW, xBtnY + xBtnH - r);
    ctx.arcTo(
      xBtnX + xBtnW,
      xBtnY + xBtnH,
      xBtnX + xBtnW - r,
      xBtnY + xBtnH,
      r,
    );
    ctx.lineTo(xBtnX, xBtnY + xBtnH);
    ctx.closePath();
    ctx.fill();

    ctx.font = "bold 11px Arial";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.fillText("\u00d7", xBtnX + xBtnW / 2, xBtnY + xBtnH - 5);
    ctx.textAlign = "left";

    return { x: xBtnX, y: xBtnY, w: xBtnW, h: xBtnH };
  }

  _drawChipPanel(ctx, label, items, accentColor, startY, nodeWidth, chipRole) {
    const safeItems = Array.isArray(items) ? items : [];
    const margin = 14;
    const chipWidth = nodeWidth - margin * 2;
    const chipH = this._chipHeight();
    const gap = this._chipGap();

    ctx.font = "bold 11px Arial";
    ctx.fillStyle = accentColor;
    ctx.fillText(label, margin, startY);

    let curY = startY + this._chipPanelHeaderHeight() + 2;

    if (!safeItems.length) {
      this._drawSingleChip(
        ctx,
        "(none selected)",
        "#555555",
        "#222222",
        margin,
        curY,
        chipWidth,
        chipH,
        false,
      );
      curY += chipH + gap;
    } else {
      const fillColor = accentColor === "#6ac36a" ? "#192b19" : "#2d2610";
      for (let i = 0; i < safeItems.length; i++) {
        const removeRect = this._drawSingleChip(
          ctx,
          String(safeItems[i]),
          accentColor,
          fillColor,
          margin,
          curY,
          chipWidth,
          chipH,
          true,
        );
        if (removeRect && chipRole) {
          this._chipRemoveAreas.push({
            role: chipRole,
            index: i,
            col: safeItems[i],
            rect: removeRect,
          });
        }
        curY += chipH + gap;
      }
    }

    return curY;
  }

  _drawColumnSelectorPanel(ctx, startY, nodeWidth) {
    const margin = 14;
    const panelW = nodeWidth - margin * 2;
    const viewH = this._colSelectorViewportHeight();
    const rowH = this._colRowHeight();
    const cols = this._columns || [];
    const btnW = 28;
    const btnGap = 3;

    // Store panel bounding rect for mouse hit testing — refreshed every draw
    this._colSelectorArea = { x: margin, y: startY, w: panelW, h: viewH };

    // Panel background + border
    ctx.fillStyle = "#1c1c1c";
    ctx.fillRect(margin, startY, panelW, viewH);
    ctx.strokeStyle = "#3a3a3a";
    ctx.lineWidth = 0.8;
    ctx.strokeRect(margin, startY, panelW, viewH);

    if (!cols.length) {
      ctx.font = "11px Arial";
      ctx.fillStyle = "#555";
      ctx.fillText(
        "Upload a file to see columns here.",
        margin + 8,
        startY + 22,
      );
      this._maxColScroll = 0;
      this._colRowAreas = [];
      return startY + viewH;
    }

    const totalH = cols.length * rowH;
    this._maxColScroll = Math.max(0, totalH - viewH);
    const offset = Math.min(this._colScrollOffset || 0, this._maxColScroll);

    // Scrollbar width — only when content overflows
    const sbW = this._maxColScroll > 0 ? 7 : 0;
    const contentW = panelW - sbW;
    // Button area starts from the right of usable content
    const btnAreaX = margin + contentW - (btnW * 2 + btnGap + 4);

    // Clip rows to viewport
    ctx.save();
    ctx.beginPath();
    ctx.rect(margin, startY, contentW, viewH);
    ctx.clip();

    this._colRowAreas = [];

    for (let i = 0; i < cols.length; i++) {
      const col = cols[i];
      const rowY = startY + i * rowH - offset;

      // Cull rows outside visible viewport
      if (rowY + rowH < startY || rowY > startY + viewH) continue;

      const role = this._columnRole(col);
      const dtype = this._formatType(
        this._dtypes ? this._dtypes[col] : undefined,
      );
      const isX = role === "X";
      const isY = role === "y";

      // Row background
      if (isX) {
        ctx.fillStyle = "rgba(106,195,106,0.10)";
        ctx.fillRect(margin, rowY, contentW, rowH);
      } else if (isY) {
        ctx.fillStyle = "rgba(242,189,64,0.10)";
        ctx.fillRect(margin, rowY, contentW, rowH);
      } else if (i % 2 === 0) {
        ctx.fillStyle = "rgba(255,255,255,0.025)";
        ctx.fillRect(margin, rowY, contentW, rowH);
      }

      // Role indicator dot
      const dotColor = isX ? "#6ac36a" : isY ? "#f2bd40" : "#3a3a3a";
      ctx.fillStyle = dotColor;
      ctx.beginPath();
      ctx.arc(margin + 9, rowY + rowH / 2, 4, 0, Math.PI * 2);
      ctx.fill();

      // Column name — clipped to available width before buttons
      const nameX = margin + 20;
      const maxNameW = btnAreaX - nameX - 6;
      const avgCW = 6.2;
      const maxNameChars = Math.max(3, Math.floor(maxNameW / avgCW));
      const displayName =
        col.length > maxNameChars
          ? `${col.slice(0, maxNameChars - 1)}\u2026`
          : col;

      ctx.font = "11px Arial";
      ctx.fillStyle = isX || isY ? "#d8d8d8" : "#888";
      ctx.fillText(displayName, nameX, rowY + rowH - 6);

      // dtype badge
      ctx.font = "10px Arial";
      ctx.fillStyle = "#4a4a4a";
      const nameW = ctx.measureText(displayName).width;
      ctx.fillText(`(${dtype})`, nameX + nameW + 4, rowY + rowH - 6);

      // [+X] button
      const addXRect = { x: btnAreaX, y: rowY + 2, w: btnW, h: rowH - 4 };
      ctx.fillStyle = isX ? "#1e3a1e" : "#222";
      ctx.fillRect(addXRect.x, addXRect.y, addXRect.w, addXRect.h);
      ctx.strokeStyle = isX ? "#6ac36a" : "#383838";
      ctx.lineWidth = 0.8;
      ctx.strokeRect(addXRect.x, addXRect.y, addXRect.w, addXRect.h);
      ctx.font = "bold 10px Arial";
      ctx.fillStyle = isX ? "#6ac36a" : "#555";
      ctx.textAlign = "center";
      ctx.fillText(
        "+X",
        addXRect.x + addXRect.w / 2,
        addXRect.y + addXRect.h - 4,
      );

      // [+y] button
      const addYRect = {
        x: btnAreaX + btnW + btnGap,
        y: rowY + 2,
        w: btnW,
        h: rowH - 4,
      };
      ctx.fillStyle = isY ? "#3a2e0a" : "#222";
      ctx.fillRect(addYRect.x, addYRect.y, addYRect.w, addYRect.h);
      ctx.strokeStyle = isY ? "#f2bd40" : "#383838";
      ctx.lineWidth = 0.8;
      ctx.strokeRect(addYRect.x, addYRect.y, addYRect.w, addYRect.h);
      ctx.fillStyle = isY ? "#f2bd40" : "#555";
      ctx.fillText(
        "+y",
        addYRect.x + addYRect.w / 2,
        addYRect.y + addYRect.h - 4,
      );
      ctx.textAlign = "left";

      this._colRowAreas.push({ col, rowY, rowH, addXRect, addYRect });
    }

    ctx.restore();

    // Scrollbar drawn after restore so it's on top of the clipped content
    if (this._maxColScroll > 0) {
      const sbX = margin + panelW - sbW;
      const thumbH = Math.max(16, (viewH / totalH) * viewH);
      const thumbRatio = offset / this._maxColScroll;
      const thumbY = startY + thumbRatio * (viewH - thumbH);
      ctx.fillStyle = "#2a2a2a";
      ctx.fillRect(sbX, startY, sbW, viewH);
      ctx.fillStyle = "#606060";
      ctx.fillRect(sbX + 1, thumbY, sbW - 2, thumbH);
    }

    return startY + viewH;
  }

  _getInspectorBottomY() {
    // Column selector panel starts at y=130, has fixed viewport height
    const selectorBottom = 130 + this._colSelectorViewportHeight();
    const selectionHeight =
      this._measureChipPanelHeight(this._x_cols) +
      this._chipPanelSectionGap() +
      this._measureChipPanelHeight(this._y_cols);
    return (
      selectorBottom +
      12 +
      selectionHeight +
      (this._uiState === "error" && this._lastError ? 28 : 10)
    );
  }

  _getWidgetTopOffset() {
    return this._getInspectorBottomY() + 16;
  }

  _relayoutWidgets() {
    this.setWidgetTopOffset(this._getWidgetTopOffset());
  }

  _setUiState(state, message = "", hint = "", error = "") {
    this._uiState = state;
    this._uiMessage = message;
    this._lastHint = hint;
    this._lastError = error;
    this._relayoutWidgets();
    this.setDirtyCanvas(true, true);
  }

  _syncColumnWidgetValues() {
    if (!this._colWidget) return;
    this._colWidget.options.values = this._columns;
    if (!this._columns.includes(this.properties.selected_col)) {
      this.properties.selected_col = this._columns[0] || "";
      this._colWidget.value = this.properties.selected_col;
    }
  }

  _syncReadyHint() {
    if (!this._file_id) {
      this._setUiState(
        "idle",
        "Step 1: Upload a CSV or Excel file.",
        "Accepted formats: .csv, .xlsx",
      );
      return;
    }

    if (!this._x_cols.length || !this._y_cols.length) {
      this._setUiState(
        "selecting",
        "Step 3: Pick at least one Feature and one Target column.",
        "Tip: Targets (y) can include multiple columns.",
      );
      return;
    }

    this._setUiState(
      "ready",
      "Ready: run the graph to build X/Y arrays.",
      `${this._x_cols.length} feature(s), ${this._y_cols.length} target(s) selected.`,
    );
  }

  _formatType(dtype) {
    if (!dtype) return "unknown";
    const text = String(dtype).toLowerCase();
    if (text.includes("float")) return "float";
    if (text.includes("int")) return "int";
    if (text.includes("bool")) return "bool";
    if (text.includes("date") || text.includes("time")) return "date/time";
    if (text.includes("object") || text.includes("string")) return "text";
    return text.length > 10 ? `${text.slice(0, 10)}...` : text;
  }

  _columnRole(colName) {
    if (this._x_cols.includes(colName)) return "X";
    if (this._y_cols.includes(colName)) return "y";
    return "-";
  }

  _addToX() {
    const c = this.properties.selected_col;
    if (!c) return;

    // prevent overlap
    if (this._y_cols.includes(c)) {
      // remove from y if present (or block)
      this._y_cols = this._y_cols.filter((k) => k !== c);
      this._setUiState(
        "selecting",
        `Moved '${c}' from Targets to Features.`,
        "A column can only be in one role at a time.",
      );
    }

    if (!this._x_cols.includes(c)) {
      this._x_cols.push(c);
      this._setUiState(
        "selecting",
        `Added '${c}' to Features (X).`,
        "Next: choose one or more Target columns (y).",
      );
    }

    this._syncColProps();
    this._syncReadyHint();
  }

  _addToY() {
    const c = this.properties.selected_col;
    if (!c) return;

    // prevent overlap
    if (this._x_cols.includes(c)) {
      this._x_cols = this._x_cols.filter((k) => k !== c);
      this._setUiState(
        "selecting",
        `Moved '${c}' from Features to Targets.`,
        "A column can only be in one role at a time.",
      );
    }

    // multi-target: append if not already present
    if (!this._y_cols.includes(c)) {
      this._y_cols.push(c);
      this._setUiState(
        "selecting",
        `Added '${c}' to Targets (y).`,
        "You can add multiple target columns.",
      );
    }

    this._syncColProps();
    this._syncReadyHint();
  }

  _removeLastX() {
    const removed = this._x_cols.pop();
    this._syncColProps();
    this._setUiState(
      "selecting",
      removed
        ? `Removed last Feature: '${removed}'.`
        : "No Feature columns to remove.",
      "Pick a column, then add it to Features (X).",
    );
    this._syncReadyHint();
  }

  _removeLastY() {
    const removed = this._y_cols.pop();
    this._syncColProps();
    this._setUiState(
      "selecting",
      removed
        ? `Removed last Target: '${removed}'.`
        : "No Target columns to remove.",
      "Targets (y) can include more than one column.",
    );
    this._syncReadyHint();
  }

  _clearX() {
    this._x_cols = [];
    this._syncColProps();
    this._setUiState(
      "selecting",
      "Cleared Feature columns.",
      "Choose one or more Feature columns (X).",
    );
    this._syncReadyHint();
  }

  _clearY() {
    this._y_cols = [];
    this._syncColProps();
    this._setUiState(
      "selecting",
      "Cleared Target columns.",
      "Choose at least one Target column (y).",
    );
    this._syncReadyHint();
  }

  _syncColProps() {
    // keep properties in sync so graph serialization works
    this.properties.x_cols = [...this._x_cols];
    this.properties.y_cols = [...this._y_cols];
  }

  _parseFeatureList(raw) {
    if (raw == null) return [];
    if (Array.isArray(raw)) {
      return [...new Set(raw.map((v) => String(v).trim()).filter(Boolean))];
    }

    const txt = String(raw).trim();
    if (!txt) return [];

    // Try strict JSON first.
    try {
      const parsed = JSON.parse(txt);
      if (Array.isArray(parsed)) {
        return [
          ...new Set(parsed.map((v) => String(v).trim()).filter(Boolean)),
        ];
      }
    } catch {}

    // Handle Python-style list strings: ['a', 'b']
    const quoted = [];
    const re = /['"]([^'"]+)['"]/g;
    let m;
    while ((m = re.exec(txt)) !== null) {
      quoted.push(m[1].trim());
    }
    if (quoted.length) return [...new Set(quoted.filter(Boolean))];

    // Fallback: comma-separated text
    return [
      ...new Set(
        txt
          .split(",")
          .map((v) => v.trim())
          .filter(Boolean),
      ),
    ];
  }

  _extractFeatureListsFromMetadata(metaInput) {
    if (!metaInput || typeof metaInput !== "object") {
      return { inputFeatures: [], outputFeatures: [] };
    }

    // Check if this is a schema object from another ExcelXYDataNode or CombineDatasetsNode
    // Schema has x_cols and y_cols directly
    if (Array.isArray(metaInput.x_cols) || Array.isArray(metaInput.y_cols)) {
      const inputFeatures = this._parseFeatureList(metaInput.x_cols);
      const outputFeatures = this._parseFeatureList(metaInput.y_cols);
      if (inputFeatures.length || outputFeatures.length) {
        return { inputFeatures, outputFeatures };
      }
    }

    // Supported payload shapes:
    // 1) Direct metadata map from LoadModel metadata output
    // 2) Nested model metadata payloads
    const candidates = [
      metaInput,
      metaInput.model_meta,
      metaInput.metadata,
      metaInput.model && metaInput.model.model_meta,
      metaInput.model && metaInput.model.metadata,
    ].filter((c) => c && typeof c === "object");

    for (const c of candidates) {
      const inputRaw = c.input_features;
      const outputRaw = c.output_features;
      const inputFeatures = this._parseFeatureList(inputRaw);
      const outputFeatures = this._parseFeatureList(outputRaw);
      if (inputFeatures.length || outputFeatures.length) {
        return { inputFeatures, outputFeatures };
      }
    }

    return { inputFeatures: [], outputFeatures: [] };
  }

  _applyMetadataAutoSelection(metaInput) {
    if (!this.properties.auto_select_from_metadata) {
      return { applied: false, reason: "disabled" };
    }
    if (!metaInput || typeof metaInput.then === "function") {
      return { applied: false, reason: "missing_or_pending" };
    }
    if (!Array.isArray(this._columns) || this._columns.length === 0) {
      return { applied: false, reason: "no_columns" };
    }

    const { inputFeatures, outputFeatures } =
      this._extractFeatureListsFromMetadata(metaInput);
    if (!inputFeatures.length && !outputFeatures.length) {
      return { applied: false, reason: "no_feature_lists" };
    }

    const colSet = new Set(this._columns);
    const matchedX = inputFeatures.filter((c) => colSet.has(c));
    const matchedY = outputFeatures.filter((c) => colSet.has(c));

    const missingX = inputFeatures.filter((c) => !colSet.has(c));
    const missingY = outputFeatures.filter((c) => !colSet.has(c));

    const fillEmptyOnly = !!this.properties.metadata_fill_empty_only;
    const canApplyX = !fillEmptyOnly || this._x_cols.length === 0;
    const canApplyY = !fillEmptyOnly || this._y_cols.length === 0;

    let nextX = this._x_cols.slice();
    let nextY = this._y_cols.slice();

    if (canApplyX && matchedX.length) {
      nextX = [...new Set(matchedX)];
    }
    if (canApplyY && matchedY.length) {
      nextY = [...new Set(matchedY.filter((c) => !nextX.includes(c)))];
    }

    // Ensure mutual exclusivity.
    nextX = nextX.filter((c) => !nextY.includes(c));

    const changed =
      JSON.stringify(nextX) !== JSON.stringify(this._x_cols) ||
      JSON.stringify(nextY) !== JSON.stringify(this._y_cols);

    if (changed) {
      this._x_cols = nextX;
      this._y_cols = nextY;
      this._syncColProps();

      const parts = [];
      parts.push(
        `Metadata auto-selected ${this._x_cols.length} X and ${this._y_cols.length} y column(s).`,
      );
      if (missingX.length || missingY.length) {
        const missParts = [];
        if (missingX.length)
          missParts.push(
            `missing X: ${missingX.slice(0, 5).join(", ")}${missingX.length > 5 ? ", ..." : ""}`,
          );
        if (missingY.length)
          missParts.push(
            `missing y: ${missingY.slice(0, 5).join(", ")}${missingY.length > 5 ? ", ..." : ""}`,
          );
        this._setUiState("selecting", parts.join(" "), missParts.join(" | "));
      } else {
        this._setUiState(
          "selecting",
          parts.join(" "),
          "Metadata-driven mapping applied in model feature order.",
        );
      }
      this._syncReadyHint();
    }

    return { applied: changed, missingX, missingY };
  }

  _applyMetadataNow() {
    const result = this._applyMetadataAutoSelection(this.getInputData(0));
    if (result.applied) {
      return;
    }

    const reason = result.reason || "no_change";
    if (reason === "disabled") {
      this._setUiState(
        "selecting",
        "Metadata auto-select is disabled.",
        "Enable 'Auto-select from Metadata' to apply metadata mapping.",
      );
      return;
    }

    if (reason === "missing_or_pending") {
      this._setUiState(
        "selecting",
        "No metadata payload available yet.",
        "Connect model metadata output to this node and run upstream nodes.",
      );
      return;
    }

    if (reason === "no_columns") {
      this._setUiState(
        "selecting",
        "No columns available yet.",
        "Upload or inspect a dataset first, then apply metadata.",
      );
      return;
    }

    if (reason === "no_feature_lists") {
      this._setUiState(
        "selecting",
        "Metadata missing feature lists.",
        "Expected input_features/output_features in connected metadata or x_cols/y_cols in schema.",
      );
      return;
    }

    this._setUiState(
      "selecting",
      "Metadata mapping produced no changes.",
      "Either selections are already aligned or fill-empty-only prevented overwrite.",
    );
  }

  _openFileDialog() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".xlsx,.csv";
    input.onchange = async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      this._setUiState("uploading", `Uploading '${file.name}'...`);
      await this._inspectExcel(file);
    };
    input.click();
  }

  async _inspectSheetById(sheetName) {
    const datasetId = this.properties.dataset_id || this._file_id;
    if (!datasetId) return;
    this._setUiState(
      "inspecting",
      `Reading worksheet '${sheetName || "default"}'...`,
    );

    const endpoint =
      this.properties.inspect_by_id_endpoint || "/excel_inspect_by_id";
    const payload = {
      dataset_id: datasetId,
      file_id: datasetId, // backward compatibility
      sheet_name: sheetName,
      preview_rows: this.properties.preview_rows,
    };

    const res = await window.MeltApi.fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await res.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch {
      json = { __raw_text: text };
    }

    if (!res.ok) {
      console.error(
        "excel_inspect_by_id failed:",
        res.status,
        res.statusText,
        text,
      );
      const msg = `Could not read worksheet (${res.status} ${res.statusText}).`;
      this._setUiState(
        "error",
        msg,
        "Try a different worksheet or re-upload the file.",
        msg,
      );
      return;
    }

    // Update columns/dtypes/preview
    this._columns = json.columns || [];
    this._dtypes = json.dtypes || {};
    this._preview = json.preview_rows || [];
    this._sheet_names = json.sheet_names || this._sheet_names;
    this._file_id = json.dataset_id || json.file_id || datasetId;
    this.properties.dataset_id = this._file_id;
    this._sheetWidget.options.values = this._sheet_names;
    this.properties.sheet_name = json.sheet_name || sheetName;
    this._sheetWidget.value = this.properties.sheet_name;

    // Update "Column" dropdown values
    this._syncColumnWidgetValues();

    // Reconcile existing X/y selections against new columns
    const colSet = new Set(this._columns);
    this._x_cols = (this._x_cols || []).filter((c) => colSet.has(c));
    this._y_cols = (this._y_cols || []).filter((c) => colSet.has(c));
    this._syncColProps();

    // If metadata is already connected and resolved, apply ordered auto-selection.
    this._applyMetadataAutoSelection(this.getInputData(0));

    this._setUiState(
      "selecting",
      `Worksheet '${this.properties.sheet_name || "default"}' loaded.`,
      `Found ${this._columns.length} column(s). Choose Features (X) and Targets (y).`,
    );
    this._syncReadyHint();
  }

  async _inspectExcel(file) {
    const endpoint = this.properties.inspect_endpoint;
    const fd = new FormData();
    fd.append("file", file, file.name);
    fd.append("preview_rows", String(this.properties.preview_rows));

    const res = await window.MeltApi.fetch(endpoint, { method: "POST", body: fd });
    const text = await res.text();

    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch {
      json = { __raw_text: text };
    }

    if (!res.ok) {
      const msg = json?.detail || JSON.stringify(json);
      console.error("excel_inspect failed:", msg);
      this._file_id = null;
      this.properties.dataset_id = "";
      this._columns = [];
      this._sheet_names = [];
      this._dtypes = {};
      this._preview = [];
      this._x_cols = [];
      this._y_cols = [];
      this._syncColProps();
      this._setUiState(
        "error",
        "Could not read that file.",
        "Check that the file is a valid .csv or .xlsx and try again.",
        String(msg),
      );
      return;
    }

    this._file_id = json.dataset_id || json.file_id;
    this.properties.dataset_id = this._file_id || "";
    this._columns = json.columns || [];
    this._sheet_names = json.sheet_names || [];
    this._dtypes = json.dtypes || {};
    this._preview = json.preview_rows || [];

    // Populate available columns dropdown
    this._syncColumnWidgetValues();

    // reset selections on new upload
    this._x_cols = [];
    this._y_cols = [];
    this._syncColProps();

    // If metadata is already connected and resolved, apply ordered auto-selection.
    this._applyMetadataAutoSelection(this.getInputData(0));

    // Populate sheet dropdown
    this._sheetWidget.options.values = this._sheet_names;
    this.properties.sheet_name = json.sheet_name || this._sheet_names[0] || "";
    this._sheetWidget.value = this.properties.sheet_name;

    this._setUiState(
      "selecting",
      "File loaded successfully.",
      `Step 2: choose worksheet. Step 3: choose Features (X) and Targets (y).`,
    );
    this._syncReadyHint();
  }

  // Called by NodeRunner on execute
  async fetch(resolvedInput) {
    const endpoint = this.properties.build_endpoint;

    // Apply metadata-driven selection on execute as well, ensuring mappings can
    // be refreshed when upstream metadata arrives after dataset load.
    const metaInput =
      resolvedInput && typeof resolvedInput === "object"
        ? resolvedInput
        : this.getInputData(0);
    this._applyMetadataAutoSelection(metaInput);
    if (!endpoint) {
      this._setUiState(
        "error",
        "Build endpoint is missing.",
        "Set build_endpoint before running this node.",
        "No build_endpoint configured",
      );
      return {
        x_data: [],
        y_data: [],
        shape: { x: [], y: [] },
        error: "No build_endpoint configured",
      };
    }
    const datasetId = this.properties.dataset_id || this._file_id;
    if (!datasetId) {
      this._setUiState(
        "selecting",
        "No file loaded yet.",
        "Use 'Upload Excel/CSV' first.",
      );
      return {
        x_data: [],
        y_data: [],
        shape: { x: [], y: [] },
        error: "No dataset uploaded yet",
      };
    }
    // if (!this.properties.target_col) {
    //   return {
    //     x_data: [],
    //     y_data: [],
    //     shape: { x: [], y: [] },
    //     error: "No target column selected",
    //   };
    // }

    // // v1: X = all columns except y
    // const y_cols = [this.properties.target_col];
    // const x_cols = (this._columns || []).filter(
    //   (c) => c !== this.properties.target_col
    // );
    const x_cols = this.properties.x_cols || [];
    const y_cols = this.properties.y_cols || [];

    if (!x_cols.length) {
      this._setUiState(
        "selecting",
        "Pick at least one Feature column.",
        "Select a column, then click 'Add to Features (X)'.",
      );
      return {
        x_data: [],
        y_data: [],
        shape: { x: [], y: [] },
        error: "No feature columns selected",
      };
    }
    if (!y_cols.length) {
      this._setUiState(
        "selecting",
        "Pick at least one Target column.",
        "Select a column, then click 'Add to Targets (y)'.",
      );
      return {
        x_data: [],
        y_data: [],
        shape: { x: [], y: [] },
        error: "No target column selected",
      };
    }

    this._setUiState(
      "building",
      "Building training arrays from your worksheet...",
      `${x_cols.length} feature(s), ${y_cols.length} target(s).`,
    );

    const payload = {
      dataset_id: datasetId,
      file_id: datasetId, // backward compatibility
      sheet_name: this.properties.sheet_name || null,
      x_cols,
      y_cols,
    };

    const res = await window.MeltApi.fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await res.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch {
      json = { __raw_text: text };
    }

    if (!res.ok) {
      const errText = `Build failed (${res.status} ${res.statusText}).`;
      this._setUiState(
        "error",
        errText,
        "Check your selected columns and try again.",
        JSON.stringify(json),
      );
      const errMsg = `HTTP ${res.status} ${res.statusText} - ${JSON.stringify(
        json,
      )}`;
      throw new Error(errMsg);
    }

    const xShape = (json && json.shape && json.shape.x) || [];
    const yShape = (json && json.shape && json.shape.y) || [];
    this._setUiState(
      "ready",
      "Build complete.",
      `X shape: [${xShape.join(", ")}], y shape: [${yShape.join(", ")}].`,
    );

    return json;
  }

  get extractors() {
    return [
      (j) => (j && j.x_data) || [],
      (j) => (j && j.y_data) || [],
      (j) => (j && j.shape) || { x: [], y: [] },
      (j) => ({
        x_cols: (j && j.x_cols) || [],
        y_cols: (j && j.y_cols) || [],
        dataset_id: (j && j.dataset_id) || this.properties.dataset_id || "",
        sheet_name: (j && j.sheet_name) || this.properties.sheet_name || "",
        shape: (j && j.shape) || { x: [], y: [] },
        stats: (j && j.stats) || {},
        warnings: (j && j.warnings) || [],
        file_type: (j && j.file_type) || "",
      }),
    ];
  }

  get defaults() {
    return [
      [],
      [],
      { x: [], y: [] },
      {
        x_cols: [],
        y_cols: [],
        dataset_id: "",
        sheet_name: "",
        shape: { x: [], y: [] },
        stats: {},
        warnings: [],
        file_type: "",
      },
    ];
  }

  onConfigure(info) {
    if (!info || !info.properties) return;

    const props = info.properties;
    if (props.dataset_id) {
      this.properties.dataset_id = props.dataset_id;
      this._file_id = props.dataset_id;
    }

    if (Array.isArray(props.x_cols)) this._x_cols = [...props.x_cols];
    if (Array.isArray(props.y_cols)) this._y_cols = [...props.y_cols];

    this._setUiState(
      "inspecting",
      "Restoring saved worksheet selection...",
      "Reading saved dataset metadata.",
    );

    const hasDataset = !!(this.properties.dataset_id || this._file_id);
    if (hasDataset && (!this._columns || this._columns.length === 0)) {
      const sheet = this.properties.sheet_name || "";
      this._inspectSheetById(sheet).catch((err) => {
        console.warn("Failed to restore dataset metadata:", err);
        this._setUiState(
          "error",
          "Could not restore saved worksheet metadata.",
          "Re-upload the file, then pick columns again.",
          String(err),
        );
      });
      return;
    }

    this._syncReadyHint();
  }

  // onDrawForeground(ctx) {
  //   // Optional: show quick status
  //   ctx.save();
  //   ctx.font = "12px Arial";
  //   ctx.fillStyle = "#AAA";
  //   const y = 18;

  //   const status = this._file_id ? `Cols: ${this._columns.length}` : "No file";
  //   ctx.fillText(status, 10, y);

  //   if (this.properties.target_col) {
  //     const xCount = Math.max(0, (this._columns?.length || 0) - 1);
  //     ctx.fillText(
  //       `X cols: ${xCount}  |  y: ${this.properties.target_col}`,
  //       10,
  //       y + 16
  //     );
  //   }
  //   ctx.restore();
  // }
  onDrawForeground(ctx) {
    ctx.save();
    ctx.font = "12px Arial";
    const colors = {
      idle: "#8f8f8f",
      uploading: "#3aa0ff",
      inspecting: "#3aa0ff",
      selecting: "#f2bd40",
      building: "#3aa0ff",
      ready: "#6ac36a",
      error: "#e06a6a",
    };

    const borderColor = colors[this._uiState] || "#8f8f8f";

    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(8, 8, this.size[0] - 16, this.size[1] - 16);

    ctx.fillStyle = borderColor;
    const stateName = (this._uiState || "idle").toUpperCase();
    ctx.fillText(`Status: ${stateName}`, 14, 24);

    ctx.fillStyle = "#d7d7d7";
    if (this._uiMessage) {
      ctx.fillText(this._uiMessage, 14, 42);
    }
    if (this._lastHint) {
      ctx.fillText(this._lastHint, 14, 58);
    }

    const datasetLabel = this._file_id ? "Loaded" : "Not loaded";
    const sheetLabel = this.properties.sheet_name || "(none)";
    ctx.fillStyle = "#b4b4b4";
    ctx.fillText(`Data Source: ${datasetLabel}`, 14, 78);
    ctx.fillText(`Worksheet: ${sheetLabel}`, 14, 94);
    ctx.fillText(
      `Selections: ${this._x_cols.length} Feature(s), ${this._y_cols.length} Target(s)`,
      14,
      110,
    );

    // Reset per-frame hit areas before drawing
    this._chipRemoveAreas = [];
    this._colRowAreas = [];

    ctx.font = "bold 11px Arial";
    ctx.fillStyle = "#777";
    ctx.fillText("Columns \u2014 click +X or +y to assign", 14, 126);

    const selectorBottom = this._drawColumnSelectorPanel(
      ctx,
      130,
      this.size[0],
    );

    const selectionStartY = selectorBottom + 10;
    let afterChipsY = this._drawChipPanel(
      ctx,
      "Features (X)",
      this._x_cols,
      "#6ac36a",
      selectionStartY,
      this.size[0],
      "x",
    );
    afterChipsY += this._chipPanelSectionGap();
    afterChipsY = this._drawChipPanel(
      ctx,
      "Targets (y)",
      this._y_cols,
      "#f2bd40",
      afterChipsY,
      this.size[0],
      "y",
    );

    if (this._uiState === "error" && this._lastError) {
      ctx.font = "11px Arial";
      ctx.fillStyle = "#e06a6a";
      ctx.fillText(`Error: ${this._lastError}`, 14, afterChipsY + 10);
    }

    ctx.restore();
  }

  onMouseDown(event, pos) {
    const px = pos[0];
    const py = pos[1];

    // 1. Chip × remove buttons (highest priority)
    for (const area of this._chipRemoveAreas || []) {
      const r = area.rect;
      if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h) {
        if (area.role === "x") {
          this._x_cols.splice(area.index, 1);
          this._setUiState("selecting", `Removed Feature '${area.col}'.`, "");
        } else {
          this._y_cols.splice(area.index, 1);
          this._setUiState("selecting", `Removed Target '${area.col}'.`, "");
        }
        this._syncColProps();
        this._syncReadyHint();
        return true;
      }
    }

    // 2. Column selector row [+X] / [+y] buttons
    for (const rowArea of this._colRowAreas || []) {
      const ax = rowArea.addXRect;
      const ay = rowArea.addYRect;
      if (px >= ax.x && px <= ax.x + ax.w && py >= ax.y && py <= ax.y + ax.h) {
        this._assignColToX(rowArea.col);
        return true;
      }
      if (px >= ay.x && px <= ay.x + ay.w && py >= ay.y && py <= ay.y + ay.h) {
        this._assignColToY(rowArea.col);
        return true;
      }
    }

    // 3. Start scroll drag if click is inside the column selector panel
    const panel = this._colSelectorArea;
    if (
      panel &&
      px >= panel.x &&
      px <= panel.x + panel.w &&
      py >= panel.y &&
      py <= panel.y + panel.h
    ) {
      this._isDraggingColScroll = true;
      this._dragStartY = py;
      this._scrollStart = this._colScrollOffset || 0;
      if (event.preventDefault) event.preventDefault();
      if (event.stopPropagation) event.stopPropagation();
      return true;
    }

    return false;
  }

  onMouseMove(event, pos) {
    if (!this._isDraggingColScroll) return false;
    const delta = pos[1] - this._dragStartY;
    let next = this._scrollStart + delta;
    if (next < 0) next = 0;
    if (next > (this._maxColScroll || 0)) next = this._maxColScroll || 0;
    this._colScrollOffset = next;
    this.setDirtyCanvas(true, true);
    return true;
  }

  onMouseUp(event, pos) {
    if (this._isDraggingColScroll) {
      this._isDraggingColScroll = false;
      return true;
    }
    return false;
  }

  onMouseWheel(event, pos) {
    const panel = this._colSelectorArea;
    if (!panel) return false;
    const px = pos[0];
    const py = pos[1];
    if (
      px < panel.x ||
      px > panel.x + panel.w ||
      py < panel.y ||
      py > panel.y + panel.h
    ) {
      return false;
    }
    const delta = event.deltaY || event.wheelDeltaY || event.wheelDelta || 0;
    let next = (this._colScrollOffset || 0) + (delta > 0 ? 30 : -30);
    if (next < 0) next = 0;
    if (next > (this._maxColScroll || 0)) next = this._maxColScroll || 0;
    this._colScrollOffset = next;
    this.setDirtyCanvas(true, true);
    return true;
  }
}

ExcelXYDataNode.title = "Load Excel/CSV Data";
ExcelXYDataNode.desc =
  "Upload CSV/XLSX data, select feature and target columns, or auto-select from connected model metadata input_features/output_features.";

LiteGraph.registerNodeType("MELT/Data/ExcelXYDataNode", ExcelXYDataNode);

class CombineDatasetsNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Combine Datasets");

    this.addProperty("dataset_count", 1, "number");
    this.addProperty("strict_schema", true, "boolean");
    this.addProperty("allow_missing_schema", true, "boolean");
    this.addProperty("drop_empty_inputs", true, "boolean");

    this._ensureDatasetCount(1, false);

    this.addOutput("x_data", "array");
    this.addOutput("y_data", "array");
    this.addOutput("shape", "object");
    this.addOutput("metadata", "object");

    this.addWidget("button", "Add Dataset", "", () =>
      this._setDatasetCount(this._currentDatasetCount() + 1),
    );
    this.addWidget("button", "Remove Dataset", "", () =>
      this._setDatasetCount(this._currentDatasetCount() - 1),
    );
    this.addBooleanPropertyWidget("Strict Schema", "strict_schema", {
      default: true,
    });
    this.addBooleanPropertyWidget(
      "Allow Missing Schema",
      "allow_missing_schema",
      { default: true },
    );
    this.addBooleanPropertyWidget("Drop Empty Inputs", "drop_empty_inputs", {
      default: true,
    });

    this.size = [320, 230];
    this._updateNodeSize();
  }

  _currentDatasetCount() {
    return Math.ceil(((this.inputs && this.inputs.length) || 0) / 3);
  }

  _datasetInputBase(index) {
    return (index - 1) * 3;
  }

  _ensureDatasetCount(count, invalidate = true) {
    const target = Math.max(1, parseInt(count) || 1);

    while (this._currentDatasetCount() < target) {
      const next = this._currentDatasetCount() + 1;
      this.addInput(`x_${next}`, "array");
      this.addInput(`y_${next}`, "array");
      this.addInput(`schema_${next}`, "object");
    }

    while (((this.inputs && this.inputs.length) || 0) > target * 3) {
      this._removeInputAt(this.inputs.length - 1);
    }

    this.properties.dataset_count = target;
    this._updateNodeSize();

    if (invalidate && this._runner) {
      this._runner.invalidate();
      this.setDirtyCanvas(true, true);
    }
  }

  _setDatasetCount(count) {
    this._ensureDatasetCount(count, true);
  }

  _removeInputAt(index) {
    if (!this.inputs || index < 0 || index >= this.inputs.length) return;

    try {
      if (typeof this.disconnectInput === "function") {
        this.disconnectInput(index);
      }
    } catch (err) {}

    if (typeof this.removeInput === "function") {
      this.removeInput(index);
      return;
    }

    this.inputs.splice(index, 1);
  }

  _updateNodeSize() {
    const count = Math.max(1, this._currentDatasetCount());
    if (!this.size) this.size = [320, 230];
    this.size[1] = Math.max(230, 170 + count * 26);
  }

  _isEmptyData(value) {
    if (value === undefined || value === null) return true;
    if (Array.isArray(value)) return value.length === 0;
    if (typeof value === "object" && value.error) return true;
    return false;
  }

  _normalizeMatrix(value, label, datasetIndex) {
    if (!Array.isArray(value) || value.length === 0) {
      return {
        error: `Dataset ${datasetIndex}: ${label} must be a non-empty array.`,
      };
    }

    const rows = value.map((row) => (Array.isArray(row) ? row.slice() : [row]));
    const width = rows[0].length;
    if (width === 0) {
      return {
        error: `Dataset ${datasetIndex}: ${label} rows must not be empty.`,
      };
    }

    for (let i = 0; i < rows.length; i++) {
      if (rows[i].length !== width) {
        return {
          error: `Dataset ${datasetIndex}: ${label} row ${i + 1} has ${rows[i].length} column(s), expected ${width}.`,
        };
      }
    }

    return { rows, width };
  }

  _orderedList(value) {
    return Array.isArray(value) ? value.map((item) => String(item)) : [];
  }

  _sameOrderedList(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return false;
    }
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  }

  _schemaSummary(schema) {
    const safeSchema = schema && typeof schema === "object" ? schema : {};
    return {
      x_cols: this._orderedList(safeSchema.x_cols),
      y_cols: this._orderedList(safeSchema.y_cols),
      dataset_id: safeSchema.dataset_id || "",
      sheet_name: safeSchema.sheet_name || "",
      stats: safeSchema.stats || {},
      warnings: Array.isArray(safeSchema.warnings) ? safeSchema.warnings : [],
      file_type: safeSchema.file_type || "",
      shape: safeSchema.shape || null,
    };
  }

  _emptyResult(error, warnings = []) {
    return {
      x_data: [],
      y_data: [],
      shape: { x: [], y: [], datasets: 0, source_row_ranges: [] },
      metadata: {
        input_features: [],
        output_features: [],
        source_datasets: [],
        source_row_ranges: [],
        warnings,
        error,
      },
      error,
    };
  }

  async fetch(resolvedInput) {
    const values = Array.isArray(resolvedInput) ? resolvedInput : [];
    const groupCount = this._currentDatasetCount();
    const warnings = [];
    const normalized = [];

    for (let datasetIndex = 1; datasetIndex <= groupCount; datasetIndex++) {
      const base = this._datasetInputBase(datasetIndex);
      const xRaw = values[base];
      const yRaw = values[base + 1];
      const schemaRaw = values[base + 2];

      const xEmpty = this._isEmptyData(xRaw);
      const yEmpty = this._isEmptyData(yRaw);
      if (xEmpty && yEmpty && this.properties.drop_empty_inputs) continue;

      if (xEmpty || yEmpty) {
        return this._emptyResult(
          `Dataset ${datasetIndex}: both x and y inputs are required.`,
          warnings,
        );
      }

      const x = this._normalizeMatrix(xRaw, "x", datasetIndex);
      if (x.error) return this._emptyResult(x.error, warnings);

      const y = this._normalizeMatrix(yRaw, "y", datasetIndex);
      if (y.error) return this._emptyResult(y.error, warnings);

      if (x.rows.length !== y.rows.length) {
        return this._emptyResult(
          `Dataset ${datasetIndex}: x has ${x.rows.length} row(s), but y has ${y.rows.length}.`,
          warnings,
        );
      }

      normalized.push({
        datasetIndex,
        x,
        y,
        schema: this._schemaSummary(schemaRaw),
      });
    }

    if (!normalized.length) {
      return this._emptyResult("No datasets connected.", warnings);
    }

    const reference = normalized[0];
    const referenceHasSchema =
      reference.schema.x_cols.length > 0 && reference.schema.y_cols.length > 0;
    const combinedX = [];
    const combinedY = [];
    const sourceRowRanges = [];
    const sourceDatasets = [];

    if (!referenceHasSchema) {
      const message =
        "Dataset 1: schema is missing; validating dimensions only.";
      if (!this.properties.allow_missing_schema) {
        return this._emptyResult(message, warnings);
      }
      warnings.push(message);
    }

    for (const item of normalized) {
      if (item.x.width !== reference.x.width) {
        return this._emptyResult(
          `Dataset ${item.datasetIndex}: x has ${item.x.width} feature column(s), expected ${reference.x.width}.`,
          warnings,
        );
      }
      if (item.y.width !== reference.y.width) {
        return this._emptyResult(
          `Dataset ${item.datasetIndex}: y has ${item.y.width} target column(s), expected ${reference.y.width}.`,
          warnings,
        );
      }

      const itemHasSchema =
        item.schema.x_cols.length > 0 && item.schema.y_cols.length > 0;
      if (!itemHasSchema) {
        const message = `Dataset ${item.datasetIndex}: schema is missing; validating dimensions only.`;
        if (!this.properties.allow_missing_schema) {
          return this._emptyResult(message, warnings);
        }
        if (item.datasetIndex !== reference.datasetIndex)
          warnings.push(message);
      } else if (this.properties.strict_schema && referenceHasSchema) {
        if (
          !this._sameOrderedList(reference.schema.x_cols, item.schema.x_cols)
        ) {
          return this._emptyResult(
            `Dataset ${item.datasetIndex}: feature column order does not match the first dataset.`,
            warnings,
          );
        }
        if (
          !this._sameOrderedList(reference.schema.y_cols, item.schema.y_cols)
        ) {
          return this._emptyResult(
            `Dataset ${item.datasetIndex}: target column order does not match the first dataset.`,
            warnings,
          );
        }
      }

      const start = combinedX.length;
      combinedX.push(...item.x.rows);
      combinedY.push(...item.y.rows);
      const end = combinedX.length;

      sourceRowRanges.push({
        dataset_index: item.datasetIndex,
        dataset_id: item.schema.dataset_id,
        start,
        end,
        rows: end - start,
      });
      sourceDatasets.push({
        dataset_index: item.datasetIndex,
        dataset_id: item.schema.dataset_id,
        sheet_name: item.schema.sheet_name,
        file_type: item.schema.file_type,
        rows: end - start,
        x_cols: item.schema.x_cols,
        y_cols: item.schema.y_cols,
        shape: item.schema.shape || {
          x: [item.x.rows.length, item.x.width],
          y: [item.y.rows.length, item.y.width],
        },
        stats: item.schema.stats,
        warnings: item.schema.warnings,
      });
      for (const warning of item.schema.warnings) {
        warnings.push(`Dataset ${item.datasetIndex}: ${warning}`);
      }
    }

    const inputFeatures = referenceHasSchema
      ? reference.schema.x_cols.slice()
      : Array.from({ length: reference.x.width }, (_, i) => `x_${i + 1}`);
    const outputFeatures = referenceHasSchema
      ? reference.schema.y_cols.slice()
      : Array.from({ length: reference.y.width }, (_, i) => `y_${i + 1}`);

    return {
      x_data: combinedX,
      y_data: combinedY,
      shape: {
        x: [combinedX.length, reference.x.width],
        y: [combinedY.length, reference.y.width],
        datasets: normalized.length,
        source_row_ranges: sourceRowRanges,
      },
      metadata: {
        input_features: inputFeatures,
        output_features: outputFeatures,
        source_datasets: sourceDatasets,
        source_row_ranges: sourceRowRanges,
        warnings,
        strict_schema: !!this.properties.strict_schema,
      },
    };
  }

  get extractors() {
    return [
      (j) => (j && j.x_data) || [],
      (j) => (j && j.y_data) || [],
      (j) => (j && j.shape) || { x: [], y: [], datasets: 0 },
      (j) => (j && j.metadata) || {},
    ];
  }

  get defaults() {
    return [[], [], { x: [], y: [], datasets: 0 }, {}];
  }

  onConfigure(info) {
    const propCount =
      info && info.properties ? parseInt(info.properties.dataset_count) : 0;
    const inputCount =
      info && Array.isArray(info.inputs) ? info.inputs.length : 0;
    const serializedCount = inputCount > 0 ? Math.ceil(inputCount / 3) : 0;
    this._ensureDatasetCount(
      Math.max(propCount || 0, serializedCount || 0, 1),
      false,
    );
    this.setDirtyCanvas(true, true);
  }

  onExecute() {
    this._ensureDatasetCount(this.properties.dataset_count, false);
    super.onExecute();
    this._pXData = this._p0;
    this._pYData = this._p1;
    this._pShape = this._p2;
    this._pMetadata = this._p3;
  }
}

CombineDatasetsNode.title = "Combine Datasets";
CombineDatasetsNode.desc =
  "Stack multiple x/y datasets row-wise after validating feature and target column order.";

LiteGraph.registerNodeType(
  "MELT/Data/CombineDatasetsNode",
  CombineDatasetsNode,
);

class TimeSeriesRegressionDataNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Make Time-Series Regression Data");

    this.addOutput("x", "array");
    this.addOutput("y", "array");
    this.addOutput("shape", "object");

    this.addProperty("n_samples", 2000, "number");
    this.addProperty("n_features", 4, "number");
    this.addProperty("season_period", 24, "number");
    this.addProperty("trend_strength", 0.001, "number");
    this.addProperty("noise", 0.05, "number");
    this.addProperty("random_state", 42, "number");
    this.addProperty("endpoint", "/time_series_regression_data", "string");

    this.addIntPropertyWidget("N Samples", "n_samples", {
      min: 10,
      default: 2000,
    });
    this.addIntPropertyWidget("N Features", "n_features", {
      min: 1,
      default: 4,
    });
    this.addIntPropertyWidget("Season Period", "season_period", {
      min: 2,
      default: 24,
    });
    this.addFloatPropertyWidget("Trend Strength", "trend_strength", {
      min: 0,
      default: 0.001,
      step: 0.001,
      precision: 4,
    });
    this.addFloatPropertyWidget("Noise", "noise", {
      min: 0,
      default: 0.05,
      step: 0.01,
      precision: 3,
    });
    this.addIntPropertyWidget("Seed", "random_state", {
      min: 0,
      default: 42,
    });

    this.size = [250, 230];
  }

  async fetch() {
    const payload = {
      n_samples: parseInt(this.properties.n_samples),
      n_features: parseInt(this.properties.n_features),
      season_period: parseInt(this.properties.season_period),
      trend_strength: Number(this.properties.trend_strength),
      noise: Number(this.properties.noise),
      random_state: parseInt(this.properties.random_state),
    };

    const response = await window.MeltApi.fetch(this.properties.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (err) {
      json = { __raw_text: text };
    }
    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status} ${response.statusText} - ${JSON.stringify(json)}`,
      );
    }
    return json;
  }

  get extractors() {
    return [
      (j) => (j && j.x) || [],
      (j) => (j && j.y) || [],
      (j) => (j && j.shape) || { x: [], y: [] },
    ];
  }

  get defaults() {
    return [[], [], { x: [], y: [] }];
  }
}

TimeSeriesRegressionDataNode.title = "Make Time-Series Regression Data";
TimeSeriesRegressionDataNode.desc =
  "Generate synthetic time-dependent regression data for sequence workflows.";

LiteGraph.registerNodeType(
  "MELT/Data/TimeSeriesRegressionDataNode",
  TimeSeriesRegressionDataNode,
);

class SequenceWindowDataNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Build Sequence Windows");

    this.addInput("x", "array");
    this.addInput("y", "array");

    this.addOutput("x_seq", "array");
    this.addOutput("y_seq", "array");
    this.addOutput("lengths", "array");
    this.addOutput("shape", "object");

    this.addProperty("seq_length", 24, "number");
    this.addProperty("stride", 1, "number");
    this.addProperty("target_offset", 0, "number");

    this.addIntPropertyWidget("Seq Length", "seq_length", {
      min: 2,
      default: 24,
    });
    this.addIntPropertyWidget("Stride", "stride", {
      min: 1,
      default: 1,
    });
    this.addIntPropertyWidget("Target Offset", "target_offset", {
      min: 0,
      default: 0,
    });

    this.size = [250, 150];
  }

  async fetch(resolvedInput) {
    let x = null;
    let y = null;
    if (Array.isArray(resolvedInput)) {
      [x, y] = resolvedInput;
    } else if (resolvedInput && typeof resolvedInput === "object") {
      x = resolvedInput.x;
      y = resolvedInput.y;
    }

    if (!x || !y) {
      x = this.getInputData(0);
      y = this.getInputData(1);
    }

    if (!x || !y || !Array.isArray(x) || !Array.isArray(y)) {
      return {
        x_seq: [],
        y_seq: [],
        lengths: [],
        shape: { x_seq: [], y_seq: [], lengths: [] },
        error: "Missing or invalid x or y input",
      };
    }

    const seqLength = parseInt(this.properties.seq_length);
    const stride = parseInt(this.properties.stride) || 1;
    const targetOffset = parseInt(this.properties.target_offset);

    if (x.length < seqLength) {
      return {
        x_seq: [],
        y_seq: [],
        lengths: [],
        shape: { x_seq: [], y_seq: [], lengths: [] },
        error: `Input data length (${x.length}) is less than seq_length (${seqLength})`,
      };
    }

    const x_seq = [];
    const y_seq = [];
    const lengths = [];

    for (let i = 0; i <= x.length - seqLength; i += stride) {
      // Check if we have enough data for the target offset
      // In the notebook: y_train.append(torch.tensor(y_train[i + seq_length - 1]))
      // This means the target is at index (i + seq_length - 1) relative to the start of the window
      const targetIdx = i + seqLength - 1 + targetOffset;

      if (targetIdx >= y.length) break;

      const windowX = x.slice(i, i + seqLength);
      const targetY = y[targetIdx];

      // Handle both scalar and array targets
      const yVal = Array.isArray(targetY) ? targetY : [targetY];

      x_seq.push(windowX);
      y_seq.push(yVal);
      lengths.push(seqLength);
    }

    if (x_seq.length === 0) {
      return {
        x_seq: [],
        y_seq: [],
        lengths: [],
        shape: { x_seq: [], y_seq: [], lengths: [] },
        error: "No sequences could be built with the given parameters",
      };
    }

    return {
      x_seq,
      y_seq,
      lengths,
      shape: {
        x_seq: [x_seq.length, seqLength, x[0].length || 1],
        y_seq: [y_seq.length, y_seq[0].length],
        lengths: [lengths.length],
      },
    };
  }

  get extractors() {
    return [
      (j) => (j && j.x_seq) || [],
      (j) => (j && j.y_seq) || [],
      (j) => (j && j.lengths) || [],
      (j) => (j && j.shape) || { x_seq: [], y_seq: [], lengths: [] },
    ];
  }

  get defaults() {
    return [[], [], [], { x_seq: [], y_seq: [], lengths: [] }];
  }

  onExecute() {
    super.onExecute();
    this._pXSeq = this._p0;
    this._pYSeq = this._p1;
    this._pLengths = this._p2;
    this._pShape = this._p3;
  }
}

SequenceWindowDataNode.title = "Build Sequence Windows";
SequenceWindowDataNode.desc =
  "Convert flat time-ordered x/y arrays into sequence windows for temporal trainers. For notebook-style seq-to-one behavior, use target_offset=0 (target at end of each window).";

LiteGraph.registerNodeType(
  "MELT/Data/SequenceWindowDataNode",
  SequenceWindowDataNode,
);

class VAESyntheticDataNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Make VAE Synthetic Data");

    this.addOutput("x", "array");
    this.addOutput("labels", "array");
    this.addOutput("shape", "object");

    this.addProperty("n_samples", 2000, "number");
    this.addProperty("n_features", 2, "number");
    this.addProperty("centers", 5, "number");
    this.addProperty("cluster_std", 1.0, "number");
    this.addProperty("random_state", 42, "number");
    this.addProperty("endpoint", "/vae_synthetic_data", "string");

    this.addIntPropertyWidget("N Samples", "n_samples", {
      min: 10,
      default: 2000,
    });
    this.addIntPropertyWidget("N Features", "n_features", {
      min: 2,
      default: 2,
    });
    this.addIntPropertyWidget("Centers", "centers", {
      min: 2,
      default: 5,
    });
    this.addFloatPropertyWidget("Cluster Std", "cluster_std", {
      min: 0.01,
      default: 1.0,
      step: 0.05,
      precision: 2,
    });
    this.addIntPropertyWidget("Seed", "random_state", {
      min: 0,
      default: 42,
    });

    this.size = [230, 210];
  }

  async fetch() {
    const payload = {
      n_samples: parseInt(this.properties.n_samples),
      n_features: parseInt(this.properties.n_features),
      centers: parseInt(this.properties.centers),
      cluster_std: Number(this.properties.cluster_std),
      random_state: parseInt(this.properties.random_state),
    };

    const response = await window.MeltApi.fetch(this.properties.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (err) {
      json = { __raw_text: text };
    }
    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status} ${response.statusText} - ${JSON.stringify(json)}`,
      );
    }
    return json;
  }

  get extractors() {
    return [
      (j) => (j && j.x) || [],
      (j) => (j && j.labels) || [],
      (j) => (j && j.shape) || { x: [], labels: [] },
    ];
  }

  get defaults() {
    return [[], [], { x: [], labels: [] }];
  }
}

VAESyntheticDataNode.title = "Make VAE Synthetic Data";
VAESyntheticDataNode.desc =
  "Generate clustered synthetic feature data for VAE training and latent-space inspection.";

LiteGraph.registerNodeType(
  "MELT/Data/VAESyntheticDataNode",
  VAESyntheticDataNode,
);

class SaveDataNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Save Data");

    this.addInput("x", "array");
    this.addInput("y", "array");
    this.addInput("labels", "array");

    this.addOutput("path", "string");
    this.addOutput("shape", "object");
    this.addOutput("metadata", "object");

    this.addProperty("format", "xlsx", "string");
    this.addProperty("save_dir", "saved_data", "string");
    this.addProperty("filename", "", "string");
    this.addProperty("sheet_name", "data", "string");
    this.addProperty("split", "train", "string");
    this.addProperty("include_index", false, "boolean");
    this.addProperty("overwrite", false, "boolean");
    this.addProperty("label_col", "label", "string");
    this.addProperty("endpoint", "/save_tabular_data", "string");

    this.addDropdownPropertyWidget("Format", "format", {
      values: ["xlsx", "csv"],
      default: "xlsx",
    });
    this.addWidget("text", "Save Dir", this.properties.save_dir, (v) => {
      this.properties.save_dir = v;
    });
    this.addWidget("text", "Filename", this.properties.filename, (v) => {
      this.properties.filename = v;
    });
    this.addWidget("text", "Sheet", this.properties.sheet_name, (v) => {
      this.properties.sheet_name = v;
    });
    this.addDropdownPropertyWidget("Split", "split", {
      values: ["train", "val", "test"],
      default: "train",
    });
    this.addWidget("text", "Label Col", this.properties.label_col, (v) => {
      this.properties.label_col = v;
    });
    this.addBooleanPropertyWidget("Include Index", "include_index", {
      default: false,
    });
    this.addBooleanPropertyWidget("Overwrite", "overwrite", {
      default: false,
    });

    this.size = [290, 275];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties.endpoint || "/save_tabular_data";

    let x = null;
    let y = null;
    let labels = null;

    if (Array.isArray(resolvedInput)) {
      [x, y, labels] = resolvedInput;
    } else if (resolvedInput && typeof resolvedInput === "object") {
      x = resolvedInput.x;
      y = resolvedInput.y;
      labels = resolvedInput.labels;
    }

    if (!x) {
      x = this.getInputData(0);
      y = this.getInputData(1);
      labels = this.getInputData(2);
    }

    if (!x && !y && !labels) {
      return {
        path: "",
        shape: { rows: 0, cols: 0 },
        metadata: {},
        error: "No input data to save",
      };
    }

    const payload = {
      x,
      y,
      labels,
      format: this.properties.format || "xlsx",
      save_dir: this.properties.save_dir || "saved_data",
      filename: this.properties.filename || "",
      sheet_name: this.properties.sheet_name || "data",
      split: this.properties.split || "train",
      include_index: !!this.properties.include_index,
      overwrite: !!this.properties.overwrite,
      label_col: this.properties.label_col || "label",
    };

    const response = await window.MeltApi.fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (err) {
      json = { __raw_text: text };
    }

    if (!response.ok) {
      const errMsg = `HTTP ${response.status} ${response.statusText} - ${JSON.stringify(json)}`;
      throw new Error(errMsg);
    }

    return json;
  }

  get extractors() {
    return [
      (j) => (j && j.path) || "",
      (j) => (j && j.shape) || { rows: 0, cols: 0 },
      (j) => (j && j.metadata) || {},
    ];
  }

  get defaults() {
    return ["", { rows: 0, cols: 0 }, {}];
  }

  onExecute() {
    super.onExecute();
    this._pPath = this._p0;
    this._pShape = this._p1;
    this._pMetadata = this._p2;
  }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue && this._runner) {
      this._runner.invalidate();
    }
  }
}

SaveDataNode.title = "Save Data";
SaveDataNode.desc =
  "Save x/y/labels arrays to XLSX or CSV for downstream analysis workflows.";

LiteGraph.registerNodeType("MELT/Data/SaveDataNode", SaveDataNode);

class PrepareTemporalEvalDataNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Prepare Temporal Eval Data");

    // Inputs: raw 2D arrays + optional pre-fitted scalers from a loaded model
    this.addInput("x", "array");
    this.addInput("y", "array");
    this.addInput("x_normalizer", "object");
    this.addInput("y_normalizer", "object");

    // Outputs - names match what /evaluate_temporal_supervised_model expects
    this.addOutput("x_data_scaled", "object"); // 3D windowed scaled  -> eval node x_data
    this.addOutput("y_data", "object"); // unscaled truth      -> eval node y_data
    this.addOutput("x_data", "object"); // unscaled windowed x (inspection)
    this.addOutput("y_data_scaled", "object"); // scaled y            (inspection)
    this.addOutput("shape", "object");

    this.addProperty("val_size", 0.1, "number");
    this.addProperty("test_size", 0.1, "number");
    this.addProperty("seq_length", 60, "number");
    this.addProperty("seq_to_one", true, "boolean");
    this.addProperty("endpoint", "/prepare_temporal_evaluation_data", "string");

    this.addFloatPropertyWidget("Val Size", "val_size", {
      min: 0.01,
      max: 0.49,
      step: 0.05,
      precision: 2,
      default: 0.1,
    });
    this.addFloatPropertyWidget("Test Size", "test_size", {
      min: 0.01,
      max: 0.49,
      step: 0.05,
      precision: 2,
      default: 0.1,
    });
    this.addIntPropertyWidget("Seq Length", "seq_length", {
      min: 1,
      default: 60,
    });
    this.addBooleanPropertyWidget("Seq-to-One", "seq_to_one", {
      default: true,
    });

    this.size = [300, 200];
  }

  async fetch(resolvedInput) {
    const endpoint =
      this.properties.endpoint || "/prepare_temporal_evaluation_data";

    let x = null,
      y = null,
      xNorm = null,
      yNorm = null;
    if (Array.isArray(resolvedInput)) {
      [x, y, xNorm, yNorm] = resolvedInput;
    }
    if (!x) x = this.getInputData(0);
    if (!y) y = this.getInputData(1);
    if (!xNorm) xNorm = this.getInputData(2);
    if (!yNorm) yNorm = this.getInputData(3);

    if (!x || !y) {
      return {
        x_data_scaled: {},
        y_data: {},
        x_data: {},
        y_data_scaled: {},
        shape: {},
        error: "Missing x or y input",
      };
    }

    const payload = {
      x: x,
      y: y,
      val_size: Number(this.properties.val_size),
      test_size: Number(this.properties.test_size),
      seq_length: parseInt(this.properties.seq_length),
      seq_to_one: !!this.properties.seq_to_one,
      x_normalizer:
        xNorm && typeof xNorm === "object" && Object.keys(xNorm).length > 0
          ? xNorm
          : null,
      y_normalizer:
        yNorm && typeof yNorm === "object" && Object.keys(yNorm).length > 0
          ? yNorm
          : null,
    };

    const response = await window.MeltApi.fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (err) {
      json = { __raw_text: text };
    }
    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status} ${response.statusText} - ${JSON.stringify(json)}`,
      );
    }
    return json;
  }

  get extractors() {
    return [
      (j) => (j && j.x_data_scaled) || {},
      (j) => (j && j.y_data) || {},
      (j) => (j && j.x_data) || {},
      (j) => (j && j.y_data_scaled) || {},
      (j) => (j && j.shape) || {},
    ];
  }

  get defaults() {
    return [{}, {}, {}, {}, {}];
  }

  onExecute() {
    super.onExecute();
    this._pXDataScaled = this._p0;
    this._pYData = this._p1;
    this._pXData = this._p2;
    this._pYDataScaled = this._p3;
    this._pShape = this._p4;
  }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue && this._runner) this._runner.invalidate();
  }
}

PrepareTemporalEvalDataNode.title = "Prepare Temporal Eval Data";
PrepareTemporalEvalDataNode.desc =
  "Splits raw 2D x/y chronologically, applies external scalers from a loaded model, " +
  "and windows into sequences ready for the temporal evaluation node. " +
  "Connect x_data_scaled -> temporal eval x_data, y_data -> temporal eval y_data.";

LiteGraph.registerNodeType(
  "MELT/Data/PrepareTemporalEvalDataNode",
  PrepareTemporalEvalDataNode,
);
