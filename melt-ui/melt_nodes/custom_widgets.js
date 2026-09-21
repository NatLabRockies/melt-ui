// INTEGER helper
LGraphNode.prototype.addIntPropertyWidget = function (
  label,
  propertyName,
  options = {},
) {
  const node = this;

  if (!node.properties) node.properties = {};
  if (node.properties[propertyName] === undefined) {
    node.addProperty(propertyName, options.default ?? 0, "number");
  }

  const initial = Number.isFinite(node.properties[propertyName])
    ? node.properties[propertyName]
    : parseInt(node.properties[propertyName] || 0, 10);

  const widgetOptions = Object.assign(
    {
      precision: 0,
      min: 0,
      step: 1,
    },
    options,
  );

  const widget = node.addWidget(
    "number",
    label,
    initial,
    function (v, graphCanvas, n) {
      const widget = this;
      const opts = widget.options || {};

      const min = typeof opts.min === "number" ? opts.min : -Infinity;
      const max = typeof opts.max === "number" ? opts.max : +Infinity;

      const prev = Number.isFinite(n.properties[propertyName])
        ? n.properties[propertyName]
        : parseInt(n.properties[propertyName] || 0, 10);

      let raw = Number(v);
      if (!Number.isFinite(raw)) raw = prev;

      const diff = raw - prev;
      let next;

      // small diff => likely arrow/slider => treat as ±1
      if (Math.abs(diff) < 0.9) {
        if (diff > 0) next = prev + 1;
        else if (diff < 0) next = prev - 1;
        else next = prev;
      } else {
        // big jump => user typed a value
        next = Math.round(raw);
      }

      if (next < min) next = min;
      if (next > max) next = max;

      n.properties[propertyName] = next;
      widget.value = next;

      return next;
    },
    widgetOptions,
  );

  widget.linkedProperty = propertyName;
  return widget;
};

// FLOAT helper
LGraphNode.prototype.addFloatPropertyWidget = function (
  label,
  propertyName,
  options = {},
) {
  const node = this;

  if (!node.properties) node.properties = {};

  // Only create the property if it doesn't already exist
  if (node.properties[propertyName] === undefined) {
    node.addProperty(propertyName, options.default ?? 0, "number");
  }

  const initial = Number.isFinite(node.properties[propertyName])
    ? node.properties[propertyName]
    : parseFloat(node.properties[propertyName] || 0);

  const widgetOptions = Object.assign({ property: propertyName }, options);

  const widget = node.addWidget(
    "number",
    label,
    initial,
    function (v, graphCanvas, n) {
      const widget = this;
      const opts = widget.options || {};

      const min = typeof opts.min === "number" ? opts.min : -Infinity;
      const max = typeof opts.max === "number" ? opts.max : +Infinity;

      let val = Number(v);
      if (!Number.isFinite(val)) {
        // fall back to previous value if parsing failed
        val = Number(n.properties[propertyName]) || 0;
      }

      // clamp
      if (val < min) val = min;
      if (val > max) val = max;

      // optional precision
      if (typeof opts.precision === "number") {
        val = parseFloat(val.toFixed(opts.precision));
      }

      n.properties[propertyName] = val;
      widget.value = val; // keep widget in sync

      return val;
    },
    widgetOptions,
  );

  widget.linkedProperty = propertyName;
  return widget;
};

// Boolean helper
LGraphNode.prototype.addBooleanPropertyWidget = function (
  label,
  propertyName,
  options = {},
) {
  const node = this;

  if (!node.properties) node.properties = {};
  if (node.properties[propertyName] === undefined) {
    node.addProperty(propertyName, options.default ?? false, "boolean");
  }

  const initial = Boolean(node.properties[propertyName]);

  const widgetOptions = Object.assign({ property: propertyName }, options);

  const widget = node.addWidget(
    "toggle",
    label,
    initial,
    // signature matches other widgets: (value, graphCanvas, node)
    function (v, graphCanvas, n) {
      const widget = this;
      const next = Boolean(v);

      // ensure we update the node property and the widget UI
      n.properties[propertyName] = next;
      widget.value = next;

      // returning the new value ensures LiteGraph updates the internal widget state
      return next;
    },
    widgetOptions,
  );

  widget.linkedProperty = propertyName;
  return widget;
};

// Dropdown (combo) helper
LGraphNode.prototype.addDropdownPropertyWidget = function (
  label,
  propertyName,
  options = {},
) {
  const node = this;

  if (!node.properties) node.properties = {};
  if (node.properties[propertyName] === undefined) {
    node.addProperty(
      propertyName,
      options.default ?? (options.values ? options.values[0] : ""),
      "string",
    );
  }

  const initial = String(node.properties[propertyName]);

  const widgetOptions = Object.assign({ property: propertyName }, options);

  const widget = node.addWidget(
    "combo",
    label,
    initial,
    function (v, graphCanvas, n) {
      const widget = this;
      const prev =
        n.properties[propertyName] === undefined ||
        n.properties[propertyName] === null
          ? ""
          : String(n.properties[propertyName]);
      const next = String(v);

      n.properties[propertyName] = next;
      widget.value = next;

      if (next !== prev) {
        if (typeof n.onPropertyChanged === "function") {
          n.onPropertyChanged(propertyName, next, prev);
        } else if (n._runner && typeof n._runner.invalidate === "function") {
          n._runner.invalidate();
        }

        invalidateDownstreamNodeRunners(n);

        if (typeof n.setDirtyCanvas === "function") {
          n.setDirtyCanvas(true, true);
        }
      }

      return next;
    },
    widgetOptions,
  );

  widget.linkedProperty = propertyName;
  return widget;
};

function invalidateDownstreamNodeRunners(node) {
  const graph = node && node.graph;
  if (!graph || !node.outputs) return;

  const visited = new Set();
  const stack = [node];

  while (stack.length) {
    const current = stack.pop();
    if (!current || visited.has(current.id)) continue;
    visited.add(current.id);

    if (!current.outputs) continue;
    for (let i = 0; i < current.outputs.length; i++) {
      const links = current.outputs[i] && current.outputs[i].links;
      if (!links || !Array.isArray(links)) continue;

      for (let j = 0; j < links.length; j++) {
        const linkRef = links[j];
        let linkedNode = null;

        if (typeof linkRef === "number" || typeof linkRef === "string") {
          linkedNode = graph.getNodeById(linkRef);
        }

        if (!linkedNode && graph.links && graph.links[linkRef]) {
          const linkObj = graph.links[linkRef];
          const targetId = linkObj && (linkObj.target_id || linkObj.target);
          if (targetId !== undefined && targetId !== null) {
            linkedNode = graph.getNodeById(targetId);
          }
        }

        if (!linkedNode && typeof linkRef === "object" && linkRef !== null) {
          const targetId = linkRef.target_id || linkRef.target || linkRef.node;
          if (targetId !== undefined && targetId !== null) {
            linkedNode = graph.getNodeById(targetId);
          }
        }

        if (!linkedNode) continue;

        if (
          linkedNode._runner &&
          typeof linkedNode._runner.invalidate === "function"
        ) {
          linkedNode._runner.invalidate();
        } else {
          linkedNode._lastResult = null;
          linkedNode._lastInputSig = null;
          linkedNode._invalidateRequested = true;
        }

        if (typeof linkedNode.setDirtyCanvas === "function") {
          linkedNode.setDirtyCanvas(true, true);
        }

        stack.push(linkedNode);
      }
    }
  }
}

LGraphNode.prototype.invalidateDownstreamNodeRunners = function () {
  invalidateDownstreamNodeRunners(this);
};

// TEXT helper
LGraphNode.prototype.addTextPropertyWidget = function (
  label,
  propertyName,
  options = {},
) {
  const node = this;

  if (!node.properties) node.properties = {};
  if (node.properties[propertyName] === undefined) {
    node.addProperty(propertyName, options.default ?? "", "string");
  }

  const initial =
    node.properties[propertyName] === undefined ||
    node.properties[propertyName] === null
      ? ""
      : String(node.properties[propertyName]);

  const widgetOptions = Object.assign({ property: propertyName }, options);
  const hasEmptyValue = Object.prototype.hasOwnProperty.call(
    options,
    "emptyValue",
  );

  const widget = node.addWidget(
    "text",
    label,
    initial,
    function (v, graphCanvas, n) {
      const widget = this;
      const prev =
        n.properties[propertyName] === undefined ||
        n.properties[propertyName] === null
          ? ""
          : String(n.properties[propertyName]);
      let next = v === undefined || v === null ? "" : String(v);
      if (hasEmptyValue && next === "") next = String(options.emptyValue);

      n.properties[propertyName] = next;
      widget.value = next;

      if (next !== prev) {
        if (typeof n.onPropertyChanged === "function") {
          n.onPropertyChanged(propertyName, next, prev);
        } else if (n._runner && typeof n._runner.invalidate === "function") {
          n._runner.invalidate();
        }

        invalidateDownstreamNodeRunners(n);

        if (typeof n.setDirtyCanvas === "function") {
          n.setDirtyCanvas(true, true);
        }
      }

      return next;
    },
    widgetOptions,
  );

  widget.linkedProperty = propertyName;
  return widget;
};

// INTEGER ARRAY helper: edit as "1, 2, 3" but store [1, 2, 3]
LGraphNode.prototype.addIntArrayPropertyWidget = function (
  label,
  propertyName,
  options = {},
) {
  const node = this;
  if (!node.properties) node.properties = {};

  // ---- helpers ----
  function toInt(el) {
    const parsed = parseInt(el, 10);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function normalizeArray(value) {
    if (Array.isArray(value)) {
      return value.map(toInt);
    }
    if (typeof value === "string") {
      // split on commas OR whitespace
      return value
        .split(/[,\s]+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
        .map(toInt);
    }
    if (value == null) return [];
    return [toInt(value)];
  }

  // ---- initial property value ----
  if (node.properties[propertyName] === undefined) {
    const def = normalizeArray(options.default ?? []);
    node.addProperty(propertyName, def, "array");
  }

  const initialArray = normalizeArray(node.properties[propertyName]);
  const initialText = initialArray.join(", ");

  // clone options so we don't mutate caller's object
  const widgetOptions = Object.assign({ property: propertyName }, options);
  delete widgetOptions.default;

  // ---- widget (use "text" so it actually renders) ----
  const widget = node.addWidget(
    "text",
    label,
    initialText,
    function (v, graphCanvas, n) {
      const widget = this;

      const arr = normalizeArray(v);
      n.properties[propertyName] = arr;

      // keep the display nicely formatted
      widget.value = arr.join(", ");
      return widget.value;
    },
    widgetOptions,
  );

  widget.linkedProperty = propertyName;
  return widget;
};

LGraphNode.prototype.syncPropertiesToWidgets = function () {
  if (!this.widgets || !this.properties) return;

  for (const widget of this.widgets) {
    const propertyKey =
      widget?.linkedProperty || widget?.options?.property || null;
    if (!propertyKey || !(propertyKey in this.properties)) continue;

    const value = this.properties[propertyKey];
    if (Array.isArray(value) && widget.type === "text") {
      widget.value = value.join(", ");
    } else {
      widget.value = value;
    }
  }

  if (typeof this.setDirtyCanvas === "function") {
    this.setDirtyCanvas(true, true);
  }
};

// Button Group helper
LGraphNode.prototype.addButtonGroup = function (buttons, options = {}) {
  const node = this;
  const gap = typeof options.gap === "number" ? options.gap : 6;
  const widgetHeight = LiteGraph.NODE_WIDGET_HEIGHT || 22;

  const widget = this.addCustomWidget({
    name: "ButtonGroup_" + ((this.widgets && this.widgets.length) || 0),
    value: 0, // unused, but LiteGraph likes a value field
    _areas: [], // per-button hit areas
    buttons: buttons, // keep a reference

    // how tall this row is
    computeSize(width) {
      return [width, widgetHeight];
    },

    // draw side-by-side buttons
    draw(ctx, node, width, y, height) {
      const margin = 6;
      const totalGap = gap * (this.buttons.length - 1);
      const btnWidth = (width - margin * 2 - totalGap) / this.buttons.length;
      const btnHeight = height - 4;

      this._areas = [];

      ctx.save();
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = (LiteGraph.NODE_TEXT_SIZE || 14) + "px sans-serif";

      let x = margin;

      for (let i = 0; i < this.buttons.length; i++) {
        const b = this.buttons[i];
        const area = { x, y: y + 2, w: btnWidth, h: btnHeight, index: i };
        this._areas.push(area);

        // background
        ctx.fillStyle = "#333";
        ctx.fillRect(area.x, area.y, area.w, area.h);

        // border
        ctx.strokeStyle = "#555";
        ctx.strokeRect(area.x, area.y, area.w, area.h);

        // label
        ctx.fillStyle = "#ddd";
        ctx.fillText(
          b.label || "Button",
          area.x + area.w / 2,
          area.y + area.h / 2 + 1,
        );

        x += btnWidth + gap;
      }

      ctx.restore();
    },

    // handle clicks
    mouse(event, pos, node) {
      // In ComfyUI with pointer events, this will be "pointerdown"
      if (event.type !== "pointerdown" && event.type !== "mousedown") {
        return false;
      }

      if (!this._areas || !this._areas.length) return false;

      const x = pos[0]; // already node-local, do NOT subtract node.pos
      const y = pos[1];

      for (const area of this._areas) {
        if (
          x >= area.x &&
          x <= area.x + area.w &&
          y >= area.y &&
          y <= area.y + area.h
        ) {
          const def = this.buttons[area.index];
          if (def && typeof def.onClick === "function") {
            try {
              def.onClick.call(node, event);
            } catch (err) {
              console.error(err);
            }
          }
          // returning true tells LiteGraph "I handled this, don't start dragging the node"
          return true;
        }
      }

      return false;
    },
  });

  return widget;
};

// "Engineering-friendly" number widget that keeps the classic inline arrows.
LGraphNode.prototype.addEngineeringPropertyWidget = function (
  label,
  propertyName,
  options = {},
) {
  const node = this;
  if (!node.properties) node.properties = {};

  // Default value (you can pass default: 1e-3, or default: "1e-3")
  let defaultVal = 1e-3;
  if (typeof options.default === "number") {
    defaultVal = options.default;
  } else if (typeof options.default === "string") {
    const parsed = Number(options.default);
    if (Number.isFinite(parsed)) defaultVal = parsed;
  }

  // Create backing numeric property if it doesn't exist
  if (node.properties[propertyName] === undefined) {
    node.addProperty(propertyName, defaultVal, "number");
  }

  // Initial numeric value
  const initial = Number(node.properties[propertyName]) || defaultVal;

  const widgetOptions = Object.assign({}, options);
  delete widgetOptions.default;
  widgetOptions.property = propertyName;

  // Reasonable defaults
  if (widgetOptions.precision == null) widgetOptions.precision = 6; // how many decimals to show
  if (widgetOptions.step == null) widgetOptions.step = defaultVal; // arrow step size

  // Optional min/max clamping
  const min =
    typeof widgetOptions.min === "number" ? widgetOptions.min : -Infinity;
  const max =
    typeof widgetOptions.max === "number" ? widgetOptions.max : +Infinity;

  const widget = node.addWidget(
    "number",
    label,
    initial,
    function (v, graphCanvas, n) {
      let val = Number(v);

      // If parsing failed, fall back to previous value
      if (!Number.isFinite(val)) {
        val = Number(n.properties[propertyName]) || defaultVal;
      }

      // Clamp
      if (val < min) val = min;
      if (val > max) val = max;

      // Precision
      if (typeof widgetOptions.precision === "number") {
        val = Number(val.toFixed(widgetOptions.precision));
      }

      n.properties[propertyName] = val;
      return val;
    },
    widgetOptions,
  );

  // OPTIONAL: if your LiteGraph version supports a format function,
  // you can display engineering notation like "7e-4" instead of 0.0007:
  //
  widget.options.format = function (v) {
    if (!Number.isFinite(v) || v === 0) return "0";
    return v.toExponential(0); // 0 decimals in mantissa -> "7e-4"
  };

  widget.linkedProperty = propertyName;

  return widget;
};
