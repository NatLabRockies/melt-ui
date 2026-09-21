// Simple helper to standardize central-runner interactions for single-output nodes.
class NodeRunner {
  constructor(node, outputIndex = 0) {
    this.node = node;
    this.o = outputIndex;

    node._pending = node._pending || false;
    node._currentPromise = node._currentPromise || null;
    node._pOutput = node._pOutput || null;
    node._lastResult = node._lastResult || null;

    node._lastInputSig = node._lastInputSig || null;
    node._invalidateRequested = node._invalidateRequested || false;
  }

  invalidate() {
    const node = this.node;
    node._lastResult = null;
    node._lastInputSig = null;
    node._invalidateRequested = true;
  }

  _makeSig(input) {
    try {
      return JSON.stringify(input);
    } catch {
      return String(input);
    }
  }

  run(maybeInput, fetchFn, extractFn = (j) => j, defaultEmpty = []) {
    const node = this.node;

    // Detect input change when we have a synchronous input
    if (
      !node._pending &&
      maybeInput !== undefined &&
      maybeInput !== null &&
      typeof maybeInput.then !== "function"
    ) {
      const sig = this._makeSig(maybeInput);
      if (node._lastInputSig !== sig) {
        node._lastResult = null;
        node._lastInputSig = sig;
      }
    }

    // Property-based invalidation (set by base class)
    if (node._invalidateRequested) {
      node._lastResult = null;
      node._invalidateRequested = false;
    }

    // If there's an active request, reuse it
    if (node._currentPromise) {
      node.setOutputData(this.o, node._pOutput || node.getOutputData(this.o));
      return node._pOutput;
    }

    // If we have a valid cached result, just re-emit it
    if (node._lastResult !== null && node._lastResult !== undefined) {
      const last = node._lastResult;

      if (last && last.error) {
        node.setOutputData(this.o, last);
        return last;
      }

      if (
        last &&
        Object.prototype.hasOwnProperty.call(
          last,
          "__node_runner_default_immediate",
        )
      ) {
        const immediate = last.__node_runner_default_immediate;
        node.setOutputData(this.o, immediate);
        return immediate;
      }

      const immediate = extractFn(last);
      node.setOutputData(this.o, immediate);
      return immediate;
    }

    // wire a promise to node state & outputs
    const wirePromise = (p) => {
      node._pending = true;
      node._currentPromise = p;

      node._pOutput = p
        .then((json) => {
          if (
            json &&
            Object.prototype.hasOwnProperty.call(
              json,
              "__node_runner_default_immediate",
            )
          ) {
            return json.__node_runner_default_immediate;
          }
          return extractFn(json);
        })
        .catch((e) => ({ error: (e && e.message) || String(e) }));

      node.setOutputData(this.o, node._pOutput);

      p.then((json) => {
        if (
          json &&
          Object.prototype.hasOwnProperty.call(
            json,
            "__node_runner_default_immediate",
          )
        ) {
          node._lastResult = {
            __node_runner_default_immediate:
              json.__node_runner_default_immediate,
          };
          node.setOutputData(this.o, json.__node_runner_default_immediate);
        } else {
          node._lastResult = json;
          const immediate = extractFn(json);
          node.setOutputData(this.o, immediate);
        }
        return json;
      })
        .catch((err) => {
          const errObj = { error: (err && err.message) || String(err) };
          node._lastResult = errObj;
          node.setOutputData(this.o, errObj);
          return errObj;
        })
        .finally(() => {
          node._pending = false;
          node._currentPromise = null;
          try {
            if (node.canvas) node.setDirtyCanvas(true, true);
          } catch (e) {}
        });

      return node._pOutput;
    };

    // If upstream returned a promise, chain it and also track input sig
    if (maybeInput && typeof maybeInput.then === "function") {
      const chained = maybeInput
        .then((resolved) => {
          // record signature of *resolved* input for "second Execute" behavior
          if (resolved !== undefined && resolved !== null) {
            const sig = this._makeSig(resolved);
            if (node._lastInputSig !== sig) {
              node._lastResult = null;
            }
            node._lastInputSig = sig;
          }

          if (resolved === undefined || resolved === null) {
            return { __node_runner_default_immediate: defaultEmpty };
          }
          return fetchFn(resolved);
        })
        .catch((err) => {
          throw err;
        });

      return wirePromise(chained);
    }

    // otherwise call fetchFn directly and wire the returned promise
    let p;
    try {
      p = Promise.resolve(fetchFn(maybeInput));
    } catch (err) {
      p = Promise.reject(err);
    }

    return wirePromise(p);
  }
}

// method to check property-based settings changes
NodeRunner.prototype.checkSettings = function (props = {}) {
  const node = this.node;
  const sig = this._makeSig(props);
  if (node._lastSettingsSig !== sig) {
    node._lastSettingsSig = sig;
    this.invalidate();
    this._propagateInvalidateDownstream();
  }
};

NodeRunner.prototype._propagateInvalidateDownstream = function () {
  const node = this.node;
  const g = node.graph;
  if (!g || !node.outputs) return;

  const visited = new Set();
  const stack = [node];
  const toExecute = [];

  while (stack.length) {
    const n = stack.pop();
    if (!n || visited.has(n.id)) continue;
    visited.add(n.id);

    if (!n.outputs) continue;
    for (let i = 0; i < n.outputs.length; i++) {
      const links = n.outputs[i] && n.outputs[i].links;
      if (!links || !Array.isArray(links)) continue;

      for (let j = 0; j < links.length; j++) {
        const linkRef = links[j];
        let linkedNode = null;

        // Case A: outputs[].links contains node ids (common)
        if (typeof linkRef === "number" || typeof linkRef === "string") {
          linkedNode = g.getNodeById(linkRef);
        }

        // Case B: outputs[].links contains link ids referencing g.links[linkId]
        if (!linkedNode && g.links && g.links[linkRef]) {
          const linkObj = g.links[linkRef];
          const targetId = linkObj && (linkObj.target_id || linkObj.target);
          if (targetId !== undefined && targetId !== null) {
            linkedNode = g.getNodeById(targetId);
          }
        }

        // Case C: sometimes link entries are objects containing target info
        if (!linkedNode && typeof linkRef === "object" && linkRef !== null) {
          const targetId = linkRef.target_id || linkRef.target || linkRef.node;
          if (targetId !== undefined && targetId !== null) {
            linkedNode = g.getNodeById(targetId);
          }
        }

        if (!linkedNode) continue;

        // invalidate runner or fallback fields
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

        // mark dirty and schedule for execution
        try {
          if (typeof linkedNode.setDirty === "function")
            linkedNode.setDirty(true, true);
          if (typeof linkedNode.setDirtyCanvas === "function")
            linkedNode.setDirtyCanvas(true, true);
        } catch (e) {}

        toExecute.push(linkedNode);
        stack.push(linkedNode);
      }
    }
  }

  // Try to trigger the graph execution loop once:
  try {
    if (typeof g.runStep === "function") {
      // run one step to process dirty nodes (preferred if available)
      g.runStep(1);
      return;
    }
    if (typeof g.run === "function") {
      g.run();
      return;
    }
  } catch (e) {
    /* ignore */
  }

  // Fallback: call onExecute on the scheduled nodes (safe-guarded)
  for (let i = 0; i < toExecute.length; i++) {
    const nn = toExecute[i];
    try {
      if (typeof nn.onExecute === "function") nn.onExecute();
    } catch (e) {
      /* ignore */
    }
  }
};

// scan all nodes in the graph and run checkSettings on their runners
NodeRunner.prototype.scanGraphSettings = function () {
  const node = this.node;
  const g = node.graph;
  if (!g) return;
  const nodes = g.nodes || g._nodes || [];
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (!n) continue;
    if (n._runner && typeof n._runner.checkSettings === "function") {
      try {
        n._runner.checkSettings(n.properties || {});
      } catch (e) {
        /* ignore */
      }
    }
  }
};

NodeRunner.prototype.runMulti = function (
  maybeInput,
  fetchFn,
  extractors = [],
  defaults = [],
) {
  const node = this.node;

  const full = this.run(maybeInput, fetchFn, (j) => j || {}, {});
  const fullP =
    full && typeof full.then === "function" ? full : Promise.resolve(full);

  for (let i = 0; i < extractors.length; i++) {
    const ex =
      extractors[i] ||
      ((j) => {
        if (!j) return defaults[i];
        return j[i] !== undefined ? j[i] : defaults[i];
      });

    const mapped = fullP.then((j) => ex(j)).catch(() => defaults[i]);
    node[`_p${i}`] = mapped;
    node.setOutputData(i, mapped);
  }

  return fullP;
};

window.NodeRunner = NodeRunner;

function applyVerticalWidgetLayout(node) {
  if (!node || !Array.isArray(node.widgets) || !node._widgetLayoutConfig) {
    return;
  }

  const cfg = node._widgetLayoutConfig;
  const defaultWidgetHeight = LiteGraph.NODE_WIDGET_HEIGHT || 22;
  let cursorY = typeof cfg.top === "number" ? cfg.top : 0;

  for (let i = 0; i < node.widgets.length; i++) {
    const widget = node.widgets[i];
    if (!widget) continue;

    widget.y = cursorY;

    let h = defaultWidgetHeight;
    if (typeof widget.computeSize === "function") {
      try {
        const computed = widget.computeSize(node.size ? node.size[0] : 0);
        if (computed && typeof computed[1] === "number" && computed[1] > 0) {
          h = computed[1];
        }
      } catch {
        // Keep default height when widget size computation throws.
      }
    }

    cursorY += h + cfg.gap;
  }

  if (cfg.autoSize !== false) {
    const minHeight = typeof cfg.minHeight === "number" ? cfg.minHeight : 0;
    const targetHeight = Math.max(minHeight, cursorY + cfg.bottom);
    if (!node.size || !Array.isArray(node.size)) {
      node.size = [200, targetHeight];
    } else {
      node.size[1] = Math.max(node.size[1] || 0, targetHeight);
    }
  }
}

LGraphNode.prototype.configureWidgetLayout = function (options = {}) {
  const prev = this._widgetLayoutConfig || {};
  this._widgetLayoutConfig = {
    top: typeof options.top === "number" ? options.top : prev.top || 0,
    gap: typeof options.gap === "number" ? options.gap : prev.gap || 4,
    bottom:
      typeof options.bottom === "number" ? options.bottom : prev.bottom || 12,
    minHeight:
      typeof options.minHeight === "number"
        ? options.minHeight
        : prev.minHeight || 0,
    autoSize:
      typeof options.autoSize === "boolean"
        ? options.autoSize
        : prev.autoSize !== false,
  };
  this.applyWidgetLayout();
};

LGraphNode.prototype.setWidgetTopOffset = function (top) {
  if (!this._widgetLayoutConfig) {
    this.configureWidgetLayout({ top });
    return;
  }
  this._widgetLayoutConfig.top = typeof top === "number" ? top : 0;
  this.applyWidgetLayout();
};

LGraphNode.prototype.applyWidgetLayout = function () {
  applyVerticalWidgetLayout(this);
};

class AsyncNodeBase extends LiteGraph.LGraphNode {
  constructor(title, primaryOutputIndex = 0) {
    super();
    this.title = title;
    this._runner = new NodeRunner(this, primaryOutputIndex);
    this._widgetLayoutConfig = null;
  }

  configureWidgetLayout(options = {}) {
    const prev = this._widgetLayoutConfig || {};
    this._widgetLayoutConfig = {
      top: typeof options.top === "number" ? options.top : prev.top || 0,
      gap: typeof options.gap === "number" ? options.gap : prev.gap || 4,
      bottom:
        typeof options.bottom === "number" ? options.bottom : prev.bottom || 12,
      minHeight:
        typeof options.minHeight === "number"
          ? options.minHeight
          : prev.minHeight || 0,
      autoSize:
        typeof options.autoSize === "boolean"
          ? options.autoSize
          : prev.autoSize !== false,
    };
    this.applyWidgetLayout();
  }

  setWidgetTopOffset(top) {
    if (!this._widgetLayoutConfig) {
      this.configureWidgetLayout({ top });
      return;
    }
    this._widgetLayoutConfig.top = typeof top === "number" ? top : 0;
    this.applyWidgetLayout();
  }

  applyWidgetLayout() {
    applyVerticalWidgetLayout(this);
  }

  async fetch(resolvedInput) {
    throw new Error("fetch(resolvedInput) must be implemented");
  }

  extractPrimary(json) {
    return json;
  }

  get defaultPrimary() {
    return [];
  }

  onExecute() {
    // scan whole graph for settings changes; those nodes will mark themselves dirty/downstream
    if (this._runner && typeof this._runner.scanGraphSettings === "function") {
      this._runner.scanGraphSettings();
    } else if (this._runner) {
      this._runner.checkSettings(this.properties || {});
    }

    const input = this.getInputData(0);
    const fetcher = (resolved) => this.fetch(resolved);

    this._runner.run(
      input,
      fetcher,
      (json) => this.extractPrimary(json),
      this.defaultPrimary,
    );
  }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue) {
      this._runner.invalidate();
    }
  }
}

class AsyncMultiOutputNodeBase extends LiteGraph.LGraphNode {
  constructor(title) {
    super();
    this.title = title;
    this._runner = new NodeRunner(this, 0);
    this._widgetLayoutConfig = null;
  }

  configureWidgetLayout(options = {}) {
    const prev = this._widgetLayoutConfig || {};
    this._widgetLayoutConfig = {
      top: typeof options.top === "number" ? options.top : prev.top || 0,
      gap: typeof options.gap === "number" ? options.gap : prev.gap || 4,
      bottom:
        typeof options.bottom === "number" ? options.bottom : prev.bottom || 12,
      minHeight:
        typeof options.minHeight === "number"
          ? options.minHeight
          : prev.minHeight || 0,
      autoSize:
        typeof options.autoSize === "boolean"
          ? options.autoSize
          : prev.autoSize !== false,
    };
    this.applyWidgetLayout();
  }

  setWidgetTopOffset(top) {
    if (!this._widgetLayoutConfig) {
      this.configureWidgetLayout({ top });
      return;
    }
    this._widgetLayoutConfig.top = typeof top === "number" ? top : 0;
    this.applyWidgetLayout();
  }

  applyWidgetLayout() {
    applyVerticalWidgetLayout(this);
  }

  async fetch(resolvedInput) {
    throw new Error("fetch(resolvedInput) must be implemented");
  }

  get extractors() {
    return [];
  }

  get defaults() {
    return [];
  }

  onExecute() {
    // scan whole graph for settings changes first
    if (this._runner && typeof this._runner.scanGraphSettings === "function") {
      this._runner.scanGraphSettings();
    } else if (this._runner) {
      this._runner.checkSettings(this.properties || {});
    }

    // Gather inputs similar to AsyncNodeBase:
    // - if no inputs -> null
    // - if single input -> pass that value (promise or value)
    // - if multiple inputs -> pass an array [in0, in1, ...] (promises allowed)
    const inputCount = (this.inputs && this.inputs.length) || 0;
    let maybeInput = null;
    if (inputCount === 1) {
      maybeInput = this.getInputData(0);
    } else if (inputCount > 1) {
      const arr = [];
      let hasPromise = false;
      for (let i = 0; i < inputCount; i++) {
        const v = this.getInputData(i);
        arr.push(v);
        if (v && typeof v.then === "function") hasPromise = true;
      }
      maybeInput = hasPromise ? Promise.all(arr) : arr;
    }

    const fetcher = (resolved) => this.fetch(resolved);

    this._runner.runMulti(maybeInput, fetcher, this.extractors, this.defaults);
  }

  //   onExecute() {
  //     // scan whole graph for settings changes first
  //     if (this._runner && typeof this._runner.scanGraphSettings === "function") {
  //       this._runner.scanGraphSettings();
  //     } else if (this._runner) {
  //       this._runner.checkSettings(this.properties || {});
  //     }

  //     const fetcher = (resolved) => this.fetch(resolved);

  //     this._runner.runMulti(
  //       null, // no upstream input by default
  //       fetcher,
  //       this.extractors,
  //       this.defaults
  //     );
  //   }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue) {
      this._runner.invalidate();
    }
  }
}
