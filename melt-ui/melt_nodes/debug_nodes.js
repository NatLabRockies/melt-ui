function PrintDisplayNode() {
  // One input for anything, one passthrough output
  this.addInput("value", "*");
  this.addOutput("value", "*");

  // Big node so the box is large
  this.size = [320, 200];

  // Last text to display
  this._text = "";

  // Scroll state
  this._scrollOffset = 0; // in pixels
  this._maxScroll = 0;

  // Wrap cache
  this._wrapCache = {
    text: "",
    width: 0,
    lines: [],
  };

  // Drag-to-scroll state
  this._isDraggingScroll = false;
  this._dragStartY = 0;
  this._scrollStart = 0;

  // Optional label
  this.addProperty("label", "", "string");

  this._pendingPromise = null;
}

PrintDisplayNode.title = "Print";
PrintDisplayNode.desc = "Display last value in a large box (like print())";

PrintDisplayNode.prototype._valueToString = function (v) {
  try {
    if (typeof v === "string") return v;
    if (typeof v === "number" || typeof v === "boolean") return String(v);
    if (v === null) return "null";
    if (v === undefined) return "undefined";
    if (Array.isArray(v)) {
      return (
        "Array(" +
        v.length +
        "): [\n  " +
        v.map((item) => this._valueToString(item)).join(",\n  ") +
        "\n]"
      );
    }
    if (typeof v === "object") {
      return JSON.stringify(v, null, 2); // Pretty print objects
    }
    return String(v);
  } catch (e) {
    return String(v);
  }
};

PrintDisplayNode.prototype.onExecute = function () {
  const value = this.getInputData(0);

  // passthrough so you can drop this in the middle of a chain
  this.setOutputData(0, value);

  // If input is a Promise: show pending state and attach one handler to update when resolved
  if (value && typeof value.then === "function") {
    // already listening to this same promise
    if (this._pendingPromise === value) return;

    // replace pending promise handler
    this._pendingPromise = value;
    this._text = "Promise pending...";
    this._scrollOffset = 0;
    this._wrapCache.text = "";

    // immediately redraw to show pending state
    try {
      this.setDirtyCanvas(true, true);
    } catch (e) {}

    const self = this;
    value
      .then((resolved) => {
        // ignore if a new promise replaced this one
        if (self._pendingPromise !== value) return;

        // update passthrough output to resolved value
        self.setOutputData(0, resolved);

        let text = self._valueToString(resolved);
        if (self.properties && self.properties.label) {
          text = self.properties.label + ": " + text;
        }

        self._text = text;
        self._scrollOffset = 0;
        self._wrapCache.text = "";

        // redraw, but DO NOT call graph.runStep here (central runner controls execution)
        try {
          self.setDirtyCanvas(true, true);
        } catch (e) {}

        // clear the tracked pending promise so future promises can be attached
        if (self._pendingPromise === value) self._pendingPromise = null;
      })
      .catch((err) => {
        // ignore if a new promise replaced this one
        if (self._pendingPromise !== value) return;

        const msg = err && err.message ? err.message : String(err);
        self._text = "Promise rejected: " + msg;
        self._scrollOffset = 0;
        self._wrapCache.text = "";

        try {
          self.setDirtyCanvas(true, true);
        } catch (e) {}

        if (self._pendingPromise === value) self._pendingPromise = null;
      });

    return;
  }

  // non-promise path: render immediately like before
  if (value === undefined) return;

  let text = this._valueToString(value);

  if (this.properties && this.properties.label) {
    text = this.properties.label + ": " + text;
  }

  // overwrite previous text – no history
  this._text = text;
  this._scrollOffset = 0; // reset scroll to top
  this._wrapCache.text = ""; // invalidate wrap cache

  // ensure the node repaints immediately
  try {
    this.setDirtyCanvas(true, true);
  } catch (e) {}
};

// Wrap text into lines based on width, respecting \n and breaking long chunks.
// Uses a cache so we only recompute when text or width changes.
PrintDisplayNode.prototype._getWrappedLines = function (ctx, text, maxWidth) {
  const cache = this._wrapCache;
  if (cache.text === text && cache.width === maxWidth && cache.lines.length) {
    return cache.lines;
  }

  const outLines = [];
  const paragraphs = String(text).split("\n");

  for (let p = 0; p < paragraphs.length; p++) {
    const baseLine = paragraphs[p];
    if (baseLine === "") {
      outLines.push("");
      continue;
    }

    let line = "";
    for (let i = 0; i < baseLine.length; i++) {
      const ch = baseLine[i];
      const testLine = line + ch;
      const metrics = ctx.measureText(testLine);

      if (metrics.width > maxWidth && line !== "") {
        outLines.push(line);
        line = ch;
      } else {
        line = testLine;
      }
    }
    outLines.push(line);
  }

  cache.text = text;
  cache.width = maxWidth;
  cache.lines = outLines;
  return outLines;
};

PrintDisplayNode.prototype.onDrawForeground = function (ctx) {
  if (this.flags && this.flags.collapsed) return;

  ctx.save();

  const margin = 8;
  const titleHeight = 20; // space for node title
  const x = margin;
  const y = titleHeight + margin;
  const w = this.size[0] - margin * 2;
  const h = this.size[1] - y - margin;

  // background box
  ctx.fillStyle = "rgba(0,0,0,0.4)";
  ctx.fillRect(x, y, w, h);

  // border
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.strokeRect(x, y, w, h);

  // text style
  ctx.font = "12px monospace";
  ctx.fillStyle = "#DDD";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";

  const innerMargin = 4;
  const lineHeight = 14;
  const textX = x + innerMargin;
  const textY = y + innerMargin;
  const textW = w - innerMargin * 2;
  const textH = h - innerMargin * 2;

  const text = this._text || "";

  // Wrapped lines + scroll bounds
  const lines = this._getWrappedLines(ctx, text, textW);
  const totalHeight = lines.length * lineHeight;
  const maxScroll = Math.max(0, totalHeight - textH);
  this._maxScroll = maxScroll;

  // Clamp scroll offset
  let scroll = this._scrollOffset || 0;
  if (scroll < 0) scroll = 0;
  if (scroll > maxScroll) scroll = maxScroll;
  this._scrollOffset = scroll;

  // Draw only visible lines with scroll offset
  const visibleTop = textY;
  const visibleBottom = textY + textH;

  for (let i = 0; i < lines.length; i++) {
    const lineY = textY + i * lineHeight - scroll;
    if (lineY + lineHeight < visibleTop) continue; // above viewport
    if (lineY > visibleBottom) break; // below viewport
    ctx.fillText(lines[i], textX, lineY);
  }

  // Simple scrollbar indicator on the right
  if (maxScroll > 0) {
    const barWidth = 4;
    const barX = x + w - barWidth - 2;
    const barY = y + 2;
    const barH = h - 4;

    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fillRect(barX, barY, barWidth, barH);

    const ratio = textH / totalHeight;
    const thumbH = Math.max(10, barH * ratio);
    const scrollRatio = scroll / maxScroll;
    const thumbY = barY + (barH - thumbH) * scrollRatio;

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.fillRect(barX, thumbY, barWidth, thumbH);
  }

  ctx.restore();
};

// ---- Drag-to-scroll support ----
// pos is in node-local coordinates: [x, y] relative to this node
PrintDisplayNode.prototype._isInTextBox = function (pos) {
  const margin = 8;
  const titleHeight = 20;
  const x = margin;
  const y = titleHeight + margin;
  const w = this.size[0] - margin * 2;
  const h = this.size[1] - y - margin;

  const innerMargin = 4;
  const textX = x + innerMargin;
  const textY = y + innerMargin;
  const textW = w - innerMargin * 2;
  const textH = h - innerMargin * 2;

  const mx = pos[0];
  const my = pos[1];

  return (
    mx >= textX && mx <= textX + textW && my >= textY && my <= textY + textH
  );
};

PrintDisplayNode.prototype.onMouseDown = function (event, pos, graphcanvas) {
  if (!this._isInTextBox(pos)) return false;

  this._isDraggingScroll = true;
  this._dragStartY = pos[1];
  this._scrollStart = this._scrollOffset || 0;

  if (event.preventDefault) event.preventDefault();
  if (event.stopPropagation) event.stopPropagation();
  return true; // consume the event
};

PrintDisplayNode.prototype.onMouseMove = function (event, pos, graphcanvas) {
  if (!this._isDraggingScroll) return false;

  const deltaY = pos[1] - this._dragStartY;
  let newOffset = this._scrollStart - deltaY;

  if (newOffset < 0) newOffset = 0;
  if (this._maxScroll && newOffset > this._maxScroll) {
    newOffset = this._maxScroll;
  }

  this._scrollOffset = newOffset;
  this.setDirtyCanvas(true, true);

  if (event.preventDefault) event.preventDefault();
  if (event.stopPropagation) event.stopPropagation();
  return true;
};

PrintDisplayNode.prototype.onMouseUp = function (event, pos, graphcanvas) {
  if (!this._isDraggingScroll) return false;

  this._isDraggingScroll = false;

  if (event.preventDefault) event.preventDefault();
  if (event.stopPropagation) event.stopPropagation();
  return true;
};

PrintDisplayNode.prototype.onMouseWheel = function (event, pos, graphcanvas) {
  if (!this._isInTextBox(pos)) return false;

  const delta = event.deltaY || event.wheelDeltaY || event.wheelDelta || 0;
  const step = 30; // pixels per wheel tick

  this._scrollOffset = (this._scrollOffset || 0) + (delta > 0 ? step : -step);

  if (this._scrollOffset < 0) this._scrollOffset = 0;
  if (this._maxScroll && this._scrollOffset > this._maxScroll) {
    this._scrollOffset = this._maxScroll;
  }

  this.setDirtyCanvas(true, true);

  if (event.preventDefault) event.preventDefault();
  if (event.stopPropagation) event.stopPropagation();
  return true;
};

LiteGraph.registerNodeType("MELT/Debug/PrintDisplay", PrintDisplayNode);
