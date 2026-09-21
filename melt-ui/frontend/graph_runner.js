class GraphRunner {
  constructor(graph) {
    this.graph = graph;
    this._running = false; // prevent concurrent runs
  }

  _collectPromiseOutputs() {
    const nodes = this.graph._nodes || this.graph.nodes || [];
    const promises = new Set();

    for (const node of nodes) {
      if (!node || !node.outputs) continue;
      for (let i = 0; i < node.outputs.length; i++) {
        try {
          const out = node.getOutputData(i);
          if (out && typeof out.then === "function") {
            promises.add(out);
          }
        } catch (e) {}
      }
    }
    return Array.from(promises);
  }

  runOnce() {
    try {
      this.graph.runStep(1);
    } catch (e) {
      console.error("GraphRunner.runOnce runStep error:", e);
    }
    return this._collectPromiseOutputs();
  }

  async runUntilSettled(maxRounds = 10) {
    if (this._running) {
      console.warn("GraphRunner: run already in progress");
      return false;
    }
    this._running = true;
    try {
      for (let round = 0; round < maxRounds; round++) {
        const promises = this.runOnce();
        if (!promises || promises.length === 0) return true;
        await Promise.allSettled(promises);
      }
      console.warn("GraphRunner: reached maxRounds in runUntilSettled");
      return false;
    } finally {
      this._running = false;
    }
  }
}

window.GraphRunner = GraphRunner;
