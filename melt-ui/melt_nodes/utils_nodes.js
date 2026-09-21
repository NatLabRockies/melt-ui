// utils_nodes.js — Graph organisation utilities

// ---------------------------------------------------------------------------
// RerouteNode
// A tiny pass-through node whose only purpose is to redirect wires so the
// graph layout stays readable. Input and output are both generic (any type).
// ---------------------------------------------------------------------------
function RerouteNode() {
  this.addInput("in", 0); // 0 = any type in LiteGraph
  this.addOutput("out", 0);

  // Compact rectangle — just wide enough to hold the two connector dots
  this.size = [75, 30];
}

RerouteNode.title = "Reroute";
RerouteNode.desc =
  "Pass any value straight through. Use to redirect and organise wires in the graph.";
RerouteNode.title_mode = LiteGraph.NO_TITLE; // hide title bar → wire-junction look
RerouteNode.shape = LiteGraph.ROUND_SHAPE;

RerouteNode.prototype.onExecute = function () {
  this.setOutputData(0, this.getInputData(0));
};

LiteGraph.registerNodeType("MELT/Utils/Reroute", RerouteNode);
