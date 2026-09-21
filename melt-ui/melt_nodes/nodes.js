// frontend/nodes.js
function NumberNode() {
  this.addOutput("value", "number");
  this.addProperty("value", 0, "number");
  this.widget = this.addWidget(
    "number",
    "value",
    this.properties.value,
    (v) => {
      this.properties.value = v;
      this.updateSize();
    }
  );
  this.size = [200, 60];
}
NumberNode.title = "Number";
NumberNode.prototype.onExecute = function () {
  this.setOutputData(0, Number(this.properties.value) || 0);
};

// Register it under a menu path (creates the right-click sections)
LiteGraph.registerNodeType("MELT/Inputs/Number", NumberNode);
