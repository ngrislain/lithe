/-
  Lithe/Viz.lean — DOT graph generation and HTML visualization
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Ops
import Lithe.Tensor

namespace Lithe.Viz

/-- A node in the visualization DAG. -/
structure VizNode where
  id    : Nat
  label : String
  shape : Shape
  color : String  -- category color
  deriving Repr

/-- An edge in the DAG. -/
structure VizEdge where
  from_ : Nat
  to    : Nat
  deriving Repr

/-- Collected graph for visualization. -/
structure VizGraph where
  nodes : Array VizNode
  edges : Array VizEdge
  deriving Repr

/-- State for graph construction (simple counter-based). -/
structure BuildState where
  nextId : Nat := 0
  nodes  : Array VizNode := #[]
  edges  : Array VizEdge := #[]

private def categoryColor : String → String
  | "data"      => "#4A90D9"  -- blue
  | "elemwise"  => "#7BC67E"  -- green
  | "shape"     => "#F5A623"  -- orange
  | "reduce"    => "#D94A4A"  -- red
  | "contract"  => "#9B59B6"  -- purple
  | _           => "#999999"  -- gray

/-- Build visualization graph from a TensorExpr. Returns (nodeId, state). -/
partial def buildGraph (st : BuildState) : TensorExpr Float s → (Nat × BuildState)
  | .literal s _ =>
    let id := st.nextId
    let node := { id, label := "literal", shape := s, color := categoryColor "data" }
    (id, { st with nextId := id + 1, nodes := st.nodes.push node })
  | .fill s v =>
    let id := st.nextId
    let node := { id, label := s!"fill({repr v})", shape := s, color := categoryColor "data" }
    (id, { st with nextId := id + 1, nodes := st.nodes.push node })
  | .var name s =>
    let id := st.nextId
    let node := { id, label := s!"var({name})", shape := s, color := categoryColor "data" }
    (id, { st with nextId := id + 1, nodes := st.nodes.push node })
  | .unary op e =>
    let (childId, st') := buildGraph st e
    let id := st'.nextId
    let node := { id, label := op.name, shape := s, color := categoryColor "elemwise" }
    let edge := { from_ := childId, to := id }
    (id, { st' with nextId := id + 1, nodes := st'.nodes.push node,
                     edges := st'.edges.push edge })
  | .binary op e₁ e₂ =>
    let (id1, st1) := buildGraph st e₁
    let (id2, st2) := buildGraph st1 e₂
    let id := st2.nextId
    let node := { id, label := op.name, shape := s, color := categoryColor "elemwise" }
    let e1 := { from_ := id1, to := id }
    let e2 := { from_ := id2, to := id }
    (id, { st2 with nextId := id + 1, nodes := st2.nodes.push node,
                     edges := st2.edges.push e1 |>.push e2 })
  | .smul c e =>
    let (childId, st') := buildGraph st e
    let id := st'.nextId
    let node := { id, label := s!"smul({repr c})", shape := s, color := categoryColor "elemwise" }
    let edge := { from_ := childId, to := id }
    (id, { st' with nextId := id + 1, nodes := st'.nodes.push node,
                     edges := st'.edges.push edge })
  | .select c t f =>
    let (cId, st1) := buildGraph st c
    let (tId, st2) := buildGraph st1 t
    let (fId, st3) := buildGraph st2 f
    let id := st3.nextId
    let node := { id, label := "select", shape := s, color := categoryColor "elemwise" }
    (id, { st3 with nextId := id + 1, nodes := st3.nodes.push node,
                     edges := st3.edges.push { from_ := cId, to := id }
                       |>.push { from_ := tId, to := id }
                       |>.push { from_ := fId, to := id } })
  | .reshape e _ =>
    let (childId, st') := buildGraph st e
    let id := st'.nextId
    let node := { id, label := "reshape", shape := s, color := categoryColor "shape" }
    (id, { st' with nextId := id + 1, nodes := st'.nodes.push node,
                     edges := st'.edges.push { from_ := childId, to := id } })
  | .transpose e _ =>
    let (childId, st') := buildGraph st e
    let id := st'.nextId
    let node := { id, label := "transpose", shape := s, color := categoryColor "shape" }
    (id, { st' with nextId := id + 1, nodes := st'.nodes.push node,
                     edges := st'.edges.push { from_ := childId, to := id } })
  | .broadcast e _ _ =>
    let (childId, st') := buildGraph st e
    let id := st'.nextId
    let node := { id, label := "broadcast", shape := s, color := categoryColor "shape" }
    (id, { st' with nextId := id + 1, nodes := st'.nodes.push node,
                     edges := st'.edges.push { from_ := childId, to := id } })
  | .slice e _ _ _ =>
    let (childId, st') := buildGraph st e
    let id := st'.nextId
    let node := { id, label := "slice", shape := s, color := categoryColor "shape" }
    (id, { st' with nextId := id + 1, nodes := st'.nodes.push node,
                     edges := st'.edges.push { from_ := childId, to := id } })
  | .pad e _ _ _ =>
    let (childId, st') := buildGraph st e
    let id := st'.nextId
    let node := { id, label := "pad", shape := s, color := categoryColor "shape" }
    (id, { st' with nextId := id + 1, nodes := st'.nodes.push node,
                     edges := st'.edges.push { from_ := childId, to := id } })
  | .concat e₁ e₂ _ _ =>
    let (id1, st1) := buildGraph st e₁
    let (id2, st2) := buildGraph st1 e₂
    let id := st2.nextId
    let node := { id, label := "concat", shape := s, color := categoryColor "shape" }
    (id, { st2 with nextId := id + 1, nodes := st2.nodes.push node,
                     edges := st2.edges.push { from_ := id1, to := id }
                       |>.push { from_ := id2, to := id } })
  | .gather e _ _ =>
    let (childId, st') := buildGraph st e
    let id := st'.nextId
    let node := { id, label := "gather", shape := s, color := categoryColor "shape" }
    (id, { st' with nextId := id + 1, nodes := st'.nodes.push node,
                     edges := st'.edges.push { from_ := childId, to := id } })
  | .reduce op _ e =>
    let (childId, st') := buildGraph st e
    let id := st'.nextId
    let node := { id, label := s!"reduce_{op.name}", shape := s, color := categoryColor "reduce" }
    (id, { st' with nextId := id + 1, nodes := st'.nodes.push node,
                     edges := st'.edges.push { from_ := childId, to := id } })
  | .scan op _ e =>
    let (childId, st') := buildGraph st e
    let id := st'.nextId
    let node := { id, label := s!"scan_{op.name}", shape := s, color := categoryColor "reduce" }
    (id, { st' with nextId := id + 1, nodes := st'.nodes.push node,
                     edges := st'.edges.push { from_ := childId, to := id } })
  | .einsum _ _ _ eA eB _ =>
    let (idA, st1) := buildGraph st eA
    let (idB, st2) := buildGraph st1 eB
    let id := st2.nextId
    let node := { id, label := "einsum", shape := s, color := categoryColor "contract" }
    (id, { st2 with nextId := id + 1, nodes := st2.nodes.push node,
                     edges := st2.edges.push { from_ := idA, to := id }
                       |>.push { from_ := idB, to := id } })

/-- Render shape as string. -/
private def shapeStr (s : Shape) : String :=
  "[" ++ ",".intercalate (s.map toString) ++ "]"

/-- Convert a TensorExpr to DOT format. -/
def toDot (e : TensorExpr Float s) : String :=
  let (_, st) := buildGraph {} e
  let header := "digraph TensorExpr {\n  rankdir=TB;\n  node [style=filled, fontname=\"Helvetica\"];\n"
  let nodes := st.nodes.foldl (fun acc n =>
    acc ++ s!"  n{n.id} [label=\"{n.label}\\n{shapeStr n.shape}\", fillcolor=\"{n.color}\", fontcolor=white];\n"
  ) ""
  let edges := st.edges.foldl (fun acc e =>
    acc ++ s!"  n{e.from_} -> n{e.to};\n"
  ) ""
  header ++ nodes ++ edges ++ "}\n"

/-- Generate self-contained HTML with embedded DOT rendering. -/
def toHTML (e : TensorExpr Float s) : String :=
  let dot := toDot e
  let dotEscaped := dot.replace "\\" "\\\\" |>.replace "`" "\\`" |>.replace "$" "\\$"
  s!"<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\">
<title>Lithe — Tensor Expression Graph</title>
<style>
  body \{ font-family: Helvetica, Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
  h1 \{ color: #4A90D9; }
  #graph \{ background: #16213e; border-radius: 8px; padding: 20px; }
  .legend \{ display: flex; gap: 20px; margin: 10px 0; }
  .legend-item \{ display: flex; align-items: center; gap: 6px; }
  .legend-color \{ width: 16px; height: 16px; border-radius: 3px; }
</style>
</head>
<body>
<h1>Lithe — Tensor Expression Graph</h1>
<div class=\"legend\">
  <div class=\"legend-item\"><div class=\"legend-color\" style=\"background:#4A90D9\"></div>Data</div>
  <div class=\"legend-item\"><div class=\"legend-color\" style=\"background:#7BC67E\"></div>Elementwise</div>
  <div class=\"legend-item\"><div class=\"legend-color\" style=\"background:#F5A623\"></div>Shape</div>
  <div class=\"legend-item\"><div class=\"legend-color\" style=\"background:#D94A4A\"></div>Reduction</div>
  <div class=\"legend-item\"><div class=\"legend-color\" style=\"background:#9B59B6\"></div>Contraction</div>
</div>
<div id=\"graph\"></div>
<script src=\"https://cdn.jsdelivr.net/npm/viz.js@2.1.2/viz.js\"></script>
<script src=\"https://cdn.jsdelivr.net/npm/viz.js@2.1.2/full.render.js\"></script>
<script>
const dot = `{dotEscaped}`;
const viz = new Viz();
viz.renderSVGElement(dot).then(svg => \{
  document.getElementById('graph').appendChild(svg);
}).catch(err => \{
  document.getElementById('graph').innerHTML = '<pre>' + err + '</pre>';
});
</script>
</body>
</html>"

end Lithe.Viz

namespace TensorExpr

/-- Generate DOT graph representation. -/
def toDot (e : TensorExpr Float s) : String := Lithe.Viz.toDot e

/-- Generate self-contained HTML visualization. -/
def toHTML (e : TensorExpr Float s) : String := Lithe.Viz.toHTML e

end TensorExpr
