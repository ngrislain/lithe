/-
  Lithe/Viz.lean — DOT graph generation and HTML visualization
  Tableau-colored, HTML-label nodes, modern viz-js rendering.
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Ops
import Lithe.Tensor

namespace Lithe.Viz

-- ── Tableau color palette ────────────────────────────────────────────

private def colorConstant  := "#76B7B2"  -- Teal
private def colorVariable  := "#BAB0AC"  -- Grey
private def colorAdd       := "#F28E2B"  -- Orange
private def colorMul       := "#E15759"  -- Red
private def colorMaxMin    := "#FF9DA7"  -- Pink
private def colorUnary     := "#59A14F"  -- Green
private def colorSelect    := "#9C755F"  -- Brown
private def colorShape     := "#B07AA1"  -- Purple
private def colorReduceSum := "#EDC948"  -- Yellow
private def colorReduceProd:= "#FF9DA7"  -- Pink
private def colorReduceMM  := "#F28E2B"  -- Orange
private def colorScan      := "#59A14F"  -- Green
private def colorEinsum    := "#4E79A7"  -- Blue
private def colorDark      := "#555555"

-- ── Helper functions ─────────────────────────────────────────────────

private def escapeHtml (s : String) : String :=
  s |>.replace "&" "&amp;"
    |>.replace "<" "&lt;"
    |>.replace ">" "&gt;"
    |>.replace "\"" "&quot;"
    |>.replace "'" "&#39;"

private def hexDigit (c : Char) : Option UInt8 :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - '0'.toNat).toUInt8
  else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 'a'.toNat + 10).toUInt8
  else if 'A' ≤ c && c ≤ 'F' then some (c.toNat - 'A'.toNat + 10).toUInt8
  else none

private def parseHexColor (hex : String) : Option (UInt8 × UInt8 × UInt8) := do
  let s := if hex.startsWith "#" then (hex.toList.drop 1) else hex.toList
  if s.length != 6 then none
  else
    let r1 ← hexDigit (List.getD s 0 '0')
    let r0 ← hexDigit (List.getD s 1 '0')
    let g1 ← hexDigit (List.getD s 2 '0')
    let g0 ← hexDigit (List.getD s 3 '0')
    let b1 ← hexDigit (List.getD s 4 '0')
    let b0 ← hexDigit (List.getD s 5 '0')
    some (r1 * 16 + r0, g1 * 16 + g0, b1 * 16 + b0)

private def fontColorForBg (hex : String) : String :=
  match parseHexColor hex with
  | some (r, g, b) =>
    let brightness := 0.299 * r.toFloat + 0.587 * g.toFloat + 0.114 * b.toFloat
    if brightness > 140.0 then "black" else "white"
  | none => "black"

private def toHex2 (n : UInt8) : String :=
  let hi := n / 16
  let lo := n % 16
  let hexChar (v : UInt8) : Char :=
    if v < 10 then Char.ofNat (v.toNat + '0'.toNat)
    else Char.ofNat (v.toNat - 10 + 'A'.toNat)
  String.ofList [hexChar hi, hexChar lo]

private def darkenColor (hex : String) : String :=
  match parseHexColor hex with
  | some (r, g, b) =>
    let d (c : UInt8) : UInt8 := (c.toFloat * 0.7).toUInt8
    "#" ++ toHex2 (d r) ++ toHex2 (d g) ++ toHex2 (d b)
  | none => colorDark

private def shapeStr (s : Shape) : String :=
  if s.isEmpty then "scalar"
  else " × ".intercalate (s.map toString)

-- ── Per-constructor color + symbol ───────────────────────────────────

private def binaryInfo : BinaryOp → String × String
  | .add => ("+", colorAdd)
  | .sub => ("−", colorAdd)
  | .mul => ("×", colorMul)
  | .div => ("÷", colorMul)
  | .pow => ("^", colorMul)
  | .max => ("max", colorMaxMin)
  | .min => ("min", colorMaxMin)

private def reduceInfo : ReduceOp → String × String
  | .sum  => ("∑", colorReduceSum)
  | .prod => ("∏", colorReduceProd)
  | .max  => ("reduce_max", colorReduceMM)
  | .min  => ("reduce_min", colorReduceMM)

-- ── Build state ──────────────────────────────────────────────────────

private structure BuildState where
  nextId : Nat := 0
  lines  : Array String := #[]

-- ── Node emitter ─────────────────────────────────────────────────────

/-- Emit a DOT node with HTML label and return (nodeId, updatedState). -/
private def emitNode (st : BuildState) (symbol : String) (info : String)
    (shape : Shape) (fill : String) : Nat × BuildState :=
  let id := st.nextId
  let fc := fontColorForBg fill
  let bc := darkenColor fill
  let shp := escapeHtml (shapeStr shape)
  let infoLine := if info.isEmpty then "" else escapeHtml info ++ "<BR/>"
  let label := "<B>" ++ escapeHtml symbol ++ "</B><BR/>" ++ infoLine ++ shp
  let line := "  n" ++ toString id
    ++ " [label=<" ++ label ++ ">"
    ++ ", style=filled"
    ++ ", fillcolor=\"" ++ fill ++ "\""
    ++ ", color=\"" ++ bc ++ "\""
    ++ ", fontcolor=\"" ++ fc ++ "\""
    ++ "];"
  (id, { nextId := id + 1, lines := st.lines.push line })

/-- Emit an edge line. -/
private def emitEdge (st : BuildState) (fromId toId : Nat) : BuildState :=
  { st with lines := st.lines.push ("  n" ++ toString fromId ++ " -> n" ++ toString toId ++ ";") }

-- ── Recursive graph builder ──────────────────────────────────────────

/-- Recursively traverse a `TensorExpr` and emit DOT graph nodes and edges.
    Returns the root node ID and the accumulated build state. -/
partial def buildGraph (st : BuildState) : TensorExpr Float s → (Nat × BuildState)
  | .literal s _ =>
    emitNode st "Const" "" s colorConstant
  | .fill s v =>
    emitNode st "Fill" (toString (repr v)) s colorConstant
  | .var name s =>
    emitNode st name "" s colorVariable
  | .unary op e =>
    let (cId, st') := buildGraph st e
    let (id, st'') := emitNode st' op.name "" s colorUnary
    (id, emitEdge st'' cId id)
  | .binary op e₁ e₂ =>
    let (id1, st1) := buildGraph st e₁
    let (id2, st2) := buildGraph st1 e₂
    let (sym, col) := binaryInfo op
    let (id, st3) := emitNode st2 sym "" s col
    (id, emitEdge (emitEdge st3 id1 id) id2 id)
  | .smul c e =>
    let (cId, st') := buildGraph st e
    let (id, st'') := emitNode st' ("×" ++ toString (repr c)) "" s colorMul
    (id, emitEdge st'' cId id)
  | .select c t f =>
    let (cId, st1) := buildGraph st c
    let (tId, st2) := buildGraph st1 t
    let (fId, st3) := buildGraph st2 f
    let (id, st4) := emitNode st3 "select" "" s colorSelect
    (id, emitEdge (emitEdge (emitEdge st4 cId id) tId id) fId id)
  | .reshape e _ =>
    let (cId, st') := buildGraph st e
    let (id, st'') := emitNode st' "reshape" "" s colorShape
    (id, emitEdge st'' cId id)
  | .transpose e _ =>
    let (cId, st') := buildGraph st e
    let (id, st'') := emitNode st' "transpose" "" s colorShape
    (id, emitEdge st'' cId id)
  | .broadcast e _ _ =>
    let (cId, st') := buildGraph st e
    let (id, st'') := emitNode st' "broadcast" "" s colorShape
    (id, emitEdge st'' cId id)
  | .slice e _ _ _ =>
    let (cId, st') := buildGraph st e
    let (id, st'') := emitNode st' "slice" "" s colorShape
    (id, emitEdge st'' cId id)
  | .pad e _ _ _ =>
    let (cId, st') := buildGraph st e
    let (id, st'') := emitNode st' "pad" "" s colorShape
    (id, emitEdge st'' cId id)
  | .concat e₁ e₂ _ _ =>
    let (id1, st1) := buildGraph st e₁
    let (id2, st2) := buildGraph st1 e₂
    let (id, st3) := emitNode st2 "concat" "" s colorShape
    (id, emitEdge (emitEdge st3 id1 id) id2 id)
  | .gather e _ _ =>
    let (cId, st') := buildGraph st e
    let (id, st'') := emitNode st' "gather" "" s colorShape
    (id, emitEdge st'' cId id)
  | .reduce op ax e =>
    let (cId, st') := buildGraph st e
    let (sym, col) := reduceInfo op
    let (id, st'') := emitNode st' sym ("axis " ++ toString ax.val) s col
    (id, emitEdge st'' cId id)
  | .scan op ax e =>
    let (cId, st') := buildGraph st e
    let (id, st'') := emitNode st' ("scan_" ++ op.name) ("axis " ++ toString ax.val) s colorScan
    (id, emitEdge st'' cId id)
  | .einsum subsA subsB subsOut eA eB _ =>
    let (idA, st1) := buildGraph st eA
    let (idB, st2) := buildGraph st1 eB
    let fmtSubs (l : List Nat) : String :=
      if l.isEmpty then "∅" else " ".intercalate (l.map toString)
    let formula := fmtSubs subsA ++ " , " ++ fmtSubs subsB ++ " → " ++ fmtSubs subsOut
    let (id, st3) := emitNode st2 "Einsum" formula s colorEinsum
    (id, emitEdge (emitEdge st3 idA id) idB id)

-- ── Public API ───────────────────────────────────────────────────────

/-- Convert a `TensorExpr` to a DOT-format directed graph string for Graphviz rendering.
    The resulting string can be passed to `dot` or any Graphviz-compatible tool. -/
def toDot (e : TensorExpr Float s) : String :=
  let (rootId, st) := buildGraph {} e
  let header := "digraph TensorExpr {\n"
    ++ "  rankdir=TD;\n"
    ++ "  node [shape=box, fontname=\"Helvetica\", fontsize=11];\n"
    ++ "  edge [fontname=\"Helvetica\", fontsize=10, color=\"" ++ colorDark ++ "\"];\n"
  let body := "\n".intercalate st.lines.toList
  let entry := "\n  entry [shape=circle, style=filled, fillcolor=\"" ++ colorDark
    ++ "\", fontcolor=\"" ++ colorDark ++ "\", label=\"\", width=0.2, height=0.2];\n"
    ++ "  entry -> n" ++ toString rootId ++ ";\n"
    ++ "  n" ++ toString rootId ++ " [penwidth=2];\n"
  header ++ body ++ entry ++ "}\n"

/-- Generate a self-contained HTML page with embedded DOT rendering via viz-js CDN.
    The page auto-refreshes every 60 seconds and uses modern SVG rendering. -/
def toHTML (e : TensorExpr Float s) : String :=
  let dot := toDot e
  "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
  ++ "  <meta charset=\"utf-8\" />\n"
  ++ "  <meta http-equiv=\"refresh\" content=\"60\">\n"
  ++ "  <title>Lithe — Tensor Expression Graph</title>\n"
  ++ "  <style>\n"
  ++ "    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; background-color: #f8fafc; }\n"
  ++ "    #graph { min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 24px; box-sizing: border-box; }\n"
  ++ "    svg { max-width: 100%; height: auto; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15); border-radius: 8px; background: white; }\n"
  ++ "  </style>\n"
  ++ "</head>\n<body>\n"
  ++ "  <div id=\"graph\">Rendering graph...</div>\n"
  ++ "  <script id=\"tensor-dot\" type=\"text/plain\">\n"
  ++ dot
  ++ "  </script>\n"
  ++ "  <script type=\"module\">\n"
  ++ "    import { instance } from \"https://cdn.jsdelivr.net/npm/@viz-js/viz@3.20.0/dist/viz.js?module\";\n"
  ++ "    (async () => {\n"
  ++ "      const dot = document.getElementById(\"tensor-dot\").textContent.trim();\n"
  ++ "      const viz = await instance();\n"
  ++ "      const svg = viz.renderSVGElement(dot);\n"
  ++ "      const container = document.getElementById(\"graph\");\n"
  ++ "      container.innerHTML = \"\";\n"
  ++ "      container.appendChild(svg);\n"
  ++ "    })();\n"
  ++ "  </script>\n"
  ++ "</body>\n</html>"

end Lithe.Viz

namespace TensorExpr

/-- Generate DOT graph representation of this tensor expression.
    Delegates to `Lithe.Viz.toDot`. -/
def toDot (e : TensorExpr Float s) : String := Lithe.Viz.toDot e

/-- Generate self-contained HTML visualization of this tensor expression.
    Delegates to `Lithe.Viz.toHTML`. -/
def toHTML (e : TensorExpr Float s) : String := Lithe.Viz.toHTML e

end TensorExpr
