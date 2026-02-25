import Lithe

open TensorExpr Tensor

def main : IO Unit := do
  IO.println "=== Lithe v2 — Symbolic Tensor Library ==="
  IO.println ""

  -- 1. Basic elementwise ops
  IO.println "--- Elementwise Ops ---"
  let a : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let b : TensorExpr Float [2, 3] := .literal [2, 3] #v[7, 8, 9, 10, 11, 12]
  IO.println s!"a + b = {repr (a + b).eval.toList}"
  IO.println s!"a * b = {repr (a * b).eval.toList}"
  IO.println s!"a - b = {repr (a - b).eval.toList}"
  IO.println s!"-a    = {repr (-a).eval.toList}"
  IO.println ""

  -- 2. Unary ops
  IO.println "--- Unary Ops ---"
  let x : TensorExpr Float [4] := .literal [4] #v[-1.0, 0.0, 1.0, 2.0]
  IO.println s!"relu(x)    = {repr (Tensor.relu x).eval.toList}"
  IO.println s!"sigmoid(x) = {repr (Tensor.sigmoid x).eval.toList}"
  IO.println s!"abs(x)     = {repr (Tensor.abs x).eval.toList}"
  IO.println ""

  -- 3. Matmul via einsum
  IO.println "--- Matmul (via einsum) ---"
  let m1 : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let m2 : TensorExpr Float [3, 2] := .literal [3, 2] #v[1, 0, 0, 1, 1, 1]
  let prod := Tensor.matmul m1 m2
  IO.println s!"[2,3] @ [3,2] = {repr prod.eval.toList}"
  IO.println ""

  -- 4. Reshape
  IO.println "--- Reshape ---"
  let flat := TensorExpr.reshape (s₂ := [6]) a (by native_decide)
  IO.println s!"reshape [2,3] -> [6] = {repr flat.eval.toList}"
  IO.println ""

  -- 5. Reduce
  IO.println "--- Reduce ---"
  let sumAxis0 := TensorExpr.reduce .sum ⟨0, by show 0 < 2; omega⟩ a
  IO.println s!"sum(a, axis=0) = {repr sumAxis0.eval.toList}"
  let sumAxis1 := TensorExpr.reduce .sum ⟨1, by show 1 < 2; omega⟩ a
  IO.println s!"sum(a, axis=1) = {repr sumAxis1.eval.toList}"
  IO.println ""

  -- 6. Scan (cumulative sum)
  IO.println "--- Scan (cumsum) ---"
  let v5 : TensorExpr Float [5] := .literal [5] #v[1, 2, 3, 4, 5]
  let cs := Tensor.cumsum ⟨0, by show 0 < 1; omega⟩ v5
  IO.println s!"cumsum([1,2,3,4,5]) = {repr cs.eval.toList}"
  IO.println ""

  -- 7. Variables with Env
  IO.println "--- Variables & Env ---"
  let wExpr := TensorExpr.var "w" [3, 2]
  let xExpr := TensorExpr.var "x" [2, 3]
  let yExpr := Tensor.matmul xExpr wExpr
  let env : Env Float := [
    ("w", ⟨[3, 2], #v[1, 0, 0, 1, 1, 1]⟩),
    ("x", ⟨[2, 3], #v[1, 2, 3, 4, 5, 6]⟩)
  ]
  match yExpr.evalWith env with
  | .ok result => IO.println s!"x @ w (via env) = {repr result.toList}"
  | .error e => IO.println s!"Error: {e}"
  IO.println ""

  -- 8. Module (linear layer)
  IO.println "--- Module (Linear Layer) ---"
  let layer := Module.linear 3 2 "fc"
  IO.println s!"Linear layer params: {repr (layer.params.map (·.1))}"
  let moduleEnv : Env Float := [
    ("fc.weight", ⟨[3, 2], #v[1, 0, 0, 1, 1, 1]⟩),
    ("fc.bias",   ⟨[2],    #v[0.1, 0.2]⟩)
  ]
  let input : Vector Float 3 := #v[1, 2, 3]
  let inputExpr := TensorExpr.literal [3] input
  match (layer.forward inputExpr).evalWith moduleEnv with
  | .ok result => IO.println s!"linear([1,2,3]) = {repr result.toList}"
  | .error e => IO.println s!"Error: {e}"
  IO.println ""

  -- 9. Autodiff
  IO.println "--- Autodiff ---"
  let wVar : TensorExpr Float [2] := .var "w" [2]
  let xLit : TensorExpr Float [2] := .literal [2] #v[3.0, 4.0]
  let loss := Tensor.dot xLit wVar
  let grads := loss.grad
  IO.println s!"grad(dot(x, w)) w.r.t. vars:"
  for (name, ⟨shape, _⟩) in grads do
    IO.println s!"  {name}: shape={repr shape}"
  IO.println ""

  -- 10. Visualization — write HTML
  IO.println "--- Visualization ---"
  let vizExpr := Tensor.relu (Tensor.matmul m1 m2)
  let html := vizExpr.toHTML
  IO.FS.writeFile "output/graph.html" html
  IO.println "Wrote output/graph.html (expression DAG visualization)"
  IO.println ""

  -- 11. CPU Backend
  IO.println "--- CPU Backend ---"
  let simpleExpr := a + b
  let plan := Lithe.Backend.TensorExpr.toExecPlan simpleExpr
  IO.println s!"DAG nodes: {plan.nodes.size}, output: {plan.output}"
  let result ← plan.executeIO
  IO.println s!"CPU backend (a+b) = {repr result.toList}"
  IO.println ""

  -- 12. CUDA codegen
  IO.println "--- CUDA Codegen ---"
  let cudaSrc := simpleExpr.toCUDA
  IO.println s!"Generated CUDA source ({cudaSrc.length} chars)"
  IO.println ""

  -- 13. WebGPU HTML
  IO.println "--- WebGPU HTML ---"
  let gpuExpr := Tensor.matmul
    (TensorExpr.var "weights" [3, 2])
    (TensorExpr.var "input" [2, 1])
  let gpuEnv : Env Float := [
    ("weights", ⟨[3, 2], #v[1, 0, 0, 1, 1, 1]⟩),
    ("input",   ⟨[2, 1], #v[3, 4]⟩)
  ]
  let gpuHtml := gpuExpr.toWebGPUHTML gpuEnv
  IO.FS.writeFile "output/compute.html" gpuHtml
  IO.println "Wrote output/compute.html (WebGPU compute demo)"
  IO.println ""

  IO.println "=== Done ==="
