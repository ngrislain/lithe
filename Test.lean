/-
  Test.lean — Test runner for Lithe
-/
import Lithe

open TensorExpr Tensor

/-! ### Test framework -/

structure TestResult where
  name    : String
  passed  : Bool
  message : String

def mkFail (name msg : String) : TestResult :=
  { name := name, passed := false, message := msg }

def mkPass (name : String) : TestResult :=
  { name := name, passed := true, message := "OK" }

def assertEq [BEq α] [Repr α] (name : String) (actual expected : α) : IO TestResult :=
  if actual == expected then
    pure (mkPass name)
  else
    let msg := "expected " ++ toString (repr expected) ++ ", got " ++ toString (repr actual)
    pure (mkFail name msg)

def assertClose (name : String) (actual expected : Float) (tol : Float := 1e-6) : IO TestResult :=
  if (actual - expected).abs < tol then
    pure (mkPass name)
  else
    let msg := "expected " ++ toString expected ++ ", got " ++ toString actual
    pure (mkFail name msg)

def assertListClose (name : String) (actual expected : List Float) (tol : Float := 1e-4)
    : IO TestResult :=
  if actual.length != expected.length then
    let msg := "length mismatch: " ++ toString actual.length ++ " vs " ++ toString expected.length
    pure (mkFail name msg)
  else
    let allClose := (actual.zip expected).all fun p => (p.1 - p.2).abs < tol
    if allClose then
      pure (mkPass name)
    else
      let msg := "values differ: got " ++ toString (repr actual) ++ ", expected " ++ toString (repr expected)
      pure (mkFail name msg)

private def stringContains (haystack needle : String) : Bool :=
  (haystack.splitOn needle).length > 1

def assertContains (name : String) (haystack needle : String) : IO TestResult :=
  if stringContains haystack needle then
    pure (mkPass name)
  else
    let msg := "string does not contain \"" ++ needle ++ "\""
    pure (mkFail name msg)

def assertTrue (name : String) (cond : Bool) (msg : String := "condition was false")
    : IO TestResult :=
  if cond then pure (mkPass name)
  else pure (mkFail name msg)

/-! ### Eval tests -/

def evalTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- Elementwise add
  let a : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let b : TensorExpr Float [2, 3] := .literal [2, 3] #v[7, 8, 9, 10, 11, 12]
  results := results ++ [← assertListClose "eval: a + b"
    (a + b).eval.toList [8, 10, 12, 14, 16, 18]]

  -- Elementwise mul
  results := results ++ [← assertListClose "eval: a * b"
    (a * b).eval.toList [7, 16, 27, 40, 55, 72]]

  -- Elementwise sub
  results := results ++ [← assertListClose "eval: a - b"
    (a - b).eval.toList [-6, -6, -6, -6, -6, -6]]

  -- Neg
  results := results ++ [← assertListClose "eval: -a"
    (-a).eval.toList [-1, -2, -3, -4, -5, -6]]

  -- Unary: relu
  let x : TensorExpr Float [4] := .literal [4] #v[-1.0, 0.0, 1.0, 2.0]
  results := results ++ [← assertListClose "eval: relu"
    (Tensor.relu x).eval.toList [0.0, 0.0, 1.0, 2.0]]

  -- Unary: abs
  results := results ++ [← assertListClose "eval: abs"
    (Tensor.abs x).eval.toList [1.0, 0.0, 1.0, 2.0]]

  -- Unary: sigmoid
  let sig := (Tensor.sigmoid x).eval.toList
  results := results ++ [← assertClose "eval: sigmoid[0]" (sig.getD 0 0) 0.2689 0.001]
  results := results ++ [← assertClose "eval: sigmoid[1]" (sig.getD 1 0) 0.5 0.001]
  results := results ++ [← assertClose "eval: sigmoid[2]" (sig.getD 2 0) 0.7311 0.001]

  -- Matmul via einsum: [2,3] @ [3,2]
  let m1 : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let m2 : TensorExpr Float [3, 2] := .literal [3, 2] #v[1, 0, 0, 1, 1, 1]
  let prod := Tensor.matmul m1 m2
  results := results ++ [← assertListClose "eval: matmul [2,3]@[3,2]"
    prod.eval.toList [4, 5, 10, 11]]

  -- Reshape
  let flat := TensorExpr.reshape (s₂ := [6]) a (by native_decide)
  results := results ++ [← assertListClose "eval: reshape [2,3]->[6]"
    flat.eval.toList [1, 2, 3, 4, 5, 6]]

  -- Reduce sum axis=0: [2,3] -> [3]
  let sumAxis0 := TensorExpr.reduce .sum ⟨0, by show 0 < 2; omega⟩ a
  results := results ++ [← assertListClose "eval: reduce sum axis=0"
    sumAxis0.eval.toList [5, 7, 9]]

  -- Reduce sum axis=1: [2,3] -> [2]
  let sumAxis1 := TensorExpr.reduce .sum ⟨1, by show 1 < 2; omega⟩ a
  results := results ++ [← assertListClose "eval: reduce sum axis=1"
    sumAxis1.eval.toList [6, 15]]

  -- Scan (cumsum)
  let v5 : TensorExpr Float [5] := .literal [5] #v[1, 2, 3, 4, 5]
  let cs := Tensor.cumsum ⟨0, by show 0 < 1; omega⟩ v5
  results := results ++ [← assertListClose "eval: cumsum"
    cs.eval.toList [1, 3, 6, 10, 15]]

  -- Dot product
  let da : TensorExpr Float [3] := .literal [3] #v[1, 2, 3]
  let db : TensorExpr Float [3] := .literal [3] #v[4, 5, 6]
  let dotResult := Tensor.dot da db
  results := results ++ [← assertListClose "eval: dot product"
    dotResult.eval.toList [32]]

  return results

/-! ### Env tests -/

def envTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  let wExpr := TensorExpr.var "w" [3, 2]
  let xExpr := TensorExpr.var "x" [2, 3]
  let yExpr := Tensor.matmul xExpr wExpr

  let env : Env Float := [
    ("w", ⟨[3, 2], #v[1, 0, 0, 1, 1, 1]⟩),
    ("x", ⟨[2, 3], #v[1, 2, 3, 4, 5, 6]⟩)
  ]

  match yExpr.evalWith env with
  | .ok result =>
    results := results ++ [← assertListClose "env: matmul via env"
      result.toList [4, 5, 10, 11]]
  | .error e =>
    results := results ++ [mkFail "env: matmul via env" e]

  -- evalWith returns error for missing var
  let missing := TensorExpr.var "missing" [2]
  match missing.evalWith env with
  | .ok _ =>
    results := results ++ [mkFail "env: missing var error" "expected error for missing var"]
  | .error _ =>
    results := results ++ [mkPass "env: missing var error"]

  -- evalWith returns error for shape mismatch
  let wrongShape := TensorExpr.var "w" [5]
  match wrongShape.evalWith env with
  | .ok _ =>
    results := results ++ [mkFail "env: shape mismatch error" "expected error for shape mismatch"]
  | .error _ =>
    results := results ++ [mkPass "env: shape mismatch error"]

  return results

/-! ### Autodiff tests -/

def autodiffTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- grad of dot(x, w) w.r.t. w should mention "w"
  let wVar : TensorExpr Float [2] := .var "w" [2]
  let xLit : TensorExpr Float [2] := .literal [2] #v[3.0, 4.0]
  let loss := Tensor.dot xLit wVar
  let grads := loss.grad

  let hasW := grads.any fun p => p.1 == "w"
  results := results ++ [← assertTrue "autodiff: grad has 'w'" hasW
    "gradient map does not contain variable 'w'"]

  -- Check gradient shape for w is [2]
  match grads.find? fun p => p.1 == "w" with
  | some (_, ⟨shape, _⟩) =>
    results := results ++ [← assertEq "autodiff: grad 'w' shape" shape [2]]
  | none =>
    results := results ++ [mkFail "autodiff: grad 'w' shape" "variable 'w' not in grads"]

  -- backward of (a + b) w.r.t. both a, b (using backward directly, not through reduce)
  let aVar : TensorExpr Float [3] := .var "a" [3]
  let bVar : TensorExpr Float [3] := .var "b" [3]
  let addExpr := aVar + bVar
  let grads2 := addExpr.backward (.fill [3] 1.0)
  let hasA := grads2.any fun p => p.1 == "a"
  let hasB := grads2.any fun p => p.1 == "b"
  results := results ++ [← assertTrue "autodiff: add backward has 'a'" hasA]
  results := results ++ [← assertTrue "autodiff: add backward has 'b'" hasB]

  return results

/-! ### Viz tests -/

def vizTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  let m1 : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let m2 : TensorExpr Float [3, 2] := .literal [3, 2] #v[1, 0, 0, 1, 1, 1]
  let expr := Tensor.relu (Tensor.matmul m1 m2)

  let dot := expr.toDot
  results := results ++ [← assertContains "viz: toDot has 'digraph'" dot "digraph"]
  results := results ++ [← assertContains "viz: toDot has 'fillcolor'" dot "fillcolor"]
  results := results ++ [← assertContains "viz: toDot has node refs" dot "n0"]

  let html := expr.toHTML
  results := results ++ [← assertContains "viz: toHTML has html tag" html "<html"]
  results := results ++ [← assertContains "viz: toHTML has DOT" html "digraph"]

  return results

/-! ### CPU Backend tests -/

def cpuBackendTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- Literal + binary add
  let a : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let b : TensorExpr Float [2, 3] := .literal [2, 3] #v[7, 8, 9, 10, 11, 12]
  let plan := Lithe.Backend.TensorExpr.toExecPlan (a + b)
  match plan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu: literal + binary add"
      result.toList [8, 10, 12, 14, 16, 18]]
  | .error e =>
    results := results ++ [mkFail "cpu: literal + binary add" e]

  -- executeIO variant
  let result ← plan.executeIO
  results := results ++ [← assertListClose "cpu: executeIO"
    result.toList [8, 10, 12, 14, 16, 18]]

  -- Unary neg
  let negPlan := Lithe.Backend.TensorExpr.toExecPlan (-a)
  match negPlan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu: unary neg"
      result.toList [-1, -2, -3, -4, -5, -6]]
  | .error e =>
    results := results ++ [mkFail "cpu: unary neg" e]

  -- Variable resolution via env
  let varExpr := TensorExpr.var "x" [3]
  let varPlan := Lithe.Backend.TensorExpr.toExecPlan varExpr
  let env : Env Float := [("x", ⟨[3], #v[10, 20, 30]⟩)]
  match varPlan.execute env with
  | .ok result =>
    results := results ++ [← assertListClose "cpu: var resolution"
      result.toList [10, 20, 30]]
  | .error e =>
    results := results ++ [mkFail "cpu: var resolution" e]

  -- Missing variable returns error
  match varPlan.execute with
  | .ok _ =>
    results := results ++ [mkFail "cpu: missing var error" "expected error"]
  | .error _ =>
    results := results ++ [mkPass "cpu: missing var error"]

  return results

/-! ### Codegen tests -/

def codegenTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  let a : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let b : TensorExpr Float [2, 3] := .literal [2, 3] #v[7, 8, 9, 10, 11, 12]
  let expr := a + b

  -- CUDA codegen
  let cuda := expr.toCUDA
  results := results ++ [← assertContains "codegen: CUDA has __global__" cuda "__global__"]
  results := results ++ [← assertContains "codegen: CUDA has cuda_runtime" cuda "cuda_runtime"]
  results := results ++ [← assertContains "codegen: CUDA has kernel fn" cuda "kernel_"]

  -- WebGPU HTML codegen (use elementwise ops that generate actual kernels)
  let gpuExpr := a + b
  let gpuEnv : Env Float := []
  let gpuHtml := gpuExpr.toWebGPUHTML gpuEnv
  results := results ++ [← assertContains "codegen: WGSL has @compute" gpuHtml "@compute"]
  results := results ++ [← assertContains "codegen: WGSL has @workgroup_size" gpuHtml "@workgroup_size"]
  results := results ++ [← assertContains "codegen: WebGPU HTML has script" gpuHtml "<script>"]

  return results

/-! ### Module tests -/

def moduleTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- Linear layer params
  let layer := Module.linear 3 2 "fc"
  let paramNames := layer.params.map Prod.fst
  results := results ++ [← assertTrue "module: linear has weight param"
    (paramNames.contains "fc.weight")]
  results := results ++ [← assertTrue "module: linear has bias param"
    (paramNames.contains "fc.bias")]

  -- Linear layer forward
  let env : Env Float := [
    ("fc.weight", ⟨[3, 2], #v[1, 0, 0, 1, 1, 1]⟩),
    ("fc.bias",   ⟨[2],    #v[0.1, 0.2]⟩)
  ]
  let input : Vector Float 3 := #v[1, 2, 3]
  let inputExpr := TensorExpr.literal [3] input
  match (layer.forward inputExpr).evalWith env with
  | .ok result =>
    results := results ++ [← assertListClose "module: linear forward"
      result.toList [4.1, 5.2]]
  | .error e =>
    results := results ++ [mkFail "module: linear forward" e]

  -- MLP params
  let net := Module.mlp 4 3 2 "net"
  let netParams := net.params.map Prod.fst
  results := results ++ [← assertTrue "module: mlp has layer1.weight"
    (netParams.contains "net.layer1.weight")]
  results := results ++ [← assertTrue "module: mlp has layer2.bias"
    (netParams.contains "net.layer2.bias")]

  return results

/-! ### Runner -/

def main : IO Unit := do
  IO.println "=== Lithe Test Suite ==="
  IO.println ""

  let suites : List (String × IO (List TestResult)) := [
    ("Eval",        evalTests),
    ("Env",         envTests),
    ("Autodiff",    autodiffTests),
    ("Viz",         vizTests),
    ("CPU Backend", cpuBackendTests),
    ("Codegen",     codegenTests),
    ("Module",      moduleTests)
  ]

  let mut totalPassed := 0
  let mut totalFailed := 0

  for (suiteName, runSuite) in suites do
    IO.println ("--- " ++ suiteName ++ " ---")
    let results ← runSuite
    for r in results do
      if r.passed then
        IO.println ("  PASS  " ++ r.name)
        totalPassed := totalPassed + 1
      else
        IO.println ("  FAIL  " ++ r.name ++ ": " ++ r.message)
        totalFailed := totalFailed + 1
    IO.println ""

  IO.println "========================"
  IO.println ("  " ++ toString totalPassed ++ " passed, " ++ toString totalFailed ++ " failed")
  if totalFailed > 0 then
    IO.println "  SOME TESTS FAILED"
    IO.Process.exit 1
  else
    IO.println "  ALL TESTS PASSED"
