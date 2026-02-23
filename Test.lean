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

/-! ### Dim / DimShape tests -/

def dimTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- Dim construction
  let d1 := Dim.named "batch" 32
  results := results ++ [← assertEq "dim: named val" d1.val 32]
  results := results ++ [← assertEq "dim: named name" d1.name (some "batch")]

  -- Anonymous dim
  let d2 : Dim := ⟨16, none⟩
  results := results ++ [← assertEq "dim: anon val" d2.val 16]
  results := results ++ [← assertEq "dim: anon name" d2.name none]

  -- Compatibility: both named same → true
  results := results ++ [← assertTrue "dim: compatible same name"
    (Dim.compatible (Dim.named "x" 5) (Dim.named "x" 5))]

  -- Compatibility: both named different → false
  results := results ++ [← assertTrue "dim: incompatible diff name"
    (!(Dim.compatible (Dim.named "x" 5) (Dim.named "y" 5)))]

  -- Compatibility: one anonymous → true (values match)
  results := results ++ [← assertTrue "dim: compatible anon"
    (Dim.compatible (Dim.named "x" 5) ⟨5, none⟩)]

  -- DimShape.toShape
  let ds : DimShape := [Dim.named "batch" 2, ⟨3, none⟩]
  results := results ++ [← assertEq "dimshape: toShape" ds.toShape [2, 3]]

  -- DimShape.fromShape roundtrip
  let s : Shape := [4, 5]
  results := results ++ [← assertEq "dimshape: fromShape toShape" (DimShape.fromShape s).toShape s]

  -- DimShape.product
  results := results ++ [← assertEq "dimshape: product" ds.product 6]

  -- DimShape.findDimIdx
  results := results ++ [← assertEq "dimshape: findDimIdx found" (ds.findDimIdx "batch") (some 0)]
  results := results ++ [← assertEq "dimshape: findDimIdx missing" (ds.findDimIdx "seq") none]

  return results

/-! ### NamedTensor tests -/

def namedTensorTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- Arithmetic with matching dims
  let ds : DimShape := [Dim.named "row" 2, ⟨3, none⟩]
  let a := NamedTensor.literal ds (by simp [DimShape.toShape]; exact #v[1, 2, 3, 4, 5, 6])
  let b := NamedTensor.literal ds (by simp [DimShape.toShape]; exact #v[7, 8, 9, 10, 11, 12])
  let sum := a + b
  results := results ++ [← assertListClose "named: add"
    sum.eval.toList [8, 10, 12, 14, 16, 18]]

  let diff := a - b
  results := results ++ [← assertListClose "named: sub"
    diff.eval.toList [-6, -6, -6, -6, -6, -6]]

  let neg := -a
  results := results ++ [← assertListClose "named: neg"
    neg.eval.toList [-1, -2, -3, -4, -5, -6]]

  -- Zeros / fill
  let z := NamedTensor.zeros (α := Float) [Dim.named "x" 3]
  results := results ++ [← assertListClose "named: zeros"
    z.eval.toList [0, 0, 0]]

  -- Unary ops
  let x := NamedTensor.literal [⟨4, none⟩] (by simp [DimShape.toShape]; exact #v[-1.0, 0.0, 1.0, 2.0])
  let r := NamedTensor.relu x
  results := results ++ [← assertListClose "named: relu"
    r.eval.toList [0.0, 0.0, 1.0, 2.0]]

  -- Smul
  let scaled := NamedTensor.smul 2.0 a
  results := results ++ [← assertListClose "named: smul"
    scaled.eval.toList [2, 4, 6, 8, 10, 12]]

  return results

/-! ### Slicing tests (plain TensorExpr) -/

def slicingTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- sliceWith on [4, 3]: take rows 1..2, all cols
  let t : TensorExpr Float [4, 3] := .literal [4, 3] #v[1,2,3, 4,5,6, 7,8,9, 10,11,12]
  let s := Tensor.sliceWith t [(1, 2), (0, 3)]
  results := results ++ [← assertListClose "slice: sliceWith [4,3]->[(1,2),(0,3)]"
    s.eval.toList [4, 5, 6, 7, 8, 9]]

  -- head: select row 0 from [4, 3] → [3]
  let row0 := Tensor.head t 0
  results := results ++ [← assertListClose "slice: head row 0"
    row0.eval.toList [1, 2, 3]]

  -- head: select row 2 from [4, 3] → [3]
  let row2 := Tensor.head t 2
  results := results ++ [← assertListClose "slice: head row 2"
    row2.eval.toList [7, 8, 9]]

  return results

/-! ### Named slicing tests -/

def namedSlicingTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  let ds : DimShape := [Dim.named "batch" 4, Dim.named "feat" 3]
  let t := NamedTensor.literal ds (by simp [DimShape.toShape]; exact #v[1,2,3, 4,5,6, 7,8,9, 10,11,12])

  -- sliceAt: take batch 1..2
  match t.sliceAt "batch" 1 2 with
  | some ⟨_, t2⟩ =>
    results := results ++ [← assertListClose "named slice: sliceAt batch"
      t2.eval.toList [4, 5, 6, 7, 8, 9]]
  | none =>
    results := results ++ [mkFail "named slice: sliceAt batch" "returned none"]

  -- sliceAt: take feat 0..1
  match t.sliceAt "feat" 0 1 with
  | some ⟨_, t3⟩ =>
    results := results ++ [← assertListClose "named slice: sliceAt feat"
      t3.eval.toList [1, 4, 7, 10]]
  | none =>
    results := results ++ [mkFail "named slice: sliceAt feat" "returned none"]

  -- sliceAt: invalid dim name returns none
  results := results ++ [← assertTrue "named slice: sliceAt invalid"
    (t.sliceAt "missing" 0 1).isNone]

  -- reduceAt: sum over feat
  match t.reduceAt "feat" .sum with
  | some ⟨_, t4⟩ =>
    results := results ++ [← assertListClose "named slice: reduceAt feat sum"
      t4.eval.toList [6, 15, 24, 33]]
  | none =>
    results := results ++ [mkFail "named slice: reduceAt feat sum" "returned none"]

  -- reduceAt: sum over batch
  match t.reduceAt "batch" .sum with
  | some ⟨_, t5⟩ =>
    results := results ++ [← assertListClose "named slice: reduceAt batch sum"
      t5.eval.toList [22, 26, 30]]
  | none =>
    results := results ++ [mkFail "named slice: reduceAt batch sum" "returned none"]

  return results

/-! ### Einsum tests -/

def einsumTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- Parse correctness
  match Einsum.parse "ij,jk->ik" with
  | some (subsA, subsB, subsOut) =>
    results := results ++ [← assertEq "einsum: parse matmul subsA" subsA [0, 1]]
    results := results ++ [← assertEq "einsum: parse matmul subsB" subsB [1, 2]]
    results := results ++ [← assertEq "einsum: parse matmul subsOut" subsOut [0, 2]]
  | none =>
    results := results ++ [mkFail "einsum: parse matmul" "parse returned none"]

  match Einsum.parse "i,i->" with
  | some (subsA, subsB, subsOut) =>
    results := results ++ [← assertEq "einsum: parse dot subsA" subsA [0]]
    results := results ++ [← assertEq "einsum: parse dot subsB" subsB [0]]
    results := results ++ [← assertEq "einsum: parse dot subsOut" subsOut []]
  | none =>
    results := results ++ [mkFail "einsum: parse dot" "parse returned none"]

  -- End-to-end matmul via einsum string
  let a : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let b : TensorExpr Float [3, 2] := .literal [3, 2] #v[1, 0, 0, 1, 1, 1]
  match Einsum.ein "ij,jk->ik" a b with
  | some ⟨_, c⟩ =>
    results := results ++ [← assertListClose "einsum: matmul result"
      c.eval.toList [4, 5, 10, 11]]
  | none =>
    results := results ++ [mkFail "einsum: matmul result" "ein returned none"]

  -- Dot product via einsum string
  let x : TensorExpr Float [3] := .literal [3] #v[1, 2, 3]
  let y : TensorExpr Float [3] := .literal [3] #v[4, 5, 6]
  match Einsum.ein "i,i->" x y with
  | some ⟨_, d⟩ =>
    results := results ++ [← assertListClose "einsum: dot product"
      d.eval.toList [32]]
  | none =>
    results := results ++ [mkFail "einsum: dot product" "ein returned none"]

  -- Outer product via einsum string
  match Einsum.ein "i,j->ij" x y with
  | some ⟨_, e⟩ =>
    results := results ++ [← assertListClose "einsum: outer product"
      e.eval.toList [4, 5, 6, 8, 10, 12, 12, 15, 18]]
  | none =>
    results := results ++ [mkFail "einsum: outer product" "ein returned none"]

  -- Invalid spec returns none
  results := results ++ [← assertTrue "einsum: invalid spec"
    (Einsum.ein "abc" a b).isNone]

  return results

/-! ### Module pipeline tests -/

open Module in
def modulePipelineTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- Pipeline operator produces same params as compose
  let m1 := linear 3 4 "fc1"
  let m2 := linear 4 2 "fc2"
  let composed := compose m1 m2
  let piped := m1 |>> m2

  results := results ++ [← assertEq "module: pipeline params match compose"
    (piped.params.map Prod.fst) (composed.params.map Prod.fst)]

  -- Pipeline with relu
  let model := linear 4 3 "l1" |>> (reluLayer [3] : Module Float _ _) |>> linear 3 2 "l2"
  let paramNames := model.params.map Prod.fst
  results := results ++ [← assertTrue "module: pipeline has l1.weight"
    (paramNames.contains "l1.weight")]
  results := results ++ [← assertTrue "module: pipeline has l2.bias"
    (paramNames.contains "l2.bias")]
  results := results ++ [← assertEq "module: pipeline param count" model.params.length 4]

  -- Identity module
  let idMod := identity (α := Float) [3]
  results := results ++ [← assertEq "module: identity no params" idMod.params.length 0]

  return results

/-! ### CPU Backend Extended tests (transpose, broadcast, slice, reduce, einsum) -/

def cpuExtendedTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- Transpose via CPU backend: [2,3] → [3,2]
  let a : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let aT := Tensor.transpose2D a
  let plan := Lithe.Backend.TensorExpr.toExecPlan aT
  match plan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu ext: transpose [2,3]->[3,2]"
      result.toList [1, 4, 2, 5, 3, 6]]
  | .error e =>
    results := results ++ [mkFail "cpu ext: transpose [2,3]->[3,2]" e]

  -- Broadcast via CPU backend: [1,3] → [2,3]
  let b1 : TensorExpr Float [1, 3] := .literal [1, 3] #v[10, 20, 30]
  let bBcast := TensorExpr.broadcast b1 [2, 3] sorry
  let bPlan := Lithe.Backend.TensorExpr.toExecPlan bBcast
  match bPlan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu ext: broadcast [1,3]->[2,3]"
      result.toList [10, 20, 30, 10, 20, 30]]
  | .error e =>
    results := results ++ [mkFail "cpu ext: broadcast [1,3]->[2,3]" e]

  -- Slice via CPU backend: [4,3] slice [(1,2),(0,3)]
  let t : TensorExpr Float [4, 3] := .literal [4, 3] #v[1,2,3, 4,5,6, 7,8,9, 10,11,12]
  let sliced := Tensor.sliceWith t [(1, 2), (0, 3)]
  let slicePlan := Lithe.Backend.TensorExpr.toExecPlan sliced
  match slicePlan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu ext: slice [4,3]->[(1,2),(0,3)]"
      result.toList [4, 5, 6, 7, 8, 9]]
  | .error e =>
    results := results ++ [mkFail "cpu ext: slice [4,3]->[(1,2),(0,3)]" e]

  -- Reduce sum via CPU backend: [2,3] → [3]
  let sumAxis0 := TensorExpr.reduce .sum ⟨0, by show 0 < 2; omega⟩ a
  let reducePlan := Lithe.Backend.TensorExpr.toExecPlan sumAxis0
  match reducePlan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu ext: reduce sum axis=0"
      result.toList [5, 7, 9]]
  | .error e =>
    results := results ++ [mkFail "cpu ext: reduce sum axis=0" e]

  -- Einsum (matmul) via CPU backend
  let m1 : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let m2 : TensorExpr Float [3, 2] := .literal [3, 2] #v[1, 0, 0, 1, 1, 1]
  let mmul := Tensor.matmul m1 m2
  let mmulPlan := Lithe.Backend.TensorExpr.toExecPlan mmul
  match mmulPlan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu ext: einsum matmul"
      result.toList [4, 5, 10, 11]]
  | .error e =>
    results := results ++ [mkFail "cpu ext: einsum matmul" e]

  -- Gather via CPU backend: gather rows 2,0 from [3,2]
  let g : TensorExpr Float [3, 2] := .literal [3, 2] #v[10, 20, 30, 40, 50, 60]
  let gathered := TensorExpr.gather g ⟨0, by show 0 < 2; omega⟩ #v[2, 0]
  let gPlan := Lithe.Backend.TensorExpr.toExecPlan gathered
  match gPlan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu ext: gather rows 2,0"
      result.toList [50, 60, 10, 20]]
  | .error e =>
    results := results ++ [mkFail "cpu ext: gather rows 2,0" e]

  -- Scan via CPU backend
  let v5 : TensorExpr Float [5] := .literal [5] #v[1, 2, 3, 4, 5]
  let cs := Tensor.cumsum ⟨0, by show 0 < 1; omega⟩ v5
  let csPlan := Lithe.Backend.TensorExpr.toExecPlan cs
  match csPlan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu ext: scan cumsum"
      result.toList [1, 3, 6, 10, 15]]
  | .error e =>
    results := results ++ [mkFail "cpu ext: scan cumsum" e]

  -- Concat via CPU backend
  let c1 : TensorExpr Float [2, 2] := .literal [2, 2] #v[1, 2, 3, 4]
  let c2 : TensorExpr Float [2, 2] := .literal [2, 2] #v[5, 6, 7, 8]
  let catted := TensorExpr.concat c1 c2 ⟨0, by show 0 < 2; omega⟩ sorry
  let catPlan := Lithe.Backend.TensorExpr.toExecPlan catted
  match catPlan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu ext: concat axis=0"
      result.toList [1, 2, 3, 4, 5, 6, 7, 8]]
  | .error e =>
    results := results ++ [mkFail "cpu ext: concat axis=0" e]

  -- Pad via CPU backend
  let p : TensorExpr Float [2, 2] := .literal [2, 2] #v[1, 2, 3, 4]
  let padded := TensorExpr.pad p [(1, 1), (0, 0)] 0.0 sorry
  let padPlan := Lithe.Backend.TensorExpr.toExecPlan padded
  match padPlan.execute with
  | .ok result =>
    results := results ++ [← assertListClose "cpu ext: pad"
      result.toList [0, 0, 1, 2, 3, 4, 0, 0]]
  | .error e =>
    results := results ++ [mkFail "cpu ext: pad" e]

  return results

/-! ### Autodiff Extended tests -/

def autodiffExtendedTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- Einsum gradient: matmul grad should produce non-empty grads for both inputs
  let wVar : TensorExpr Float [3, 2] := .var "w" [3, 2]
  let xLit : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let mmul := Tensor.matmul xLit wVar
  -- Reduce to scalar: sum all elements
  let sum0 := TensorExpr.reduce .sum ⟨0, by show 0 < 2; omega⟩ mmul
  let loss := TensorExpr.reduce .sum ⟨0, by show 0 < 1; omega⟩ sum0
  let grads := loss.grad

  let hasW := grads.any fun p => p.1 == "w"
  results := results ++ [← assertTrue "autodiff ext: matmul grad has 'w'" hasW
    "gradient map does not contain variable 'w'"]

  -- Check gradient for w has correct shape [3, 2]
  match grads.find? fun p => p.1 == "w" with
  | some (_, ⟨shape, _⟩) =>
    results := results ++ [← assertEq "autodiff ext: matmul grad 'w' shape" shape [3, 2]]
  | none =>
    results := results ++ [mkFail "autodiff ext: matmul grad 'w' shape" "variable 'w' not in grads"]

  -- Reduce sum gradient: grad of sum(x) should broadcast back
  let xVar : TensorExpr Float [3] := .var "x" [3]
  let sumExpr := TensorExpr.reduce .sum ⟨0, by show 0 < 1; omega⟩ xVar
  let sumGrads := sumExpr.grad
  let hasX := sumGrads.any fun p => p.1 == "x"
  results := results ++ [← assertTrue "autodiff ext: reduce sum grad has 'x'" hasX]

  -- Check that gradient evaluates correctly: d/dx sum(x) = [1, 1, 1]
  match sumGrads.find? fun p => p.1 == "x" with
  | some (_, ⟨_, gradExpr⟩) =>
    let env : Env Float := [("x", ⟨[3], #v[2.0, 3.0, 4.0]⟩)]
    match gradExpr.evalWith env with
    | .ok result =>
      results := results ++ [← assertListClose "autodiff ext: reduce sum grad values"
        result.toList [1.0, 1.0, 1.0]]
    | .error e =>
      results := results ++ [mkFail "autodiff ext: reduce sum grad values" e]
  | none =>
    results := results ++ [mkFail "autodiff ext: reduce sum grad values" "no 'x' in grads"]

  -- Broadcast gradient: grad through broadcast should reduce back
  let bVar : TensorExpr Float [1, 3] := .var "b" [1, 3]
  let bBcast := TensorExpr.broadcast bVar [2, 3] sorry
  let bSum0 := TensorExpr.reduce .sum ⟨0, by show 0 < 2; omega⟩ bBcast
  let bSum1 := TensorExpr.reduce .sum ⟨0, by show 0 < 1; omega⟩ bSum0
  let bGrads := bSum1.grad
  let hasB := bGrads.any fun p => p.1 == "b"
  results := results ++ [← assertTrue "autodiff ext: broadcast grad has 'b'" hasB]

  return results

/-! ### NN Layer tests -/

def nnLayerTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- GELU: check values at specific points
  let x : TensorExpr Float [4] := .literal [4] #v[-1.0, 0.0, 1.0, 2.0]
  let geluResult := NN.gelu x
  let geluVals := geluResult.eval.toList
  -- GELU(-1) ≈ -0.1588, GELU(0) = 0, GELU(1) ≈ 0.8412, GELU(2) ≈ 1.9545
  results := results ++ [← assertClose "nn: gelu(-1)" (geluVals.getD 0 0) (-0.1588) 0.01]
  results := results ++ [← assertClose "nn: gelu(0)" (geluVals.getD 1 0) 0.0 0.001]
  results := results ++ [← assertClose "nn: gelu(1)" (geluVals.getD 2 0) 0.8412 0.01]
  results := results ++ [← assertClose "nn: gelu(2)" (geluVals.getD 3 0) 1.9545 0.01]

  -- Softmax: row sums should be ~1.0
  let logits : TensorExpr Float [2, 3] := .literal [2, 3] #v[1, 2, 3, 4, 5, 6]
  let smResult := NN.softmax2D 2 3 logits
  let smVals := smResult.eval.toList
  let row0Sum := (smVals.getD 0 0) + (smVals.getD 1 0) + (smVals.getD 2 0)
  let row1Sum := (smVals.getD 3 0) + (smVals.getD 4 0) + (smVals.getD 5 0)
  results := results ++ [← assertClose "nn: softmax row0 sum=1" row0Sum 1.0 0.001]
  results := results ++ [← assertClose "nn: softmax row1 sum=1" row1Sum 1.0 0.001]

  -- Layer norm: output should have mean≈0 and var≈1 for each row
  let lnInput : TensorExpr Float [2, 4] := .literal [2, 4] #v[1, 2, 3, 4, 5, 6, 7, 8]
  let env : Env Float := [
    ("test_ln.weight", ⟨[4], #v[1, 1, 1, 1]⟩),
    ("test_ln.bias",   ⟨[4], #v[0, 0, 0, 0]⟩)
  ]
  let lnResult := NN.layerNorm 2 4 "test_ln" lnInput
  match lnResult.evalWith env with
  | .ok result =>
    let vals := result.toList
    -- Row 0 mean should be ≈ 0
    let mean0 := ((vals.getD 0 0) + (vals.getD 1 0) + (vals.getD 2 0) + (vals.getD 3 0)) / 4.0
    results := results ++ [← assertClose "nn: layernorm row0 mean≈0" mean0 0.0 0.01]
    -- Row 0 var should be ≈ 1
    let var0 := ((vals.getD 0 0) * (vals.getD 0 0) + (vals.getD 1 0) * (vals.getD 1 0) +
                 (vals.getD 2 0) * (vals.getD 2 0) + (vals.getD 3 0) * (vals.getD 3 0)) / 4.0
    results := results ++ [← assertClose "nn: layernorm row0 var≈1" var0 1.0 0.1]
  | .error e =>
    results := results ++ [mkFail "nn: layernorm output" e]

  -- Embedding: gather should select correct rows
  let embEnv : Env Float := [
    ("emb.weight", ⟨[5, 3], #v[0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4]⟩)
  ]
  let emb := NN.embedding 5 3 "emb" [2, 0, 4]
  match emb.evalWith embEnv with
  | .ok result =>
    results := results ++ [← assertListClose "nn: embedding lookup"
      result.toList [2,2,2, 0,0,0, 4,4,4]]
  | .error e =>
    results := results ++ [mkFail "nn: embedding lookup" e]

  return results

/-! ### Optimizer tests -/

def optimizerTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- Adam step moves parameters
  let env : Env Float := [
    ("w", ⟨[3], #v[1.0, 2.0, 3.0]⟩)
  ]
  let gradEnv : Env Float := [
    ("w", ⟨[3], #v[0.1, 0.2, 0.3]⟩)
  ]
  let config : Optim.AdamConfig := {}
  let state := Optim.AdamState.init [("w", [3])]
  let (newEnv, newState) := Optim.adamStep config env gradEnv state

  -- Parameters should have changed
  match newEnv.find? (fun p => p.1 == "w") with
  | some (_, td) =>
    let newVals := td.data.toList
    -- After one Adam step, params should decrease (positive gradient → decrease)
    results := results ++ [← assertTrue "optim: adam step changes params"
      ((newVals.getD 0 0) != 1.0)]
    results := results ++ [← assertTrue "optim: adam step decreases param"
      ((newVals.getD 0 0) < 1.0) "param should decrease with positive gradient"]
  | none =>
    results := results ++ [mkFail "optim: adam step" "w not in env after step"]

  -- State should have incremented step
  results := results ++ [← assertEq "optim: adam state step" newState.step 1]

  return results

/-! ### Safetensors parser tests -/

def safetensorsTests : IO (List TestResult) := do
  let mut results : List TestResult := []

  -- JSON parser test
  let jsonStr := "{\"tensor1\": {\"dtype\": \"F32\", \"shape\": [2, 3], \"data_offsets\": [0, 24]}}"
  match Safetensors.parseJson jsonStr with
  | some (.obj kvs) =>
    results := results ++ [← assertTrue "safetensors: json parse object"
      (kvs.length == 1)]
    match kvs.head? with
    | some ("tensor1", .obj fields) =>
      results := results ++ [← assertTrue "safetensors: json has dtype"
        (fields.any fun p => p.1 == "dtype")]
      results := results ++ [← assertTrue "safetensors: json has shape"
        (fields.any fun p => p.1 == "shape")]
    | _ =>
      results := results ++ [mkFail "safetensors: json parse tensor" "unexpected structure"]
  | some _ =>
    results := results ++ [mkFail "safetensors: json parse" "expected object"]
  | none =>
    results := results ++ [mkFail "safetensors: json parse" "parse returned none"]

  -- Float32 decode test
  -- IEEE 754: 1.0 = 0x3F800000 = bytes [0x00, 0x00, 0x80, 0x3F] in LE
  let oneFloat := Safetensors.float32FromBytes 0x00 0x00 0x80 0x3F
  results := results ++ [← assertClose "safetensors: decode float32 1.0" oneFloat 1.0 0.001]

  -- IEEE 754: -2.0 = 0xC0000000 = bytes [0x00, 0x00, 0x00, 0xC0] in LE
  let negTwoFloat := Safetensors.float32FromBytes 0x00 0x00 0x00 0xC0
  results := results ++ [← assertClose "safetensors: decode float32 -2.0" negTwoFloat (-2.0) 0.001]

  -- IEEE 754: 0.5 = 0x3F000000 = bytes [0x00, 0x00, 0x00, 0x3F] in LE
  let halfFloat := Safetensors.float32FromBytes 0x00 0x00 0x00 0x3F
  results := results ++ [← assertClose "safetensors: decode float32 0.5" halfFloat 0.5 0.001]

  return results

/-! ### Runner -/

def main : IO Unit := do
  IO.println "=== Lithe Test Suite ==="
  IO.println ""

  let suites : List (String × IO (List TestResult)) := [
    ("Eval",           evalTests),
    ("Env",            envTests),
    ("Autodiff",       autodiffTests),
    ("Viz",            vizTests),
    ("CPU Backend",    cpuBackendTests),
    ("Codegen",        codegenTests),
    ("Module",         moduleTests),
    ("Dim/DimShape",   dimTests),
    ("NamedTensor",    namedTensorTests),
    ("Slicing",        slicingTests),
    ("Named Slicing",  namedSlicingTests),
    ("Einsum",         einsumTests),
    ("Module Pipeline", modulePipelineTests),
    ("CPU Extended",   cpuExtendedTests),
    ("Autodiff Extended", autodiffExtendedTests),
    ("NN Layers",      nnLayerTests),
    ("Optimizer",      optimizerTests),
    ("Safetensors",    safetensorsTests)
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
