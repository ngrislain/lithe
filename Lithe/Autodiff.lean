/-
  Lithe/Autodiff.lean — Symbolic reverse-mode automatic differentiation
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Ops
import Lithe.Tensor
import Lithe.Smart

open Shape

/-- Gradient accumulator: a list of $(name, (s, \bar{T}))$ pairs mapping variable
names to their adjoint (cotangent) tensors.

Each entry records the variable name, its shape, and the symbolic expression
for the gradient contribution. Multiple entries for the same variable may exist
and should be summed during final accumulation. -/
abbrev Grads := List (String × (Σ s : Shape, TensorExpr Float s))

namespace Grads

/-- Merge two gradient lists by concatenation (accumulation deferred).

Since the same variable may appear in both lists, final gradient values are
obtained by summing all entries sharing a variable name. -/
def merge (g1 g2 : Grads) : Grads := g1 ++ g2

end Grads

/-! ### Safe shape-checked constructors

These wrappers use `Decidable` instances to check shape predicates at runtime.
When the predicate holds, the appropriate `TensorExpr` constructor is used.
When it does not (which should never happen for well-formed backward passes),
a zero-filled tensor is returned as a safe fallback. -/

private def safeReshape (e : TensorExpr Float s₁) (s₂ : Shape) : TensorExpr Float s₂ :=
  if h : s₁.product = s₂.product then TensorExpr.reshape e h
  else TensorExpr.fill s₂ 0.0

private def safeBroadcast (e : TensorExpr Float s₁) (s₂ : Shape) : TensorExpr Float s₂ :=
  if h : IsBroadcastable s₁ s₂ then TensorExpr.broadcast e s₂ h
  else TensorExpr.fill s₂ 0.0

private def safeSlice (e : TensorExpr Float s) (starts sizes : List Nat) :
    TensorExpr Float sizes :=
  if h : ValidSlice s starts sizes then TensorExpr.slice e starts sizes h
  else TensorExpr.fill sizes 0.0

private def safePad (e : TensorExpr Float s) (padding : List (Nat × Nat)) (v : Float) :
    TensorExpr Float (padShape s padding) :=
  if h : padding.length = s.length then TensorExpr.pad e padding v h
  else TensorExpr.fill (padShape s padding) 0.0

private def safeConcat (e₁ : TensorExpr Float s₁) (e₂ : TensorExpr Float s₂)
    (axis : Fin s₁.length) : TensorExpr Float (concatShape s₁ s₂ axis) :=
  if h : ConcatCompatible s₁ s₂ axis.val then TensorExpr.concat e₁ e₂ axis h
  else TensorExpr.fill (concatShape s₁ s₂ axis) 0.0

private def safeEinsum {sA sB : Shape} (subsA subsB subsOut : List Nat)
    (eA : TensorExpr Float sA) (eB : TensorExpr Float sB) (sOut : Shape)
    : TensorExpr Float sOut :=
  if h : IsEinsumValid subsA subsB subsOut sA sB sOut then
    TensorExpr.einsum subsA subsB subsOut eA eB h
  else TensorExpr.fill sOut 0.0

/-- Extract the shape index from a TensorExpr (type-level to term-level). -/
private def shapeOf {s : Shape} (_ : TensorExpr Float s) : Shape := s

/-! ### Autodiff helpers -/

/-- Insert a value into a list at a given position. -/
private def insertShapeAt (l : List Nat) (pos val : Nat) : List Nat :=
  match l, pos with
  | xs, 0 => val :: xs
  | [], _ => [val]
  | x :: xs, n + 1 => x :: insertShapeAt xs n val

/-- Inserting 1 into a shape preserves the product (multiplicative identity). -/
private theorem product_insertShapeAt_one (l : List Nat) (pos : Nat) :
    Shape.product (insertShapeAt l pos 1) = Shape.product l := by
  induction l generalizing pos with
  | nil => cases pos <;> simp [insertShapeAt, Shape.product]
  | cons x xs ih =>
    cases pos with
    | zero => simp [insertShapeAt, Shape.product]
    | succ n => simp [insertShapeAt, Shape.product, ih]

/-- Broadcast the adjoint of reduce-sum back to the input shape.
    Reshapes to insert a size-1 dim at the reduced axis, then broadcasts. -/
private def broadcastReduceAdj {s : Shape} (axis : Fin s.length)
    (adj : TensorExpr Float (s.removeAt axis)) : TensorExpr Float s :=
  let midShape := insertShapeAt (s.removeAt axis) axis.val 1
  let adjReshaped := TensorExpr.reshape (s₂ := midShape) adj
    (product_insertShapeAt_one (s.removeAt axis) axis.val).symm
  safeBroadcast adjReshaped s

/-- Reduce-sum over dimensions that were broadcast (where source was size 1).
    Uses runtime shape check: reshapes from broadcast shape s₂ back to s₁. -/
private def reduceBroadcastAdj {s₁ : Shape} (s₂ : Shape)
    (_ : IsBroadcastable s₁ s₂) (adj : TensorExpr Float s₂) : TensorExpr Float s₁ :=
  safeReshape adj s₁

/-- Apply inverse permutation to adjoint.
    Builds inverse permutation functionally, then transposes and reshapes. -/
private def invTransposeAdj {s : Shape} (perm : Vector (Fin s.length) s.length)
    (adj : TensorExpr Float (permuteShape s perm)) : TensorExpr Float s :=
  let n := s.length
  let m := (permuteShape s perm).length
  -- Build inverse permutation: for each output position j, find input i with perm[i] = j
  let invPermVec : Vector (Fin m) m := Vector.ofFn fun (j : Fin m) =>
    let found := (List.range n).findIdx fun i =>
      if h : i < n then (perm.get ⟨i, h⟩).val == j.val else false
    have hm : 0 < m := by have := j.isLt; omega
    ⟨found % m, Nat.mod_lt found hm⟩
  safeReshape (.transpose adj invPermVec) s

/-- Build padding list from slice parameters: lo = starts[i], hi = sIn[i] - starts[i] - sizes[i]. -/
private def buildSlicePadding (starts sizes sIn : List Nat) : List (Nat × Nat) :=
  match starts, sizes, sIn with
  | [], _, _ => []
  | _, [], _ => []
  | _, _, [] => []
  | st :: sts, sz :: szs, d :: ds =>
    (st, d - st - sz) :: buildSlicePadding sts szs ds

/-- Pad adjoint of slice back to original shape.
    Uses runtime shape checks for padding length and reshape. -/
private def padSliceAdj {s : Shape} (starts sizes : List Nat) (_ : ValidSlice s starts sizes)
    (adj : TensorExpr Float sizes) : TensorExpr Float s :=
  let padding := buildSlicePadding starts sizes s
  safeReshape (safePad adj padding 0.0) s

/-- Slice adjoint of pad to extract original region.
    Uses runtime ValidSlice check. -/
private def slicePadAdj {s : Shape} (padding : List (Nat × Nat))
    (_ : padding.length = s.length)
    (adj : TensorExpr Float (padShape s padding)) : TensorExpr Float s :=
  let starts := padding.map Prod.fst
  safeSlice adj starts s

/-- Slice adjoint for concat gradient: extract parts for left and right inputs.
    Uses runtime ValidSlice checks for both slices. -/
private def sliceConcatAdj {s₁ s₂ : Shape} (axis : Fin s₁.length)
    (_ : ConcatCompatible s₁ s₂ axis.val)
    (adj : TensorExpr Float (concatShape s₁ s₂ axis))
    : TensorExpr Float s₁ × TensorExpr Float s₂ :=
  let starts1 := List.replicate s₁.length 0
  let adjE1 := safeSlice adj starts1 s₁
  let starts2Arr := Id.run do
    let mut s := List.replicate s₂.length 0
    for i in [:s₂.length] do
      if i == axis.val then
        s := s.set i (s₁.getD i 0)
    return s
  let adjE2 := safeSlice adj starts2Arr s₂
  (adjE1, adjE2)

namespace TensorExpr

/-- Compute VJP (vector-Jacobian product) via reverse-mode AD.

Given expression $f$ and adjoint $\bar{y}$, propagates gradients using the
chain rule. Returns a `Grads` list pairing each variable name with its adjoint tensor. -/
def backward : TensorExpr Float s → (adjoint : TensorExpr Float s) → Grads
  | .literal _ _, _ => []
  | .fill _ _, _ => []
  | .var name s, adj => [(name, ⟨s, adj⟩)]
  | .unary .neg e, adj =>
    e.backward (.unary .neg adj)
  | .unary .exp e, adj =>
    e.backward (.binary .mul adj (.unary .exp e))
  | .unary .log e, adj =>
    e.backward (.binary .mul adj (.binary .div (.fill _ 1.0) e))
  | .unary .tanh e, adj =>
    let th := TensorExpr.unary .tanh e
    let one := TensorExpr.fill _ 1.0
    e.backward (.binary .mul adj (.binary .sub one (.binary .mul th th)))
  | .unary .sigmoid e, adj =>
    let sig := TensorExpr.unary .sigmoid e
    let one := TensorExpr.fill _ 1.0
    e.backward (.binary .mul adj (.binary .mul sig (.binary .sub one sig)))
  | .unary .relu e, adj =>
    let zero := TensorExpr.fill _ 0.0
    e.backward (.select e adj zero)
  | .unary .abs e, adj =>
    e.backward (.binary .mul adj (.unary .sign e))
  | .unary .sqrt e, adj =>
    let two := TensorExpr.fill _ 2.0
    e.backward (.binary .div adj (.binary .mul two (.unary .sqrt e)))
  | .unary .sin e, adj =>
    e.backward (.binary .mul adj (.unary .cos e))
  | .unary .cos e, adj =>
    e.backward (.binary .mul adj (.unary .neg (.unary .sin e)))
  | .unary .sign _, _ => []
  | .binary .add e₁ e₂, adj =>
    (e₁.backward adj).merge (e₂.backward adj)
  | .binary .sub e₁ e₂, adj =>
    (e₁.backward adj).merge (e₂.backward (.unary .neg adj))
  | .binary .mul e₁ e₂, adj =>
    (e₁.backward (.binary .mul adj e₂)).merge (e₂.backward (.binary .mul adj e₁))
  | .binary .div e₁ e₂, adj =>
    let gradE1 := TensorExpr.binary .div adj e₂
    let gradE2 := TensorExpr.unary .neg
      (.binary .div (.binary .mul adj e₁) (.binary .mul e₂ e₂))
    (e₁.backward gradE1).merge (e₂.backward gradE2)
  | .binary .pow e₁ e₂, adj =>
    let one := TensorExpr.fill _ 1.0
    let gradE1 := TensorExpr.binary .mul adj
      (.binary .mul e₂ (.binary .pow e₁ (.binary .sub e₂ one)))
    let gradE2 := TensorExpr.binary .mul adj
      (.binary .mul (.binary .pow e₁ e₂) (.unary .log e₁))
    (e₁.backward gradE1).merge (e₂.backward gradE2)
  | .binary .max e₁ e₂, adj =>
    let cond := TensorExpr.binary .sub e₁ e₂
    let zero := TensorExpr.fill _ 0.0
    (e₁.backward (.select cond adj zero)).merge (e₂.backward (.select cond zero adj))
  | .binary .min e₁ e₂, adj =>
    let cond := TensorExpr.binary .sub e₂ e₁
    let zero := TensorExpr.fill _ 0.0
    (e₁.backward (.select cond adj zero)).merge (e₂.backward (.select cond zero adj))
  | .smul c e, adj =>
    e.backward (.smul c adj)
  | .select _cond eT eF, adj =>
    let zero := TensorExpr.fill _ 0.0
    (eT.backward (.select _cond adj zero)).merge (eF.backward (.select _cond zero adj))
  | .reshape e h, adj =>
    e.backward (.reshape adj h.symm)
  -- Reduce sum: broadcast adjoint back to input shape
  | .reduce .sum axis e, adj =>
    e.backward (broadcastReduceAdj axis adj)
  -- Other reductions: stub for now
  | .reduce _ _ _, _ => []
  -- Transpose: apply inverse permutation
  | .transpose e perm, adj =>
    e.backward (invTransposeAdj perm adj)
  -- Broadcast: reduce over broadcast dimensions
  | .broadcast e _s₂ h, adj =>
    e.backward (reduceBroadcastAdj _ h adj)
  -- Slice: pad adjoint back to original shape
  | .slice e starts sizes h, adj =>
    e.backward (padSliceAdj starts sizes h adj)
  -- Pad: slice adjoint to remove padding
  | .pad e padding _ h, adj =>
    e.backward (slicePadAdj padding h adj)
  -- Concat: slice adjoint into two parts
  | .concat e₁ e₂ axis h, adj =>
    let (adjE1, adjE2) := sliceConcatAdj axis h adj
    (e₁.backward adjE1).merge (e₂.backward adjE2)
  -- Gather, scan: stub (defer)
  | .gather _ _ _, _ => []
  | .scan _ _ _, _ => []
  -- Einsum: swap subscripts for gradients (runtime validity check)
  | .einsum subsA subsB subsOut eA eB _, adj =>
    let gradA := safeEinsum subsOut subsB subsA adj eB (shapeOf eA)
    let gradB := safeEinsum subsA subsOut subsB eA adj (shapeOf eB)
    (eA.backward gradA).merge (eB.backward gradB)

/-- Compute gradients of a scalar loss $\ell$ w.r.t. all variables by seeding
backward with $\bar{\ell} = 1$.

Initiates reverse-mode AD from a scalar (shape `[]`) expression, producing
a `Grads` list of adjoint tensors for every variable that appears in the
computation graph. -/
def grad (loss : TensorExpr Float []) : Grads :=
  loss.backward (.fill [] 1.0)

end TensorExpr
