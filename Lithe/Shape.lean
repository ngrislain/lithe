/-
  Lithe/Shape.lean — Shape algebra for dependently-typed tensors

  Defines the `Shape` type and operations for computing output shapes of tensor
  operations, together with decidable predicates that gate well-formedness at
  type-checking time (slicing, concatenation, broadcasting, einsum).
-/

/-- Type alias for tensor shapes: a list of natural numbers
    $s = [d_1, d_2, \ldots, d_r]$ where $r$ is the rank and each $d_i$ is
    the extent of dimension $i$. -/
abbrev Shape := List Nat

namespace Shape

/-- Total number of elements: $|s| = \prod_{i=1}^{r} d_i$,
    with $|[\,]| = 1$ for scalar shapes. -/
def product : Shape → Nat
  | [] => 1
  | d :: ds => d * product ds

/-- Tensor rank (number of dimensions): $\operatorname{rank}(s) = r$. -/
def rank (s : Shape) : Nat := s.length

/-- Remove dimension $i$ from a shape:
    $[d_1, \ldots, d_r] \mapsto [d_1, \ldots, d_{i-1}, d_{i+1}, \ldots, d_r]$. -/
def removeAt (s : Shape) (i : Fin s.length) : Shape := s.eraseIdx i

-- Basic lemmas

/-- $\prod \varnothing = 1$ (empty shape is a scalar). -/
@[simp] theorem product_nil : product [] = 1 := rfl

/-- $\prod (d :: ds) = d \cdot \prod ds$. -/
@[simp] theorem product_cons (d : Nat) (ds : Shape) :
    product (d :: ds) = d * product ds := rfl

/-- Map with index helper. -/
private def mapIdxAux (f : Nat → Nat → Nat) : List Nat → Nat → List Nat
  | [], _ => []
  | x :: xs, i => f i x :: mapIdxAux f xs (i + 1)

/-- Permute shape dimensions:
    $s' = [d_{\pi(1)}, d_{\pi(2)}, \ldots, d_{\pi(r)}]$ for permutation $\pi$. -/
def permuteShape (s : Shape) (perm : Vector (Fin s.length) s.length) : Shape :=
  (Vector.ofFn fun i => s.get (perm.get i)).toList

/-- Output shape after concatenating along axis $k$:
    dimension $k$ becomes $d_k^{(1)} + d_k^{(2)}$, others unchanged. -/
def concatShape (s₁ s₂ : Shape) (axis : Fin s₁.length) : Shape :=
  mapIdxAux (fun i d => if i == axis.val then d + List.getD s₂ i 0 else d) s₁ 0

/-- Output shape after gather: dimension at axis $k$ becomes the number of
    gathered indices. -/
def gatherShape (s : Shape) (axis : Fin s.length) (numIdx : Nat) : Shape :=
  mapIdxAux (fun i d => if i == axis.val then numIdx else d) s 0

/-- Output shape after padding: $d'_i = \operatorname{lo}_i + d_i + \operatorname{hi}_i$. -/
def padShape : Shape → List (Nat × Nat) → Shape
  | [], _ => []
  | s, [] => s
  | d :: ds, (lo, hi) :: ps => (lo + d + hi) :: padShape ds ps

/-- Valid slice predicate: $\operatorname{start}_i + \operatorname{size}_i \le d_i$
    for all $i$. -/
def ValidSlice (s : Shape) (starts sizes : List Nat) : Prop :=
  starts.length = s.length ∧ sizes.length = s.length ∧
  ∀ i : Fin s.length, List.getD starts i.val 0 + List.getD sizes i.val 0 ≤ s.get i

/-- Decidability instance for `ValidSlice`. -/
instance decValidSlice (s : Shape) (starts sizes : List Nat) :
    Decidable (ValidSlice s starts sizes) := by
  unfold ValidSlice; exact inferInstance

/-- Concatenation compatibility: $\operatorname{rank}(s_1) = \operatorname{rank}(s_2)$
    and $d_i^{(1)} = d_i^{(2)}$ for all $i \ne k$. -/
def ConcatCompatible (s₁ s₂ : Shape) (axis : Nat) : Prop :=
  s₁.length = s₂.length ∧
  ∀ i : Fin s₁.length, i.val ≠ axis →
    List.getD s₁ i.val 0 = List.getD s₂ i.val 0

/-- Decidability instance for `ConcatCompatible`. -/
instance decConcatCompatible (s₁ s₂ : Shape) (axis : Nat) :
    Decidable (ConcatCompatible s₁ s₂ axis) := by
  unfold ConcatCompatible; exact inferInstance

/-- Broadcasting predicate: $\operatorname{rank}(s_1) = \operatorname{rank}(s_2)$ and
    for each $i$, either $d_i^{(1)} = d_i^{(2)}$ or $d_i^{(1)} = 1$. -/
def IsBroadcastable (s₁ s₂ : Shape) : Prop :=
  s₁.length = s₂.length ∧
  ∀ i : Fin s₁.length, List.getD s₁ i.val 0 = List.getD s₂ i.val 0 ∨ List.getD s₁ i.val 0 = 1

/-- Decidability instance for `IsBroadcastable`. -/
instance decIsBroadcastable (s₁ s₂ : Shape) :
    Decidable (IsBroadcastable s₁ s₂) := by
  unfold IsBroadcastable; exact inferInstance

/-- Einsum validity: subscript counts match ranks, shared labels imply equal
    dimensions, and every output label traces back to an input dimension. -/
def IsEinsumValid (subsA subsB subsOut : List Nat) (sA sB sOut : Shape) : Prop :=
  subsA.length = sA.length ∧ subsB.length = sB.length ∧ subsOut.length = sOut.length ∧
  -- Shared labels have matching dimensions
  (∀ ia : Fin subsA.length, ∀ ib : Fin subsB.length,
    List.getD subsA ia.val 0 = List.getD subsB ib.val 0 →
    List.getD sA ia.val 0 = List.getD sB ib.val 0) ∧
  -- Output dims match their source input dims
  (∀ o : Fin subsOut.length,
    (∃ ia : Fin subsA.length,
      List.getD subsOut o.val 0 = List.getD subsA ia.val 0 ∧
      List.getD sOut o.val 0 = List.getD sA ia.val 0) ∨
    (∃ ib : Fin subsB.length,
      List.getD subsOut o.val 0 = List.getD subsB ib.val 0 ∧
      List.getD sOut o.val 0 = List.getD sB ib.val 0))

/-- Decidability instance for `IsEinsumValid`. -/
instance decIsEinsumValid (subsA subsB subsOut : List Nat) (sA sB sOut : Shape) :
    Decidable (IsEinsumValid subsA subsB subsOut sA sB sOut) := by
  unfold IsEinsumValid; exact inferInstance

end Shape
