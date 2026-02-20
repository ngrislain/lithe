/-
  Lithe/Shape.lean — Shape definitions and operations
-/

abbrev Shape := List Nat

namespace Shape

def product : Shape → Nat
  | [] => 1
  | d :: ds => d * product ds

def rank (s : Shape) : Nat := s.length

def removeAt (s : Shape) (i : Fin s.length) : Shape := s.eraseIdx i

-- Basic lemmas

@[simp] theorem product_nil : product [] = 1 := rfl

@[simp] theorem product_cons (d : Nat) (ds : Shape) :
    product (d :: ds) = d * product ds := rfl

/-- Map with index helper. -/
private def mapIdxAux (f : Nat → Nat → Nat) : List Nat → Nat → List Nat
  | [], _ => []
  | x :: xs, i => f i x :: mapIdxAux f xs (i + 1)

/-- Permute shape dimensions according to a permutation vector. -/
def permuteShape (s : Shape) (perm : Vector (Fin s.length) s.length) : Shape :=
  (Vector.ofFn fun i => s.get (perm.get i)).toList

/-- Compute output shape of concatenation along an axis. -/
def concatShape (s₁ s₂ : Shape) (axis : Fin s₁.length) : Shape :=
  mapIdxAux (fun i d => if i == axis.val then d + List.getD s₂ i 0 else d) s₁ 0

/-- Gather shape: replace dim at axis with numIdx. -/
def gatherShape (s : Shape) (axis : Fin s.length) (numIdx : Nat) : Shape :=
  mapIdxAux (fun i d => if i == axis.val then numIdx else d) s 0

/-- Pad shape: add padding to each dimension. -/
def padShape : Shape → List (Nat × Nat) → Shape
  | [], _ => []
  | s, [] => s
  | d :: ds, (lo, hi) :: ps => (lo + d + hi) :: padShape ds ps

/-- Valid slice: starts + sizes stay within bounds. -/
def ValidSlice (s : Shape) (starts sizes : List Nat) : Prop :=
  starts.length = s.length ∧ sizes.length = s.length ∧
  ∀ i : Fin s.length, List.getD starts i.val 0 + List.getD sizes i.val 0 ≤ s.get i

instance decValidSlice (s : Shape) (starts sizes : List Nat) :
    Decidable (ValidSlice s starts sizes) := by
  unfold ValidSlice; exact inferInstance

/-- Concat-compatible: ranks match and all dims match except at axis.
    Note: We keep this simple — checking length equality and dim compatibility. -/
def ConcatCompatible (s₁ s₂ : Shape) (axis : Nat) : Prop :=
  s₁.length = s₂.length ∧
  ∀ i : Fin s₁.length, i.val ≠ axis →
    List.getD s₁ i.val 0 = List.getD s₂ i.val 0

instance decConcatCompatible (s₁ s₂ : Shape) (axis : Nat) :
    Decidable (ConcatCompatible s₁ s₂ axis) := by
  unfold ConcatCompatible; exact inferInstance

/-- Broadcastable: each dim is equal or source is 1. -/
def IsBroadcastable (s₁ s₂ : Shape) : Prop :=
  s₁.length = s₂.length ∧
  ∀ i : Fin s₁.length, List.getD s₁ i.val 0 = List.getD s₂ i.val 0 ∨ List.getD s₁ i.val 0 = 1

instance decIsBroadcastable (s₁ s₂ : Shape) :
    Decidable (IsBroadcastable s₁ s₂) := by
  unfold IsBroadcastable; exact inferInstance

/-- Einsum validity — label counts match ranks, shared labels have matching dims. -/
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

instance decIsEinsumValid (subsA subsB subsOut : List Nat) (sA sB sOut : Shape) :
    Decidable (IsEinsumValid subsA subsB subsOut sA sB sOut) := by
  unfold IsEinsumValid; exact inferInstance

end Shape
