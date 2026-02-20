/-
  Lithe/Index.lean — Index arithmetic for tensor evaluation
-/
import Lithe.Shape
import Lithe.Scalar

namespace Lithe

private theorem linearIndex_lt (m n : Nat) (i : Fin m) (j : Fin n) :
    i.val * n + j.val < m * n := by
  have hi := i.isLt
  have hj := j.isLt
  have step1 : i.val * n + j.val < i.val * n + n := Nat.add_lt_add_left hj _
  have step2 : i.val * n + n = (i.val + 1) * n := by simp [Nat.succ_mul]
  have step3 : (i.val + 1) * n ≤ m * n := Nat.mul_le_mul_right n hi
  omega

/-- Convert 2D indices (i, j) into a linear index into a flat array of size m * n. -/
def linearIndex (m n : Nat) (i : Fin m) (j : Fin n) : Fin (m * n) :=
  ⟨i.val * n + j.val, linearIndex_lt m n i j⟩

/-- Decompose a linear index into 2D indices. -/
def decomposeIndex (m n : Nat) (hn : n > 0) (idx : Fin (m * n)) : Fin m × Fin n :=
  (⟨idx.val / n, by
    have hidx := idx.isLt
    exact Nat.div_lt_of_lt_mul (Nat.mul_comm m n ▸ hidx)⟩,
   ⟨idx.val % n, Nat.mod_lt _ hn⟩)

/-- Fold over Fin n, accumulating a scalar sum. -/
def finFold [Scalar α] (n : Nat) (f : Fin n → α) : α :=
  go 0 (Nat.zero_le n) 0
where
  go (i : Nat) (_ : i ≤ n) (acc : α) : α :=
    if hlt : i < n then
      go (i + 1) hlt (acc + f ⟨i, hlt⟩)
    else
      acc

/-! ### N-dimensional index arithmetic -/

/-- Convert multi-dimensional indices to a linear index.
    `dims` is the shape, `indices` are the coordinates (one per dim). -/
def multiToLinear (dims : List Nat) (indices : List Nat) : Nat :=
  go dims indices 0
where
  go : List Nat → List Nat → Nat → Nat
    | [], _, acc => acc
    | _, [], acc => acc
    | d :: ds, i :: is, acc => go ds is (acc * d + i)

/-- Convert a linear index to multi-dimensional indices.
    Returns indices from most-significant to least-significant dimension. -/
def linearToMulti (dims : List Nat) (idx : Nat) : List Nat :=
  let strides := suffixProducts dims
  go dims strides 0 idx
where
  suffixProducts : List Nat → List Nat
    | [] => [1]
    | d :: ds =>
      let rest := suffixProducts ds
      match rest with
      | [] => [d]  -- shouldn't happen
      | h :: _ => (d * h) :: rest
  go (dims strides : List Nat) (pos : Nat) (idx : Nat) : List Nat :=
    match dims with
    | [] => []
    | d :: ds =>
      let stride := List.getD strides (pos + 1) 1
      let coord := (idx / stride) % d
      coord :: go ds strides (pos + 1) idx

/-- Compute the stride for a given axis in a shape. -/
def axisStride (s : Shape) (axis : Nat) : Nat :=
  Shape.product (s.drop (axis + 1))

end Lithe
