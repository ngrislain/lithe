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

/-- Map 2D indices to linear: $(i, j) \mapsto i \cdot n + j$, returning a `Fin (m \cdot n)`. -/
def linearIndex (m n : Nat) (i : Fin m) (j : Fin n) : Fin (m * n) :=
  ⟨i.val * n + j.val, linearIndex_lt m n i j⟩

/-- Inverse of `linearIndex`: $\ell \mapsto (\lfloor \ell / n \rfloor,\; \ell \bmod n)$. -/
def decomposeIndex (m n : Nat) (hn : n > 0) (idx : Fin (m * n)) : Fin m × Fin n :=
  (⟨idx.val / n, by
    have hidx := idx.isLt
    exact Nat.div_lt_of_lt_mul (Nat.mul_comm m n ▸ hidx)⟩,
   ⟨idx.val % n, Nat.mod_lt _ hn⟩)

/-- Fold over $\{0, \ldots, n-1\}$: $\sum_{i=0}^{n-1} f(i)$ using `Scalar` addition. -/
def finFold [Scalar α] (n : Nat) (f : Fin n → α) : α :=
  go 0 (Nat.zero_le n) 0
where
  go (i : Nat) (_ : i ≤ n) (acc : α) : α :=
    if hlt : i < n then
      go (i + 1) hlt (acc + f ⟨i, hlt⟩)
    else
      acc

/-! ### N-dimensional index arithmetic -/

/-- Row-major linearization: $(i_1, \ldots, i_r) \mapsto \sum_{k=1}^{r} i_k \cdot \prod_{j=k+1}^{r} d_j$.
    `dims` is the shape, `indices` are the coordinates (one per dim). -/
def multiToLinear (dims : List Nat) (indices : List Nat) : Nat :=
  go dims indices 0
where
  go : List Nat → List Nat → Nat → Nat
    | [], _, acc => acc
    | _, [], acc => acc
    | d :: ds, i :: is, acc => go ds is (acc * d + i)

/-- Inverse of `multiToLinear`: recovers multi-dimensional coordinates from a flat index
    via $i_k = \lfloor \ell / \sigma_k \rfloor \bmod d_k$ where $\sigma_k$ is the stride.
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

/-- Stride for axis $k$: $\sigma_k = \prod_{j=k+1}^{r} d_j$. -/
def axisStride (s : Shape) (axis : Nat) : Nat :=
  Shape.product (s.drop (axis + 1))

end Lithe
