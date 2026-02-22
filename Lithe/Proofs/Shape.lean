/-
  Lithe/Proofs/Shape.lean — Shape product lemmas
-/
import Lithe.Shape

namespace Shape

/-- $\prod(s_1 \mathbin{+\!\!+} s_2) = \prod s_1 \cdot \prod s_2$ — product distributes over list append. -/
theorem product_append (s₁ s₂ : Shape) :
    product (s₁ ++ s₂) = product s₁ * product s₂ := by
  induction s₁ with
  | nil => simp [product]
  | cons d ds ih => simp [product, ih, Nat.mul_assoc]

/-- $\prod [m, n] = \prod [n, m]$ — product is symmetric for two dimensions. -/
theorem product_swap (m n : Nat) :
    product [m, n] = product [n, m] := by
  simp [product, Nat.mul_comm]

/-- $\prod [n] = n$ — singleton shape product is the dimension itself. -/
theorem product_singleton (n : Nat) :
    product [n] = n := by
  simp [product]

/-- $\prod [m, n] = m \cdot n$ — two-element shape product is ordinary multiplication. -/
theorem product_pair (m n : Nat) :
    product [m, n] = m * n := by
  simp [product, Nat.mul_one]

/-- $\prod [a, b, c] = a \cdot (b \cdot (c \cdot 1))$ — three-element shape product unfolds as right-associated multiplication. -/
theorem product_triple (a b c : Nat) :
    product [a, b, c] = a * (b * (c * 1)) := by
  simp [product]

end Shape
