/-
  Lithe/Proofs/Shape.lean — Shape product lemmas
-/
import Lithe.Shape

namespace Shape

theorem product_append (s₁ s₂ : Shape) :
    product (s₁ ++ s₂) = product s₁ * product s₂ := by
  induction s₁ with
  | nil => simp [product]
  | cons d ds ih => simp [product, ih, Nat.mul_assoc]

theorem product_swap (m n : Nat) :
    product [m, n] = product [n, m] := by
  simp [product, Nat.mul_comm]

theorem product_singleton (n : Nat) :
    product [n] = n := by
  simp [product]

theorem product_pair (m n : Nat) :
    product [m, n] = m * n := by
  simp [product, Nat.mul_one]

theorem product_triple (a b c : Nat) :
    product [a, b, c] = a * (b * (c * 1)) := by
  simp [product]

end Shape
