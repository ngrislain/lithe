/-
  Lithe/Proofs/Eval.lean — Algebraic laws hold under eval (updated for v2 GADT)

  Note: Float does not satisfy exact algebraic laws (IEEE 754).
  These theorems use sorry. For exact proofs, use Int with ScalarLaws.
-/
import Lithe.Eval
import Lithe.Smart

namespace TensorExpr

-- Pointwise addition is commutative
theorem eval_add_comm (e₁ e₂ : TensorExpr Float s) :
    (TensorExpr.binary .add e₁ e₂).eval = (TensorExpr.binary .add e₂ e₁).eval := by
  sorry

-- Pointwise multiplication is commutative
theorem eval_mul_comm (e₁ e₂ : TensorExpr Float s) :
    (TensorExpr.binary .mul e₁ e₂).eval = (TensorExpr.binary .mul e₂ e₁).eval := by
  sorry

private theorem cast_round_trip {α : Type} {n m : Nat}
    (h₁ : n = m) (h₂ : m = n) (v : Vector α n) :
    h₂ ▸ h₁ ▸ v = v := by subst h₁; rfl

-- Reshape is invertible (this is exact — no floating point issues)
theorem eval_reshape_reshape {s₁ s₂ : Shape}
    (e : TensorExpr Float s₁) (h₁₂ : s₁.product = s₂.product) (h₂₁ : s₂.product = s₁.product) :
    (TensorExpr.reshape (TensorExpr.reshape e h₁₂) h₂₁).eval = e.eval := by
  simp only [eval]
  exact cast_round_trip h₁₂ h₂₁ e.eval

end TensorExpr
