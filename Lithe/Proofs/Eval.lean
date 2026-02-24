/-
  Lithe/Proofs/Eval.lean — Algebraic laws hold under eval (updated for v2 GADT)

  Note: Float is an opaque primitive in Lean 4. Commutativity of IEEE 754
  addition/multiplication cannot be proved without axioms (Float.add and
  Float.mul have no reduction rules). If exact algebraic proofs are needed,
  instantiate TensorExpr with a type that has ScalarLaws.
-/
import Lithe.Eval
import Lithe.Smart

namespace TensorExpr

private theorem cast_round_trip {α : Type} {n m : Nat}
    (h₁ : n = m) (h₂ : m = n) (v : Vector α n) :
    h₂ ▸ h₁ ▸ v = v := by subst h₁; rfl

/-- Reshape is involutive: reshaping $s_1 \to s_2 \to s_1$ recovers the original data.
This is exact — no floating-point caveats. -/
theorem eval_reshape_reshape {s₁ s₂ : Shape}
    (e : TensorExpr Float s₁) (h₁₂ : s₁.product = s₂.product) (h₂₁ : s₂.product = s₁.product) :
    (TensorExpr.reshape (TensorExpr.reshape e h₁₂) h₂₁).eval = e.eval := by
  simp only [eval]
  exact cast_round_trip h₁₂ h₂₁ e.eval

end TensorExpr
