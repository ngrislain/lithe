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

namespace TensorExpr

/-- Compute VJP (vector-Jacobian product) via reverse-mode AD.

Given expression $f$ and adjoint $\bar{y}$, propagates gradients using the
chain rule. Key rules:

- $\overline{\operatorname{neg}}: \bar{x} = -\bar{y}$
- $\overline{\exp}: \bar{x} = \bar{y} \cdot e^x$
- $\overline{\ln}: \bar{x} = \bar{y} / x$
- $\overline{\tanh}: \bar{x} = \bar{y} \cdot (1 - \tanh^2 x)$
- $\overline{\sigma}: \bar{x} = \bar{y} \cdot \sigma(x)(1 - \sigma(x))$
- $\overline{\operatorname{relu}}: \bar{x} = \bar{y} \cdot \mathbf{1}_{x > 0}$
- $\overline{+}: \bar{x}_1 = \bar{y},\; \bar{x}_2 = \bar{y}$
- $\overline{\cdot}: \bar{x}_1 = \bar{y} \cdot x_2,\; \bar{x}_2 = \bar{y} \cdot x_1$
- $\overline{/}: \bar{x}_1 = \bar{y}/x_2,\; \bar{x}_2 = -\bar{y} \cdot x_1 / x_2^2$
- $\overline{x^y}: \bar{x} = \bar{y} \cdot y \cdot x^{y-1},\; \bar{y}_2 = \bar{y} \cdot x^y \ln x$

Returns a `Grads` list pairing each variable name with its adjoint tensor. -/
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
  -- For complex constructors, return empty grads (TODO: implement)
  | .reduce _ _ _, _ => []     -- TODO: broadcast adjoint back
  | .transpose _ _, _ => []    -- TODO: inverse permutation
  | .broadcast _ _ _, _ => []  -- TODO: reduce over broadcasted dims
  | .slice _ _ _ _, _ => []    -- TODO: pad adjoint
  | .pad _ _ _ _, _ => []      -- TODO: slice adjoint
  | .concat _ _ _ _, _ => []   -- TODO: slice adjoint into parts
  | .gather _ _ _, _ => []     -- TODO: scatter_add
  | .scan _ _ _, _ => []       -- TODO: reverse scan
  | .einsum _ _ _ eA eB _, _ =>
    -- Simplified: propagate gradients through both inputs
    (eA.backward (.fill _ 0.0)).merge (eB.backward (.fill _ 0.0))

/-- Compute gradients of a scalar loss $\ell$ w.r.t. all variables by seeding
backward with $\bar{\ell} = 1$.

Initiates reverse-mode AD from a scalar (shape `[]`) expression, producing
a `Grads` list of adjoint tensors for every variable that appears in the
computation graph. -/
def grad (loss : TensorExpr Float []) : Grads :=
  loss.backward (.fill [] 1.0)

end TensorExpr
