/-
  Lithe/Env.lean — Environment for variable bindings
-/
import Lithe.Shape

/-- Runtime tensor data: a shape $s$ paired with a flat `Vector` of
    $|s| = \prod_i d_i$ elements. -/
structure TensorData (α : Type) where
  shape : Shape
  data  : Vector α shape.product

/-- An environment $\Gamma : \text{String} \to \text{TensorData}\;\alpha$ mapping variable names
    to their runtime tensor values. -/
abbrev Env (α : Type) := List (String × TensorData α)

namespace Env

/-- The empty environment $\Gamma = \varnothing$. -/
def empty : Env α := []

/-- Look up variable by name in $\Gamma$; returns `some v` when the stored shape product matches
    the expected shape product (allowing safe cast), `none` otherwise. -/
def lookup (env : Env α) (name : String) (s : Shape) : Option (Vector α s.product) :=
  match env.find? (·.1 == name) with
  | none => none
  | some (_, td) =>
    if h : td.shape.product = s.product then
      some (h ▸ td.data)
    else
      none

end Env
