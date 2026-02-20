/-
  Lithe/Env.lean — Environment for variable bindings
-/
import Lithe.Shape

/-- Runtime tensor data: a shape paired with flat data of matching size. -/
structure TensorData (α : Type) where
  shape : Shape
  data  : Vector α shape.product

/-- An environment maps variable names to tensor data. -/
abbrev Env (α : Type) := List (String × TensorData α)

namespace Env

/-- Empty environment. -/
def empty : Env α := []

/-- Lookup a variable, returning its data cast to the expected shape.
    Returns none if not found or shape mismatch. -/
def lookup (env : Env α) (name : String) (s : Shape) : Option (Vector α s.product) :=
  match env.find? (·.1 == name) with
  | none => none
  | some (_, td) =>
    if h : td.shape.product = s.product then
      some (h ▸ td.data)
    else
      none

end Env
