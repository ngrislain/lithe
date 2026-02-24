/-
  Lithe/NamedTensor.lean — Type-safe named-dimension tensor wrapper

  `NamedTensor α ds` wraps `TensorExpr α ds.toShape` with named dimensions.
  Type-level enforcement: `NamedTensor Float [batch 32, ⟨784, none⟩]` and
  `NamedTensor Float [⟨32, none⟩, ⟨784, none⟩]` are different types.
-/
import Lithe.Dim
import Lithe.Scalar
import Lithe.Ops
import Lithe.Tensor
import Lithe.Env
import Lithe.Eval
import Lithe.Smart

open Shape

/-- A tensor expression annotated with named dimensions. The underlying
    `TensorExpr` uses the plain shape `ds.toShape`. -/
structure NamedTensor (α : Type) [Scalar α] (ds : DimShape) where
  expr : TensorExpr α ds.toShape

namespace NamedTensor

/-! ### Arithmetic (shape-preserving, name-enforced) -/

instance [Scalar α] : Add (NamedTensor α ds) where
  add a b := ⟨a.expr + b.expr⟩

instance [Scalar α] : Mul (NamedTensor α ds) where
  mul a b := ⟨a.expr * b.expr⟩

instance [Scalar α] : Neg (NamedTensor α ds) where
  neg a := ⟨-a.expr⟩

instance [Scalar α] : Sub (NamedTensor α ds) where
  sub a b := ⟨a.expr - b.expr⟩

/-! ### Smart constructors -/

/-- A named variable tensor. -/
def var (name : String) (ds : DimShape) [Scalar α] : NamedTensor α ds :=
  ⟨.var name ds.toShape⟩

/-- A zero tensor with named dimensions. -/
def zeros (ds : DimShape) [Scalar α] : NamedTensor α ds :=
  ⟨.fill ds.toShape 0⟩

/-- A ones tensor with named dimensions. -/
def ones (ds : DimShape) [Scalar α] : NamedTensor α ds :=
  ⟨.fill ds.toShape 1⟩

/-- A constant-filled tensor with named dimensions. -/
def fill (ds : DimShape) [Scalar α] (v : α) : NamedTensor α ds :=
  ⟨.fill ds.toShape v⟩

/-- A literal tensor with named dimensions. -/
def literal (ds : DimShape) [Scalar α] (v : Vector α ds.toShape.product)
    : NamedTensor α ds :=
  ⟨.literal ds.toShape v⟩

/-! ### Unary ops (shape-preserving → name-preserving) -/

def relu [Scalar α] (t : NamedTensor α ds) : NamedTensor α ds := ⟨Tensor.relu t.expr⟩
def exp [Scalar α] (t : NamedTensor α ds) : NamedTensor α ds := ⟨Tensor.exp t.expr⟩
def log [Scalar α] (t : NamedTensor α ds) : NamedTensor α ds := ⟨Tensor.log t.expr⟩
def sqrt [Scalar α] (t : NamedTensor α ds) : NamedTensor α ds := ⟨Tensor.sqrt t.expr⟩
def sin [Scalar α] (t : NamedTensor α ds) : NamedTensor α ds := ⟨Tensor.sin t.expr⟩
def cos [Scalar α] (t : NamedTensor α ds) : NamedTensor α ds := ⟨Tensor.cos t.expr⟩
def tanh [Scalar α] (t : NamedTensor α ds) : NamedTensor α ds := ⟨Tensor.tanh t.expr⟩
def sigmoid [Scalar α] (t : NamedTensor α ds) : NamedTensor α ds := ⟨Tensor.sigmoid t.expr⟩
def abs [Scalar α] (t : NamedTensor α ds) : NamedTensor α ds := ⟨Tensor.abs t.expr⟩
def sign [Scalar α] (t : NamedTensor α ds) : NamedTensor α ds := ⟨Tensor.sign t.expr⟩

/-! ### Scalar multiplication -/

def smul [Scalar α] (c : α) (t : NamedTensor α ds) : NamedTensor α ds :=
  ⟨.smul c t.expr⟩

/-! ### Evaluation -/

/-- Evaluate a closed named tensor expression. -/
def eval (t : NamedTensor Float ds) : Vector Float ds.toShape.product :=
  t.expr.eval

/-- Evaluate with an environment for variable bindings. -/
def evalWith (t : NamedTensor Float ds) (env : Env Float)
    : Except String (Vector Float ds.toShape.product) :=
  t.expr.evalWith env

/-! ### Named dimension operations -/

/-- Build starts for a slice: zeros everywhere except `start` at `dimIdx`. -/
private def mkStarts (ds : DimShape) (dimIdx : Nat) (start : Nat) : List Nat :=
  go ds 0
where
  go : List Dim → Nat → List Nat
    | [], _ => []
    | _ :: rest, i => (if i == dimIdx then start else 0) :: go rest (i + 1)

/-- Build sizes for a slice: original dim values except `size` at `dimIdx`. -/
private def mkSizes (ds : DimShape) (dimIdx : Nat) (size : Nat) : List Nat :=
  (ds.replaceAt dimIdx size).toShape

/-- Slice by dimension name: `t.sliceAt "batch" start size`.
    Keeps the dimension (with reduced size), preserves all other names. -/
def sliceAt [Scalar α] (t : NamedTensor α ds) (dimName : String)
    (start size : Nat) : Option (Σ ds' : DimShape, NamedTensor α ds') :=
  match ds.findDimIdx dimName with
  | none => none
  | some dimIdx =>
    let starts := mkStarts ds dimIdx start
    let sizes := mkSizes ds dimIdx size
    let s := ds.toShape
    if hlen1 : starts.length = s.length then
      if hlen2 : sizes.length = s.length then
        if hvalid : ∀ i : Fin s.length,
            List.getD starts i.val 0 + List.getD sizes i.val 0 ≤ s.get i then
          let ds' := ds.replaceAt dimIdx size
          have hsizes : sizes = ds'.toShape := rfl
          let proof : ValidSlice s starts sizes := ⟨hlen1, hlen2, hvalid⟩
          some ⟨ds', ⟨hsizes ▸ (.slice t.expr starts sizes proof)⟩⟩
        else none
      else none
    else none

private theorem removeIdx_toShape_eq (ds : DimShape) (i : Nat) (h : i < ds.length) :
    (ds.removeIdx i).toShape = ds.toShape.removeAt ⟨i, by simp [DimShape.toShape]; exact h⟩ := by
  induction ds generalizing i with
  | nil => simp at h
  | cons d rest ih =>
    cases i with
    | zero => simp [DimShape.removeIdx, DimShape.toShape, Shape.removeAt]
    | succ n =>
      simp [DimShape.removeIdx, DimShape.toShape, Shape.removeAt]
      exact ih n (by simp [List.length] at h; omega)

/-- Reduce by dimension name: `t.reduceAt "batch" .sum`.
    Removes the named dimension (rank reduction). -/
def reduceAt [Scalar α] (t : NamedTensor α ds) (dimName : String)
    (op : ReduceOp) : Option (Σ ds' : DimShape, NamedTensor α ds') :=
  match ds.findDimIdx dimName with
  | none => none
  | some dimIdx =>
    if h : dimIdx < ds.length then
      let ds' := ds.removeIdx dimIdx
      have hshape : ds'.toShape = ds.toShape.removeAt ⟨dimIdx, by
          simp [DimShape.toShape]; exact h⟩ := removeIdx_toShape_eq ds dimIdx h
      let axis : Fin ds.toShape.length := ⟨dimIdx, by
          simp [DimShape.toShape]; exact h⟩
      some ⟨ds', ⟨hshape ▸ (.reduce op axis t.expr)⟩⟩
    else none

end NamedTensor

/-! ### NamedModule -/

/-- A neural network module with named-dimension inputs and outputs. -/
structure NamedModule (α : Type) [Scalar α] (dsIn dsOut : DimShape) where
  forward : NamedTensor α dsIn → NamedTensor α dsOut
  params  : List (String × DimShape)

namespace NamedModule

/-- Sequential composition of two named modules. -/
def compose [Scalar α] (m1 : NamedModule α ds1 ds2) (m2 : NamedModule α ds2 ds3)
    : NamedModule α ds1 ds3 where
  forward x := m2.forward (m1.forward x)
  params := m1.params ++ m2.params

/-- Pipeline operator for named module composition. -/
scoped infixl:50 " |>> " => NamedModule.compose

end NamedModule
