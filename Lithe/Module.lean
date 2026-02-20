/-
  Lithe/Module.lean — Module abstraction for composable layers
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Tensor
import Lithe.Env
import Lithe.Eval
import Lithe.Smart

/-- A module takes an input tensor expression and produces an output expression.
    It also declares its learnable parameters. -/
structure Module (α : Type) [Scalar α] (sIn sOut : Shape) where
  /-- Build the forward computation graph. -/
  forward : TensorExpr α sIn → TensorExpr α sOut
  /-- Declared parameter names and shapes. -/
  params  : List (String × Shape)

namespace Module

/-- Compose two modules sequentially. -/
def compose [Scalar α] (m1 : Module α s1 s2) (m2 : Module α s2 s3) : Module α s1 s3 where
  forward x := m2.forward (m1.forward x)
  params := m1.params ++ m2.params

/-- Run a module: given an env and input data, produce output. -/
def run (m : Module Float sIn sOut) (env : Env Float) (input : Vector Float sIn.product)
    : Except String (Vector Float sOut.product) :=
  let inputExpr := TensorExpr.literal sIn input
  (m.forward inputExpr).evalWith env

/-! ### Example layers -/

/-- Linear layer: y = x @ W + b, where x : [inDim], W : [inDim, outDim], b : [outDim]. -/
def linear (inDim outDim : Nat) (name : String) : Module Float [inDim] [outDim] where
  forward x :=
    let w := TensorExpr.var (name ++ ".weight") [inDim, outDim]
    let b := TensorExpr.var (name ++ ".bias") [outDim]
    -- x : [inDim] → reshape to [1, inDim] → matmul with W → reshape to [outDim] → add b
    let x1 := TensorExpr.reshape (s₂ := [1, inDim]) x (by simp [Shape.product, Nat.one_mul])
    let y := Tensor.matmul x1 w  -- [1, outDim]
    let y1 := TensorExpr.reshape (s₂ := [outDim]) y (by simp [Shape.product, Nat.one_mul, Nat.mul_one])
    y1 + b
  params := [(name ++ ".weight", [inDim, outDim]), (name ++ ".bias", [outDim])]

/-- ReLU activation (no parameters). -/
def reluLayer [Scalar α] (s : Shape) : Module α s s where
  forward x := Tensor.relu x
  params := []

/-- Simple MLP: linear → relu → linear. -/
def mlp (inDim hiddenDim outDim : Nat) (name : String) : Module Float [inDim] [outDim] :=
  compose
    (compose (linear inDim hiddenDim (name ++ ".layer1")) (reluLayer [hiddenDim]))
    (linear hiddenDim outDim (name ++ ".layer2"))

end Module
