/-
  Lithe/Module.lean — Module abstraction for composable layers
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Tensor
import Lithe.Env
import Lithe.Eval
import Lithe.Smart

/-- A neural network module mapping input tensors of shape $s_{\text{in}}$ to output tensors
    of shape $s_{\text{out}}$, with declared learnable parameters. -/
structure Module (α : Type) [Scalar α] (sIn sOut : Shape) where
  /-- Build the forward computation graph: $T_{\text{out}} = f(T_{\text{in}})$. -/
  forward : TensorExpr α sIn → TensorExpr α sOut
  /-- Declared learnable parameters as $(name, shape)$ pairs. -/
  params  : List (String × Shape)

namespace Module

/-- Sequential composition of two modules: $(m_2 \circ m_1)(x) = m_2(m_1(x))$.
    Parameters from both modules are concatenated. -/
def compose [Scalar α] (m1 : Module α s1 s2) (m2 : Module α s2 s3) : Module α s1 s3 where
  forward x := m2.forward (m1.forward x)
  params := m1.params ++ m2.params

/-- Execute a module: evaluate the forward graph given environment $\Gamma$ and
    concrete input data. Returns the output tensor as a flat vector. -/
def run (m : Module Float sIn sOut) (env : Env Float) (input : Vector Float sIn.product)
    : Except String (Vector Float sOut.product) :=
  let inputExpr := TensorExpr.literal sIn input
  (m.forward inputExpr).evalWith env

/-- Identity module (no-op, no parameters). -/
def identity [Scalar α] (s : Shape) : Module α s s where
  forward x := x
  params := []

/-- Pipeline composition operator: `m1 |>> m2 = compose m1 m2`. -/
scoped infixl:50 " |>> " => Module.compose

/-! ### Example layers -/

/-- Fully connected linear layer: $y = xW + b$ where $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$
    and $b \in \mathbb{R}^{d_{\text{out}}}$. The input is reshaped to a row vector for matrix
    multiplication, then the bias is added elementwise. -/
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

/-- ReLU activation layer (no learnable parameters): $y_i = \max(0, x_i)$. -/
def reluLayer [Scalar α] (s : Shape) : Module α s s where
  forward x := Tensor.relu x
  params := []

/-- Two-layer MLP: $\operatorname{linear}_2(\operatorname{relu}(\operatorname{linear}_1(x)))$.
    Composes a linear layer, a ReLU activation, and a second linear layer. -/
def mlp (inDim hiddenDim outDim : Nat) (name : String) : Module Float [inDim] [outDim] :=
  linear inDim hiddenDim (name ++ ".layer1")
    |>> (reluLayer [hiddenDim] : Module Float _ _)
    |>> linear hiddenDim outDim (name ++ ".layer2")

end Module
