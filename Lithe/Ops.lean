/-
  Lithe/Ops.lean — Operation enums for symbolic tensor expressions
-/

/-- Unary elementwise operations. -/
inductive UnaryOp where
  | neg | abs | exp | log | sqrt | sin | cos | tanh | sigmoid | sign | relu
  deriving Repr, BEq, DecidableEq

/-- Binary elementwise operations. -/
inductive BinaryOp where
  | add | mul | sub | div | pow | max | min
  deriving Repr, BEq, DecidableEq

/-- Reduction operations. -/
inductive ReduceOp where
  | sum | prod | max | min
  deriving Repr, BEq, DecidableEq

private def floatMax (a b : Float) : Float := if a ≥ b then a else b
private def floatMin (a b : Float) : Float := if a ≤ b then a else b

namespace UnaryOp

/-- Interpret a unary op on Float scalars. -/
def evalFloat : UnaryOp → Float → Float
  | .neg, x     => -x
  | .abs, x     => x.abs
  | .exp, x     => x.exp
  | .log, x     => x.log
  | .sqrt, x    => x.sqrt
  | .sin, x     => x.sin
  | .cos, x     => x.cos
  | .tanh, x    => Float.tanh x
  | .sigmoid, x => 1.0 / (1.0 + (-x).exp)
  | .sign, x    => if x > 0 then 1.0 else if x < 0 then -1.0 else 0.0
  | .relu, x    => if x > 0 then x else 0.0

/-- Human-readable name. -/
def name : UnaryOp → String
  | .neg => "neg" | .abs => "abs" | .exp => "exp" | .log => "log"
  | .sqrt => "sqrt" | .sin => "sin" | .cos => "cos" | .tanh => "tanh"
  | .sigmoid => "sigmoid" | .sign => "sign" | .relu => "relu"

end UnaryOp

namespace BinaryOp

/-- Interpret a binary op on Float scalars. -/
def evalFloat : BinaryOp → Float → Float → Float
  | .add, a, b => a + b
  | .mul, a, b => a * b
  | .sub, a, b => a - b
  | .div, a, b => a / b
  | .pow, a, b => a ^ b
  | .max, a, b => floatMax a b
  | .min, a, b => floatMin a b

/-- Human-readable name. -/
def name : BinaryOp → String
  | .add => "add" | .mul => "mul" | .sub => "sub" | .div => "div"
  | .pow => "pow" | .max => "max" | .min => "min"

end BinaryOp

namespace ReduceOp

/-- Identity element for a reduce op on Float. -/
def identityFloat : ReduceOp → Float
  | .sum  => 0.0
  | .prod => 1.0
  | .max  => -1.0e38   -- large negative
  | .min  => 1.0e38    -- large positive

/-- Combine step for a reduce op on Float. -/
def combineFloat : ReduceOp → Float → Float → Float
  | .sum,  a, b => a + b
  | .prod, a, b => a * b
  | .max,  a, b => floatMax a b
  | .min,  a, b => floatMin a b

/-- Human-readable name. -/
def name : ReduceOp → String
  | .sum => "sum" | .prod => "prod" | .max => "max" | .min => "min"

end ReduceOp
