/-
  Lithe/Scalar.lean — Scalar typeclass and algebraic laws
-/

/-- Algebraic scalar typeclass providing a commutative ring structure with
$(\alpha, +, \cdot, 0, 1, -)$. Subtraction defaults to $a - b := a + (-b)$. -/
class Scalar (α : Type) extends Add α, Mul α, Neg α, Zero α, One α where
  sub (a b : α) : α := a + (-b)

/-- Proof bundle certifying that `Scalar α` satisfies the commutative ring axioms
including commutativity ($a + b = b + a$, $a \cdot b = b \cdot a$), associativity,
identity elements, distributivity ($a(b+c) = ab + ac$), and additive inverse
($a + (-a) = 0$). -/
class ScalarLaws (α : Type) [Scalar α] : Prop where
  add_comm  : ∀ (a b : α), a + b = b + a
  add_assoc : ∀ (a b c : α), a + b + c = a + (b + c)
  mul_comm  : ∀ (a b : α), a * b = b * a
  mul_assoc : ∀ (a b c : α), a * b * c = a * (b * c)
  add_zero  : ∀ (a : α), a + 0 = a
  zero_add  : ∀ (a : α), 0 + a = a
  mul_one   : ∀ (a : α), a * 1 = a
  one_mul   : ∀ (a : α), 1 * a = a
  mul_add   : ∀ (a b c : α), a * (b + c) = a * b + a * c
  add_neg   : ∀ (a : α), a + (-a) = 0

/-- Extended transcendental operations for floating-point-like scalars:
$\exp, \ln, \sqrt{\cdot}, \sin, \cos, \tanh, |\cdot|, \operatorname{sign}, x^y$. -/
class ScalarFloat (α : Type) [Scalar α] where
  exp  : α → α
  log  : α → α
  sqrt : α → α
  sin  : α → α
  cos  : α → α
  tanh : α → α
  abs  : α → α
  sign : α → α
  pow  : α → α → α

/-- Division operation $a \div b$ for scalars. -/
class ScalarDiv (α : Type) [Scalar α] where
  div : α → α → α

/-- Total ordering operations ($\le$, $<$, $\max$, $\min$) for scalars. -/
class ScalarOrd (α : Type) [Scalar α] where
  le : α → α → Bool
  lt : α → α → Bool
  max : α → α → α
  min : α → α → α

/-- `Scalar Float` instance with IEEE 754 arithmetic. -/
instance : Scalar Float where
  add := (· + ·)
  mul := (· * ·)
  neg := (- ·)
  zero := 0.0
  one := 1.0

/-- `ScalarFloat Float` backed by IEEE 754 intrinsics;
$$\operatorname{sign}(x) = \begin{cases}1 & x > 0\\-1 & x < 0\\0 & x = 0\end{cases}$$. -/
instance : ScalarFloat Float where
  exp  := Float.exp
  log  := Float.log
  sqrt := Float.sqrt
  sin  := Float.sin
  cos  := Float.cos
  tanh := Float.tanh
  abs  := Float.abs
  sign := fun x => if x > 0 then 1.0 else if x < 0 then -1.0 else 0.0
  pow  := (· ^ ·)

/-- IEEE 754 floating-point division. -/
instance : ScalarDiv Float where
  div := (· / ·)

/-- IEEE 754 floating-point ordering. -/
instance : ScalarOrd Float where
  le := (· ≤ ·)
  lt := (· < ·)
  max := fun a b => if a ≥ b then a else b
  min := fun a b => if a ≤ b then a else b

/-- `Scalar Int` with exact integer arithmetic. -/
instance : Scalar Int where
  add := (· + ·)
  mul := (· * ·)
  neg := (- ·)
  zero := 0
  one := 1

/-- Proof that `Int` satisfies all `ScalarLaws` ring axioms
(verified by Lean's standard library). -/
instance : ScalarLaws Int where
  add_comm  := Int.add_comm
  add_assoc := Int.add_assoc
  mul_comm  := Int.mul_comm
  mul_assoc := Int.mul_assoc
  add_zero  := Int.add_zero
  zero_add  := Int.zero_add
  mul_one   := Int.mul_one
  one_mul   := Int.one_mul
  mul_add   := Int.mul_add
  add_neg   := Int.add_right_neg
