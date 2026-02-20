/-
  Lithe/Scalar.lean — Scalar typeclass and algebraic laws
-/

class Scalar (α : Type) extends Add α, Mul α, Neg α, Zero α, One α where
  sub (a b : α) : α := a + (-b)

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

/-- Extended scalar operations for floating-point-like types. -/
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

/-- Division for scalars. -/
class ScalarDiv (α : Type) [Scalar α] where
  div : α → α → α

/-- Ordering for scalars. -/
class ScalarOrd (α : Type) [Scalar α] where
  le : α → α → Bool
  lt : α → α → Bool
  max : α → α → α
  min : α → α → α

-- Float instance
instance : Scalar Float where
  add := (· + ·)
  mul := (· * ·)
  neg := (- ·)
  zero := 0.0
  one := 1.0

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

instance : ScalarDiv Float where
  div := (· / ·)

instance : ScalarOrd Float where
  le := (· ≤ ·)
  lt := (· < ·)
  max := fun a b => if a ≥ b then a else b
  min := fun a b => if a ≤ b then a else b

-- Int instance with ScalarLaws
instance : Scalar Int where
  add := (· + ·)
  mul := (· * ·)
  neg := (- ·)
  zero := 0
  one := 1

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
