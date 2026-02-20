/-
  Lithe/Smart.lean — Smart constructors, operator instances, derived ops
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Ops
import Lithe.Tensor

open Shape

/-! ### Typeclass instances for TensorExpr -/

instance [Scalar α] : Add (TensorExpr α s) where
  add a b := .binary .add a b

instance [Scalar α] : Mul (TensorExpr α s) where
  mul a b := .binary .mul a b

instance [Scalar α] : Neg (TensorExpr α s) where
  neg a := .unary .neg a

instance [Scalar α] : Sub (TensorExpr α s) where
  sub a b := .binary .sub a b

namespace Tensor

/-- Zero tensor of any shape. -/
def zeros [Scalar α] (s : Shape) : Tensor α s := .fill s 0

/-- Ones tensor of any shape. -/
def ones [Scalar α] (s : Shape) : Tensor α s := .fill s 1

/-! ### Einsum proofs -/

private theorem matmul_einsum_shared (m k n : Nat) :
    ∀ (ia : Fin [0, 1].length) (ib : Fin [1, 2].length),
    List.getD [0, 1] ia.val 0 = List.getD [1, 2] ib.val 0 →
    List.getD [m, k] ia.val 0 = List.getD [k, n] ib.val 0 := by
  intro ⟨ia, hia⟩ ⟨ib, hib⟩ heq
  match ia, hia, ib, hib with
  | 0, _, 0, _ => simp [List.getD] at heq  -- label 0 = 1, contradiction
  | 0, _, 1, _ => simp [List.getD] at heq  -- label 0 = 2, contradiction
  | 1, _, 0, _ => simp [List.getD] at heq ⊢  -- label 1 = 1, goal: k = k
  | 1, _, 1, _ => simp [List.getD] at heq  -- label 1 = 2, contradiction

private theorem matmul_einsum_output (m k n : Nat) :
    ∀ (o : Fin [0, 2].length),
    (∃ ia : Fin [0, 1].length,
      List.getD [0, 2] o.val 0 = List.getD [0, 1] ia.val 0 ∧
      List.getD [m, n] o.val 0 = List.getD [m, k] ia.val 0) ∨
    (∃ ib : Fin [1, 2].length,
      List.getD [0, 2] o.val 0 = List.getD [1, 2] ib.val 0 ∧
      List.getD [m, n] o.val 0 = List.getD [k, n] ib.val 0) := by
  intro ⟨o, ho⟩
  match o, ho with
  | 0, _ => left; exact ⟨⟨0, by decide⟩, by simp [List.getD]⟩
  | 1, _ => right; exact ⟨⟨1, by decide⟩, by simp [List.getD]⟩

private theorem matmul_einsum_valid (m k n : Nat) :
    IsEinsumValid [0, 1] [1, 2] [0, 2] [m, k] [k, n] [m, n] :=
  ⟨rfl, rfl, rfl, matmul_einsum_shared m k n, matmul_einsum_output m k n⟩

/-- Matrix multiply [m,k] × [k,n] → [m,n] via einsum. -/
def matmul [Scalar α] (a : TensorExpr α [m, k]) (b : TensorExpr α [k, n])
    : TensorExpr α [m, n] :=
  .einsum [0, 1] [1, 2] [0, 2] a b (matmul_einsum_valid m k n)

/-- Batch matrix multiply [b,m,k] × [b,k,n] → [b,m,n]. -/
def batchMatmul [Scalar α] (a : TensorExpr α [ba, m, k]) (b' : TensorExpr α [ba, k, n])
    : TensorExpr α [ba, m, n] :=
  .einsum [0, 1, 2] [0, 2, 3] [0, 1, 3] a b' ⟨rfl, rfl, rfl, by
    intro ⟨ia, hia⟩ ⟨ib, hib⟩ heq
    match ia, hia, ib, hib with
    | 0, _, 0, _ => simp [List.getD] at heq ⊢  -- label 0=0, ba=ba
    | 0, _, 1, _ => simp [List.getD] at heq     -- label 0=2, contradiction
    | 0, _, 2, _ => simp [List.getD] at heq     -- label 0=3, contradiction
    | 1, _, 0, _ => simp [List.getD] at heq     -- label 1=0, contradiction
    | 1, _, 1, _ => simp [List.getD] at heq     -- label 1=2, contradiction
    | 1, _, 2, _ => simp [List.getD] at heq     -- label 1=3, contradiction
    | 2, _, 0, _ => simp [List.getD] at heq     -- label 2=0, contradiction
    | 2, _, 1, _ => simp [List.getD] at heq ⊢   -- label 2=2, k=k
    | 2, _, 2, _ => simp [List.getD] at heq     -- label 2=3, contradiction
  , by
    intro ⟨o, ho⟩
    match o, ho with
    | 0, _ => left; exact ⟨⟨0, by decide⟩, by simp [List.getD]⟩
    | 1, _ => left; exact ⟨⟨1, by decide⟩, by simp [List.getD]⟩
    | 2, _ => right; exact ⟨⟨2, by decide⟩, by simp [List.getD]⟩
  ⟩

/-- Outer product [m] × [n] → [m,n]. -/
def outer [Scalar α] (a : TensorExpr α [m]) (b : TensorExpr α [n])
    : TensorExpr α [m, n] :=
  .einsum [0] [1] [0, 1] a b ⟨rfl, rfl, rfl, by
    intro ⟨ia, hia⟩ ⟨ib, hib⟩ heq
    match ia, hia, ib, hib with
    | 0, _, 0, _ => simp [List.getD] at heq  -- label 0=1, contradiction
  , by
    intro ⟨o, ho⟩
    match o, ho with
    | 0, _ => left; exact ⟨⟨0, by decide⟩, by simp [List.getD]⟩
    | 1, _ => right; exact ⟨⟨0, by decide⟩, by simp [List.getD]⟩
  ⟩

/-- Dot product [n] × [n] → []. -/
def dot [Scalar α] (a b : TensorExpr α [n]) : TensorExpr α [] :=
  .einsum [0] [0] [] a b ⟨rfl, rfl, rfl, by
    intro ⟨ia, hia⟩ ⟨ib, hib⟩ heq
    match ia, hia, ib, hib with
    | 0, _, 0, _ => simp [List.getD] at heq ⊢  -- label 0=0, n=n
  , by intro ⟨o, ho⟩; simp [List.length] at ho
  ⟩

/-! ### Reduction-based ops -/

def cumsum [Scalar α] (axis : Fin s.length) (t : TensorExpr α s) : TensorExpr α s :=
  .scan .sum axis t

def cumprod [Scalar α] (axis : Fin s.length) (t : TensorExpr α s) : TensorExpr α s :=
  .scan .prod axis t

def mean (axis : Fin s.length) (t : TensorExpr Float s) : TensorExpr Float (s.removeAt axis) :=
  let n := List.getD s axis.val 1
  .smul (1.0 / n.toFloat) (.reduce .sum axis t)

/-! ### Unary ops as functions -/

def exp [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .exp t
def log [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .log t
def sqrt [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .sqrt t
def sin [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .sin t
def cos [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .cos t
def tanh [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .tanh t
def sigmoid [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .sigmoid t
def relu [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .relu t
def abs [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .abs t
def sign [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .sign t

/-! ### Transpose shortcuts -/

/-- Permutation vector swapping two axes for a 2D shape. -/
private def perm2D (m n : Nat) : Vector (Fin [m, n].length) [m, n].length :=
  ⟨#[⟨1, by show 1 < 2; omega⟩, ⟨0, by show 0 < 2; omega⟩], rfl⟩

def transpose2D [Scalar α] (t : TensorExpr α [m, n]) : TensorExpr α [n, m] :=
  .transpose t (perm2D m n)

end Tensor
