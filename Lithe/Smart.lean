/-
  Lithe/Smart.lean — Smart constructors, operator instances, derived ops
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Ops
import Lithe.Tensor

open Shape

/-! ### Typeclass instances for TensorExpr -/

/-- Pointwise addition: $(T_1 + T_2)_\mathbf{i} = (T_1)_\mathbf{i} + (T_2)_\mathbf{i}$. -/
instance [Scalar α] : Add (TensorExpr α s) where
  add a b := .binary .add a b

/-- Pointwise (Hadamard) product: $(T_1 \cdot T_2)_\mathbf{i} = (T_1)_\mathbf{i} \cdot (T_2)_\mathbf{i}$. -/
instance [Scalar α] : Mul (TensorExpr α s) where
  mul a b := .binary .mul a b

/-- Pointwise negation: $(-T)_\mathbf{i} = -(T_\mathbf{i})$. -/
instance [Scalar α] : Neg (TensorExpr α s) where
  neg a := .unary .neg a

/-- Pointwise subtraction: $(T_1 - T_2)_\mathbf{i} = (T_1)_\mathbf{i} - (T_2)_\mathbf{i}$. -/
instance [Scalar α] : Sub (TensorExpr α s) where
  sub a b := .binary .sub a b

namespace Tensor

/-- Zero tensor $\mathbf{0}_s$ with every element equal to $0$. -/
def zeros [Scalar α] (s : Shape) : Tensor α s := .fill s 0

/-- Ones tensor $\mathbf{1}_s$ with every element equal to $1$. -/
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

theorem matmul_einsum_valid (m k n : Nat) :
    IsEinsumValid [0, 1] [1, 2] [0, 2] [m, k] [k, n] [m, n] :=
  ⟨rfl, rfl, rfl, matmul_einsum_shared m k n, matmul_einsum_output m k n⟩

/-- Matrix multiplication via einsum: $C_{mn} = \sum_k A_{mk} B_{kn}$, encoding $ik, kj \to ij$. -/
def matmul [Scalar α] (a : TensorExpr α [m, k]) (b : TensorExpr α [k, n])
    : TensorExpr α [m, n] :=
  .einsum [0, 1] [1, 2] [0, 2] a b (matmul_einsum_valid m k n)

/-- Batched matrix multiplication: $C_{bmn} = \sum_k A_{bmk} B_{bkn}$, encoding $bik, bkj \to bij$. -/
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

/-- Outer product: $C_{ij} = a_i \cdot b_j$, encoding $i, j \to ij$. -/
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

/-- Dot (inner) product: $c = \sum_i a_i \cdot b_i$, encoding $i, i \to \varnothing$. -/
def dot [Scalar α] (a b : TensorExpr α [n]) : TensorExpr α [] :=
  .einsum [0] [0] [] a b ⟨rfl, rfl, rfl, by
    intro ⟨ia, hia⟩ ⟨ib, hib⟩ heq
    match ia, hia, ib, hib with
    | 0, _, 0, _ => simp [List.getD] at heq ⊢  -- label 0=0, n=n
  , by intro ⟨o, ho⟩; simp [List.length] at ho
  ⟩

/-! ### Reduction-based ops -/

/-- Cumulative sum along axis $k$: $y_{\ldots,j,\ldots} = \sum_{i=0}^{j} x_{\ldots,i,\ldots}$. -/
def cumsum [Scalar α] (axis : Fin s.length) (t : TensorExpr α s) : TensorExpr α s :=
  .scan .sum axis t

/-- Cumulative product along axis $k$: $y_{\ldots,j,\ldots} = \prod_{i=0}^{j} x_{\ldots,i,\ldots}$. -/
def cumprod [Scalar α] (axis : Fin s.length) (t : TensorExpr α s) : TensorExpr α s :=
  .scan .prod axis t

/-- Mean along axis $k$: $\bar{x} = \frac{1}{d_k} \sum_{i=0}^{d_k-1} x_{\ldots,i,\ldots}$. -/
def mean (axis : Fin s.length) (t : TensorExpr Float s) : TensorExpr Float (s.removeAt axis) :=
  let n := List.getD s axis.val 1
  .smul (1.0 / n.toFloat) (.reduce .sum axis t)

/-! ### Unary ops as functions -/

/-- Elementwise exponential: $(e^T)_\mathbf{i} = e^{T_\mathbf{i}}$. -/
def exp [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .exp t
/-- Elementwise natural logarithm: $(\ln T)_\mathbf{i} = \ln(T_\mathbf{i})$. -/
def log [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .log t
/-- Elementwise square root: $(\sqrt{T})_\mathbf{i} = \sqrt{T_\mathbf{i}}$. -/
def sqrt [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .sqrt t
/-- Elementwise sine: $(\sin T)_\mathbf{i} = \sin(T_\mathbf{i})$. -/
def sin [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .sin t
/-- Elementwise cosine: $(\cos T)_\mathbf{i} = \cos(T_\mathbf{i})$. -/
def cos [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .cos t
/-- Elementwise hyperbolic tangent: $(\tanh T)_\mathbf{i} = \tanh(T_\mathbf{i})$. -/
def tanh [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .tanh t
/-- Elementwise sigmoid: $(\sigma(T))_\mathbf{i} = \frac{1}{1 + e^{-T_\mathbf{i}}}$. -/
def sigmoid [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .sigmoid t
/-- Elementwise ReLU: $(\operatorname{relu}(T))_\mathbf{i} = \max(0, T_\mathbf{i})$. -/
def relu [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .relu t
/-- Elementwise absolute value: $(|T|)_\mathbf{i} = |T_\mathbf{i}|$. -/
def abs [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .abs t
/-- Elementwise sign function: $(\operatorname{sign}(T))_\mathbf{i} = \operatorname{sign}(T_\mathbf{i})$. -/
def sign [Scalar α] (t : TensorExpr α s) : TensorExpr α s := .unary .sign t

/-! ### Transpose shortcuts -/

/-- Permutation vector swapping two axes for a 2D shape. -/
private def perm2D (m n : Nat) : Vector (Fin [m, n].length) [m, n].length :=
  ⟨#[⟨1, by show 1 < 2; omega⟩, ⟨0, by show 0 < 2; omega⟩], rfl⟩

/-- Transpose a 2D tensor: $T'_{ji} = T_{ij}$, swapping axes $0 \leftrightarrow 1$. -/
def transpose2D [Scalar α] (t : TensorExpr α [m, n]) : TensorExpr α [n, m] :=
  .transpose t (perm2D m n)

/-! ### Convenient slicing -/

/-- Slice with `(start, size)` pairs — proof auto-resolved by `decide`.
    Usage: `t.sliceWith [(1, 2), (0, 3), (0, 2)]` -/
def sliceWith [Scalar α] (t : TensorExpr α s) (ranges : List (Nat × Nat))
    (h : ValidSlice s (ranges.map Prod.fst) (ranges.map Prod.snd) := by decide)
    : TensorExpr α (ranges.map Prod.snd) :=
  .slice t (ranges.map Prod.fst) (ranges.map Prod.snd) h

/-- Index along the first axis (rank reduction): `t.head i` selects index `i`
    from the leading dimension, returning a tensor of the remaining shape.
    Implemented as a slice of width 1 followed by reshape. -/
def head [Scalar α] (t : TensorExpr α (d :: rest)) (i : Nat)
    (hi : i < d := by omega)
    : TensorExpr α rest :=
  let starts := i :: rest.map (fun _ => 0)
  let sizes := 1 :: rest
  have hlen1 : starts.length = (d :: rest).length := by simp [starts, List.length_map]
  have hlen2 : sizes.length = (d :: rest).length := by simp [sizes]
  have hvalid : ∀ j : Fin (d :: rest).length,
      List.getD starts j.val 0 + List.getD sizes j.val 0 ≤ (d :: rest).get j := by
    intro ⟨j, hj⟩
    match j, hj with
    | 0, _ => simp [starts, sizes, List.getD]; omega
    | j + 1, hj =>
      simp only [starts, sizes, List.getD, List.get]
      simp only [List.length_cons] at hj
      have hj' : j < rest.length := by omega
      simp [List.getElem?_eq_getElem hj']
  have hprod : product (1 :: rest) = product rest := by
    simp [product, Nat.one_mul]
  let sliced := TensorExpr.slice t starts sizes ⟨hlen1, hlen2, hvalid⟩
  TensorExpr.reshape sliced hprod

end Tensor
