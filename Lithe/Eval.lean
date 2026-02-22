/-
  Lithe/Eval.lean — Reference evaluator: TensorExpr α s → Vector α s.product
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Ops
import Lithe.Index
import Lithe.Tensor
import Lithe.Env

open Lithe

namespace Lithe

/-- Theorem: $|[m, k]| = m \cdot k$.

The shape product of a two-element shape equals the ordinary product of its dimensions. -/
theorem shape_product_pair (m k : Nat) : Shape.product [m, k] = m * k := by
  simp [Shape.product, Nat.mul_one]

/-! ### Evaluation helpers for each constructor -/

/-- Apply a unary operation pointwise: $v'_i = f(v_i)$.

Maps the `Float` interpreter of the given `UnaryOp` over every element. -/
def evalUnaryVec (op : UnaryOp) (v : Vector Float n) : Vector Float n :=
  v.map op.evalFloat

/-- Apply a binary operation pointwise: $v'_i = g(v^{(1)}_i, v^{(2)}_i)$.

Zips two vectors through the `Float` interpreter of the given `BinaryOp`. -/
def evalBinaryVec (op : BinaryOp) (v₁ v₂ : Vector Float n) : Vector Float n :=
  v₁.zipWith op.evalFloat v₂

/-- Elementwise conditional: $v'_i = \begin{cases} t_i & c_i \neq 0 \\ f_i & c_i = 0 \end{cases}$.

Selects from `ifTrue` where `cond` is nonzero, from `ifFalse` otherwise. -/
def evalSelectVec (cond ifTrue ifFalse : Vector Float n) : Vector Float n :=
  Vector.ofFn fun i => if cond[i] != 0.0 then ifTrue[i] else ifFalse[i]

/-- Broadcast from shape $s_1$ to $s_2$: dimensions of size $1$ are replicated.

Each output position maps back to an input position by clamping axes of size $1$
to index $0$, thereby "stretching" singleton dimensions to match the target shape. -/
def broadcastVec (s₁ s₂ : Shape) (v : Vector Float s₁.product) : Vector Float s₂.product :=
  Vector.ofFn fun outIdx =>
    let outMulti := linearToMulti s₂ outIdx.val
    let inMulti := List.zipWith (fun d₁ o => if d₁ == 1 then 0 else o) s₁ outMulti
    let inLin := multiToLinear s₁ inMulti
    if h : inLin < s₁.product then v[inLin] else 0.0

/-- Extract a slice: $v'_\mathbf{i} = v_{\mathbf{start} + \mathbf{i}}$ for $\mathbf{i} \in [0, \mathbf{size})$.

Copies a contiguous sub-block of the input tensor, offset by `starts` with extent `sizes`. -/
def sliceVec (s : Shape) (starts sizes : List Nat) (v : Vector Float s.product)
    : Vector Float (Shape.product sizes) :=
  Vector.ofFn fun outIdx =>
    let outMulti := linearToMulti sizes outIdx.val
    let inMulti := List.zipWith (· + ·) starts outMulti
    let inLin := multiToLinear s inMulti
    if h : inLin < s.product then v[inLin] else 0.0

/-- Pad a tensor with a fill value, adding $\operatorname{lo}_k$ and $\operatorname{hi}_k$
elements along each axis $k$.

Output positions that fall within the padding region are filled with `fillVal`;
those inside the original extent are copied from the input. -/
def padVec (s : Shape) (padding : List (Nat × Nat)) (fillVal : Float)
    (v : Vector Float s.product) : Vector Float (Shape.padShape s padding).product :=
  let outShape := Shape.padShape s padding
  Vector.ofFn fun outIdx =>
    let outMulti := linearToMulti outShape outIdx.val
    -- Check if within valid region and compute input index
    match goCheck s padding outMulti with
    | some inMulti =>
      let inLin := multiToLinear s inMulti
      if h : inLin < s.product then v[inLin] else fillVal
    | none => fillVal
where
  goCheck : List Nat → List (Nat × Nat) → List Nat → Option (List Nat)
    | [], _, _ => some []
    | _, [], _ => some []
    | _, _, [] => some []
    | d :: ds, (lo, _) :: ps, o :: os =>
      if o < lo || o >= lo + d then none
      else match goCheck ds ps os with
        | some rest => some ((o - lo) :: rest)
        | none => none

/-- Concatenate two tensors along axis $k$.

The output shape matches both inputs except along axis $k$, where the
dimension equals $d^{(1)}_k + d^{(2)}_k$. Output indices with axis coordinate
below $d^{(1)}_k$ read from $v_1$; the rest read from $v_2$ (shifted). -/
def concatVec (s₁ s₂ : Shape) (axis : Fin s₁.length) (v₁ : Vector Float s₁.product)
    (v₂ : Vector Float s₂.product) : Vector Float (Shape.concatShape s₁ s₂ axis).product :=
  let outShape := Shape.concatShape s₁ s₂ axis
  Vector.ofFn fun outIdx =>
    let outMulti := linearToMulti outShape outIdx.val
    let axisIdx := List.getD outMulti axis.val 0
    let dim₁ := List.getD s₁ axis.val 0
    if axisIdx < dim₁ then
      let inLin := multiToLinear s₁ outMulti
      if h : inLin < s₁.product then v₁[inLin] else 0.0
    else
      let inMulti := mapAt outMulti axis.val (· - dim₁)
      let inLin := multiToLinear s₂ inMulti
      if h : inLin < s₂.product then v₂[inLin] else 0.0
where
  mapAt (l : List Nat) (pos : Nat) (f : Nat → Nat) : List Nat :=
    match l, pos with
    | [], _ => []
    | x :: xs, 0 => f x :: xs
    | x :: xs, n + 1 => x :: mapAt xs n f

/-- Gather elements along axis $k$ using an index vector.

The output has the same shape as the input except axis $k$ is replaced by
the number of indices. For each output position, the axis-$k$ coordinate is
looked up in the `indices` vector to determine the source position. -/
def gatherVec (s : Shape) (axis : Fin s.length) (indices : Vector Nat numIdx)
    (v : Vector Float s.product) : Vector Float (Shape.gatherShape s axis numIdx).product :=
  let outShape := Shape.gatherShape s axis numIdx
  Vector.ofFn fun outIdx =>
    let outMulti := linearToMulti outShape outIdx.val
    let gatherIdx := List.getD outMulti axis.val 0
    let srcIdx := if h : gatherIdx < numIdx then indices[gatherIdx] else 0
    let inMulti := setAt outMulti axis.val srcIdx
    let inLin := multiToLinear s inMulti
    if h : inLin < s.product then v[inLin] else 0.0
where
  setAt (l : List Nat) (pos : Nat) (val : Nat) : List Nat :=
    match l, pos with
    | [], _ => []
    | _ :: xs, 0 => val :: xs
    | x :: xs, n + 1 => x :: setAt xs n val

/-- Permute dimensions according to $\pi$: output at multi-index $\mathbf{j}$ reads
input at $\pi^{-1}(\mathbf{j})$.

Implements a generalized transpose by building the input multi-index from the
output multi-index using the inverse of the permutation `perm`. -/
def permuteVec (s : Shape) (perm : Vector (Fin s.length) s.length)
    (v : Vector Float s.product) : Vector Float (Shape.permuteShape s perm).product :=
  let outShape := Shape.permuteShape s perm
  Vector.ofFn fun outIdx =>
    let outMulti := linearToMulti outShape outIdx.val
    -- Build input multi-index: input[perm[i]] = outMulti[i]
    let inMulti := buildInMulti s.length perm.toList outMulti (List.replicate s.length 0)
    let inLin := multiToLinear s inMulti
    if h : inLin < s.product then v[inLin] else 0.0
where
  buildInMulti (n : Nat) (perm : List (Fin n)) (outMulti base : List Nat) : List Nat :=
    match perm, outMulti with
    | [], _ => base
    | _, [] => base
    | p :: ps, o :: os => buildInMulti n ps os (setAt base p.val o)
  setAt (l : List Nat) (pos val : Nat) : List Nat :=
    match l, pos with
    | [], _ => []
    | _ :: xs, 0 => val :: xs
    | x :: xs, n + 1 => x :: setAt xs n val

/-- Reduce along axis $k$ with operation $\oplus$:
$y_{\ldots} = \bigoplus_{i=0}^{d_k-1} x_{\ldots,i,\ldots}$.

The output shape is the input shape with axis $k$ removed. For each output
position, the operation folds over all slices along the reduced axis. -/
def reduceVec (op : ReduceOp) (s : Shape) (axis : Fin s.length)
    (v : Vector Float s.product) : Vector Float (s.removeAt axis).product :=
  let outShape := s.removeAt axis
  let axisSize := List.getD s axis.val 1
  Vector.ofFn fun outIdx =>
    let outMulti := linearToMulti outShape outIdx.val
    -- Insert axis dimension back at position
    let baseMulti := insertAt outMulti axis.val 0
    Id.run do
      let mut acc := op.identityFloat
      for k in [:axisSize] do
        let inMulti := setAt baseMulti axis.val k
        let inLin := multiToLinear s inMulti
        if h : inLin < s.product then
          acc := op.combineFloat acc v[inLin]
      return acc
where
  insertAt (l : List Nat) (pos val : Nat) : List Nat :=
    match l, pos with
    | xs, 0 => val :: xs
    | [], _ => [val]
    | x :: xs, n + 1 => x :: insertAt xs n val
  setAt (l : List Nat) (pos val : Nat) : List Nat :=
    match l, pos with
    | [], _ => []
    | _ :: xs, 0 => val :: xs
    | x :: xs, n + 1 => x :: setAt xs n val

/-- Cumulative scan along axis $k$:
$y_{\ldots,j,\ldots} = \bigoplus_{i=0}^{j} x_{\ldots,i,\ldots}$.

The output shape equals the input shape. Each element at axis position $j$
is the running fold of all elements at positions $0, \ldots, j$. -/
def scanVec (op : ReduceOp) (s : Shape) (axis : Fin s.length)
    (v : Vector Float s.product) : Vector Float s.product :=
  let axisSize := List.getD s axis.val 1
  Vector.ofFn fun outIdx =>
    let outMulti := linearToMulti s outIdx.val
    let pos := List.getD outMulti axis.val 0
    Id.run do
      let mut acc := op.identityFloat
      for k in [:pos + 1] do
        let inMulti := setAt outMulti axis.val k
        let inLin := multiToLinear s inMulti
        if h : inLin < s.product then
          acc := op.combineFloat acc v[inLin]
      return acc
where
  setAt (l : List Nat) (pos val : Nat) : List Nat :=
    match l, pos with
    | [], _ => []
    | _ :: xs, 0 => val :: xs
    | x :: xs, n + 1 => x :: setAt xs n val

/-- Find the first index of an element in a list. -/
private def findIdx [BEq α] (l : List α) (x : α) : Option Nat :=
  go l x 0
where
  go [BEq α] : List α → α → Nat → Option Nat
    | [], _, _ => none
    | y :: ys, x, i => if x == y then some i else go ys x (i + 1)

/-- Einstein summation: $C_\mathbf{o} = \sum_\mathbf{c} A_{\mathbf{a}(\mathbf{o},\mathbf{c})} \cdot B_{\mathbf{b}(\mathbf{o},\mathbf{c})}$
where $\mathbf{c}$ ranges over contracted labels.

Each output element is computed by iterating over all combinations of the
contracted (summed-over) indices, multiplying the corresponding elements
of $A$ and $B$, and accumulating the sum. Subscript lists `subsA`, `subsB`,
`subsOut` specify which labels appear in each operand and the output. -/
def einsumVec (subsA subsB subsOut : List Nat)
    (sA sB sOut : Shape)
    (vA : Vector Float sA.product) (vB : Vector Float sB.product)
    : Vector Float sOut.product :=
  -- Find contracted labels
  let allLabels := (subsA ++ subsB).eraseDups
  let contractedLabels := allLabels.filter (!subsOut.contains ·)
  let contractedDims := contractedLabels.map fun label =>
    match findIdx subsA label with
    | some idx => List.getD sA idx 1
    | none =>
      match findIdx subsB label with
      | some idx => List.getD sB idx 1
      | none => 1
  let contractedProduct := contractedDims.foldl (· * ·) 1
  Vector.ofFn fun outIdx =>
    let outMulti := linearToMulti sOut outIdx.val
    let labelMap := List.zip subsOut outMulti
    Id.run do
      let mut sum : Float := 0.0
      for cIdx in [:contractedProduct] do
        let cMulti := linearToMulti contractedDims cIdx
        let fullMap := labelMap ++ List.zip contractedLabels cMulti
        let aMulti := subsA.map fun label =>
          match fullMap.find? (·.1 == label) with
          | some (_, v) => v
          | none => 0
        let bMulti := subsB.map fun label =>
          match fullMap.find? (·.1 == label) with
          | some (_, v) => v
          | none => 0
        let aLin := multiToLinear sA aMulti
        let bLin := multiToLinear sB bMulti
        let aVal := if h : aLin < sA.product then vA[aLin] else 0.0
        let bVal := if h : bLin < sB.product then vB[bLin] else 0.0
        sum := sum + aVal * bVal
      return sum

end Lithe

namespace TensorExpr

/-- Evaluate a closed `Float` tensor expression to a flat `Vector`,
recursively interpreting all constructors.

Traverses the `TensorExpr` GADT bottom-up and produces a concrete
`Vector Float s.product`. Panics on unbound variables; use `evalWith`
when the expression contains `.var` nodes. -/
def eval : TensorExpr Float s → Vector Float s.product
  | .literal _ v        => v
  | .fill _ a           => Vector.replicate s.product a
  | .var name _         => panic! s!"Cannot eval unbound variable '{name}'. Use evalWith."
  | .unary op e         => Lithe.evalUnaryVec op e.eval
  | .binary op e₁ e₂   => Lithe.evalBinaryVec op e₁.eval e₂.eval
  | .smul c e           => e.eval.map (c * ·)
  | .select c t f       => Lithe.evalSelectVec c.eval t.eval f.eval
  | .reshape e h        => h ▸ e.eval
  | .transpose e perm   => Lithe.permuteVec _ perm e.eval
  | .broadcast e s₂ _   => Lithe.broadcastVec _ s₂ e.eval
  | .slice e starts sizes _ => Lithe.sliceVec _ starts sizes e.eval
  | .pad e padding fv h => Lithe.padVec _ padding fv e.eval
  | .concat e₁ e₂ ax _  => Lithe.concatVec _ _ ax e₁.eval e₂.eval
  | .gather e ax idxs   => Lithe.gatherVec _ ax idxs e.eval
  | .reduce op ax e     => Lithe.reduceVec op _ ax e.eval
  | .scan op ax e       => Lithe.scanVec op _ ax e.eval
  | .einsum sA sB sO eA eB _ => Lithe.einsumVec sA sB sO _ _ _ eA.eval eB.eval

/-- Evaluate with an environment $\Gamma$ for variable bindings; returns `Except`
for unbound variables.

Like `eval`, but resolves `.var name s` nodes by looking up `name` in the
supplied `Env Float`. Returns `Except.error` with a diagnostic message when
a variable is missing or its shape does not match. -/
def evalWith (env : Env Float) : TensorExpr Float s → Except String (Vector Float s.product)
  | .literal _ v        => .ok v
  | .fill _ a           => .ok (Vector.replicate s.product a)
  | .var name s         =>
    match env.lookup name s with
    | some v => .ok v
    | none => .error s!"Variable '{name}' not found or shape mismatch"
  | .unary op e         => do pure (Lithe.evalUnaryVec op (← e.evalWith env))
  | .binary op e₁ e₂   => do pure (Lithe.evalBinaryVec op (← e₁.evalWith env) (← e₂.evalWith env))
  | .smul c e           => do pure ((← e.evalWith env).map (c * ·))
  | .select c t f       => do
    pure (Lithe.evalSelectVec (← c.evalWith env) (← t.evalWith env) (← f.evalWith env))
  | .reshape e h        => do pure (h ▸ (← e.evalWith env))
  | .transpose e perm   => do pure (Lithe.permuteVec _ perm (← e.evalWith env))
  | .broadcast e s₂ _   => do pure (Lithe.broadcastVec _ s₂ (← e.evalWith env))
  | .slice e starts sizes _ => do pure (Lithe.sliceVec _ starts sizes (← e.evalWith env))
  | .pad e padding fv h => do pure (Lithe.padVec _ padding fv (← e.evalWith env))
  | .concat e₁ e₂ ax _  => do
    pure (Lithe.concatVec _ _ ax (← e₁.evalWith env) (← e₂.evalWith env))
  | .gather e ax idxs   => do pure (Lithe.gatherVec _ ax idxs (← e.evalWith env))
  | .reduce op ax e     => do pure (Lithe.reduceVec op _ ax (← e.evalWith env))
  | .scan op ax e       => do pure (Lithe.scanVec op _ ax (← e.evalWith env))
  | .einsum sA sB sO eA eB _ => do
    pure (Lithe.einsumVec sA sB sO _ _ _ (← eA.evalWith env) (← eB.evalWith env))

end TensorExpr
