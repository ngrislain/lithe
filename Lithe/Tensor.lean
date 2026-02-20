/-
  Lithe/Tensor.lean — The core dependently-typed GADT for symbolic tensor expressions
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Ops

open Shape in
/-- A symbolic tensor expression. Each constructor records an operation;
    shapes are tracked at the type level. -/
inductive TensorExpr (α : Type) [Scalar α] : Shape → Type where
  -- Data Sources
  | literal   : (s : Shape) → Vector α s.product → TensorExpr α s
  | fill      : (s : Shape) → α → TensorExpr α s
  | var       : (name : String) → (s : Shape) → TensorExpr α s

  -- Elementwise (shape-preserving)
  | unary     : UnaryOp → TensorExpr α s → TensorExpr α s
  | binary    : BinaryOp → TensorExpr α s → TensorExpr α s → TensorExpr α s
  | smul      : α → TensorExpr α s → TensorExpr α s
  | select    : TensorExpr α s → TensorExpr α s → TensorExpr α s → TensorExpr α s

  -- Shape Manipulation
  | reshape   : TensorExpr α s₁ → s₁.product = s₂.product → TensorExpr α s₂
  | transpose : TensorExpr α s → (perm : Vector (Fin s.length) s.length) →
                TensorExpr α (permuteShape s perm)
  | broadcast : TensorExpr α s₁ → (s₂ : Shape) →
                IsBroadcastable s₁ s₂ → TensorExpr α s₂
  | slice     : TensorExpr α s → (starts sizes : List Nat) →
                ValidSlice s starts sizes → TensorExpr α sizes
  | pad       : TensorExpr α s → (padding : List (Nat × Nat)) → α →
                padding.length = s.length → TensorExpr α (padShape s padding)

  -- Structural
  | concat    : TensorExpr α s₁ → TensorExpr α s₂ → (axis : Fin s₁.length) →
                ConcatCompatible s₁ s₂ axis.val → TensorExpr α (concatShape s₁ s₂ axis)
  | gather    : TensorExpr α s → (axis : Fin s.length) → Vector Nat numIdx →
                TensorExpr α (gatherShape s axis numIdx)

  -- Reductions & Scans
  | reduce    : ReduceOp → (axis : Fin s.length) → TensorExpr α s →
                TensorExpr α (s.removeAt axis)
  | scan      : ReduceOp → (axis : Fin s.length) → TensorExpr α s → TensorExpr α s

  -- Contraction (einsum)
  | einsum    : (subsA subsB subsOut : List Nat) →
                TensorExpr α sA → TensorExpr α sB →
                IsEinsumValid subsA subsB subsOut sA sB sOut →
                TensorExpr α sOut

/-- Convenience alias. -/
abbrev Tensor (α : Type) [Scalar α] := TensorExpr α
