/-
  Lithe/Einsum.lean — NumPy-style einsum string syntax

  Parses specs like "ij,jk->ik" into subscript label lists and constructs
  einsum expressions with runtime validity checking.
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Tensor

open Shape

namespace Einsum

/-- Parse a single subscript string (e.g. "ij") into label indices,
    using `labelMap` to assign sequential indices to first-seen characters.
    Returns updated label map and the label list. -/
private def parseSubscript (s : List Char) (labelMap : List (Char × Nat))
    : List (Char × Nat) × List Nat :=
  go s labelMap []
where
  go : List Char → List (Char × Nat) → List Nat → List (Char × Nat) × List Nat
    | [], lm, acc => (lm, acc.reverse)
    | c :: rest, lm, acc =>
      match lm.find? (·.1 == c) with
      | some (_, idx) => go rest lm (idx :: acc)
      | none =>
        let idx := lm.length
        go rest (lm ++ [(c, idx)]) (idx :: acc)

/-- Parse a NumPy-style einsum spec: "ij,jk->ik" → (subsA, subsB, subsOut).
    Each unique character gets a sequential label index (left-to-right first occurrence). -/
def parse (spec : String) : Option (List Nat × List Nat × List Nat) :=
  -- Split on "->"
  let parts := spec.splitOn "->"
  match parts with
  | [inputPart, outputPart] =>
    -- Split inputs on ","
    let inputs := inputPart.splitOn ","
    match inputs with
    | [a, b] =>
      let (lm1, subsA) := parseSubscript a.toList []
      let (lm2, subsB) := parseSubscript b.toList lm1
      let (_, subsOut) := parseSubscript outputPart.toList lm2
      some (subsA, subsB, subsOut)
    | _ => none
  | _ => none

/-- Build a label-to-dimension mapping from subscript labels and shape. -/
private def buildDimMap (subs : List Nat) (shape : Shape) : List (Nat × Nat) :=
  (subs.zip shape).map fun p => (p.1, p.2)

/-- Look up a label's dimension in a dimension map. -/
private def lookupDim (dimMap : List (Nat × Nat)) (label : Nat) : Option Nat :=
  (dimMap.find? (·.1 == label)).map (·.2)

/-- Compute the output shape from parsed labels and input shapes. -/
def outputShape (subsA subsB subsOut : List Nat) (sA sB : Shape) : Shape :=
  let dimMap := buildDimMap subsA sA ++ buildDimMap subsB sB
  subsOut.map fun label =>
    match lookupDim dimMap label with
    | some d => d
    | none => 1

/-- Smart constructor: parse a NumPy-style einsum spec, compute the output shape,
    check validity at runtime, and construct the einsum expression.
    Returns `none` if the spec is invalid or shapes don't match. -/
def ein [Scalar α] (spec : String) (a : TensorExpr α sA) (b : TensorExpr α sB)
    : Option (Σ sOut : Shape, TensorExpr α sOut) :=
  match parse spec with
  | none => none
  | some (subsA, subsB, subsOut) =>
    let sOut := outputShape subsA subsB subsOut sA sB
    if h : decide (IsEinsumValid subsA subsB subsOut sA sB sOut) = true
    then some ⟨sOut, .einsum subsA subsB subsOut a b (of_decide_eq_true h)⟩
    else none

end Einsum
