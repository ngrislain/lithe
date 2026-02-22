/-
  Lithe/Dim.lean — Named dimensions and DimShape

  Provides `Dim` (a natural number with an optional name) and `DimShape = List Dim`,
  a named-dimension layer over `Shape = List Nat`. Named dimensions are enforced at
  the type level: `[Dim.named "batch" 32, ⟨784⟩]` and `[⟨32⟩, ⟨784⟩]` are different types.
-/
import Lithe.Shape

/-- A dimension: a natural number with an optional name for type-level enforcement. -/
structure Dim where
  val  : Nat
  name : Option String := none
  deriving Repr, BEq, DecidableEq

namespace Dim

/-- Construct a named dimension. -/
def named (name : String) (n : Nat) : Dim := ⟨n, some name⟩

/-- Anonymous dimension from a natural number literal. -/
instance : OfNat Dim n where ofNat := ⟨n, none⟩

/-- Coerce a `Nat` to an anonymous `Dim`. -/
instance : Coe Nat Dim where coe n := ⟨n, none⟩

/-- Two dims are compatible if their values match and, when both are named, their names match. -/
def compatible (d1 d2 : Dim) : Bool :=
  d1.val == d2.val && match d1.name, d2.name with
  | some n1, some n2 => n1 == n2
  | _, _ => true

end Dim

/-- A shape with optional dimension names. -/
abbrev DimShape := List Dim

namespace DimShape

/-- Strip names, yielding a plain `Shape`. -/
def toShape (ds : DimShape) : Shape := ds.map Dim.val

/-- Wrap a plain `Shape` as anonymous `DimShape`. -/
def fromShape (s : Shape) : DimShape := s.map (⟨·, none⟩)

/-- Total element count (product of all dimension values). -/
def product : DimShape → Nat
  | [] => 1
  | d :: ds => d.val * product ds

/-- Two DimShapes are compatible if they have the same length and pairwise-compatible dims. -/
def compatible (ds1 ds2 : DimShape) : Bool :=
  ds1.length == ds2.length && (ds1.zip ds2).all fun p => p.1.compatible p.2

/-- Find the index of a dimension by name. -/
def findDimIdx (ds : DimShape) (name : String) : Option Nat :=
  go ds 0
where
  go : List Dim → Nat → Option Nat
    | [], _ => none
    | d :: rest, i => if d.name == some name then some i else go rest (i + 1)

/-- Remove the dimension at index `i`. -/
def removeIdx (ds : DimShape) (i : Nat) : DimShape :=
  match ds, i with
  | [], _ => []
  | _ :: rest, 0 => rest
  | d :: rest, n + 1 => d :: removeIdx rest n

/-- Replace the dimension at index `i` with a new dimension (preserving name). -/
def replaceAt (ds : DimShape) (i : Nat) (newVal : Nat) : DimShape :=
  match ds, i with
  | [], _ => []
  | d :: rest, 0 => { d with val := newVal } :: rest
  | d :: rest, n + 1 => d :: replaceAt rest n newVal

/-- Key simp lemma: toShape preserves product. -/
@[simp] theorem toShape_product (ds : DimShape) : (toShape ds).product = product ds := by
  induction ds with
  | nil => simp [toShape, product]
  | cons d rest ih =>
    simp [toShape, product]; exact congrArg (d.val * ·) ih

end DimShape

/-- Convenience: named "batch" dimension. -/
def batch (n : Nat) : Dim := .named "batch" n

/-- Convenience: named "seq" dimension. -/
def seq (n : Nat) : Dim := .named "seq" n

/-- Convenience: named "features" dimension. -/
def features (n : Nat) : Dim := .named "features" n
