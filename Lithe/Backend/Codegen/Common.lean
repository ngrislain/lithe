/-
  Lithe/Backend/Codegen/Common.lean — Shared utilities for GPU code generation
-/
import Lithe.Backend.CPU

namespace Lithe.Backend.Codegen

/-- A kernel specification for GPU dispatch. -/
structure KernelSpec where
  name       : String
  workSize   : Nat        -- total number of work items
  workGroup  : Nat := 64  -- workgroup / block size
  deriving Repr

/-- Buffer declaration for GPU codegen. -/
structure BufferDecl where
  name     : String
  size     : Nat
  readonly : Bool
  deriving Repr

/-- Classify a DagOp as fusible elementwise or not. -/
def isElementwise : DagOp → Bool
  | .unary _ _ => true
  | .binary _ _ _ => true
  | .smul _ _ => true
  | .select _ _ _ => true
  | _ => false

/-- Generate a unique buffer name. -/
def bufferName (idx : Nat) : String := s!"buf_{idx}"

/-- Generate a unique kernel name. -/
def kernelName (idx : Nat) : String := s!"kernel_{idx}"

/-- Compute dispatch dimensions (number of workgroups). -/
def numWorkgroups (totalWork : Nat) (workGroupSize : Nat := 64) : Nat :=
  (totalWork + workGroupSize - 1) / workGroupSize

end Lithe.Backend.Codegen
