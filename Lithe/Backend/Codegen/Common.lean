/-
  Lithe/Backend/Codegen/Common.lean — Shared utilities for GPU code generation
-/
import Lithe.Backend.CPU

namespace Lithe.Backend.Codegen

/-- GPU kernel dispatch specification: kernel name, total work items, and workgroup size. -/
structure KernelSpec where
  name       : String
  workSize   : Nat        -- total number of work items
  workGroup  : Nat := 64  -- workgroup / block size
  deriving Repr

/-- GPU buffer declaration: name, size in elements, and read-only flag. -/
structure BufferDecl where
  name     : String
  size     : Nat
  readonly : Bool
  deriving Repr

/-- Classify a `DagOp` as fusible elementwise ($O(n)$ per-element, no cross-element dependency). -/
def isElementwise : DagOp → Bool
  | .unary _ _ => true
  | .binary _ _ _ => true
  | .smul _ _ => true
  | .select _ _ _ => true
  | _ => false

/-- Generate a unique buffer identifier `buf_i` for node $i$. -/
def bufferName (idx : Nat) : String := s!"buf_{idx}"

/-- Generate a unique kernel identifier `kernel_i` for kernel $i$. -/
def kernelName (idx : Nat) : String := s!"kernel_{idx}"

/-- Compute dispatch grid size: $\lceil N / W \rceil$ where $N$ is total work and $W$ is workgroup size. -/
def numWorkgroups (totalWork : Nat) (workGroupSize : Nat := 64) : Nat :=
  (totalWork + workGroupSize - 1) / workGroupSize

end Lithe.Backend.Codegen
