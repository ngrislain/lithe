/-
  Lithe/Backend/Codegen/CUDA.lean — CUDA C source code generation
-/
import Lithe.Backend.CPU
import Lithe.Backend.Codegen.Common

namespace Lithe.Backend.Codegen.CUDA

open Lithe.Backend Lithe.Backend.Codegen

/-- Generate CUDA kernel code for a single DagOp. -/
private def genKernelBody (nodeIdx : Nat) (node : DagNode) : String :=
  let size := node.shape.product
  let outBuf := bufferName nodeIdx
  match node.op with
  | .literal _ => ""
  | .fill val sz =>
    s!"__global__ void {kernelName nodeIdx}(float* {outBuf}) \{\n" ++
    s!"  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" ++
    s!"  if (idx >= {sz}) return;\n" ++
    s!"  {outBuf}[idx] = {repr val}f;\n" ++
    "}\n\n"
  | .var _ => ""
  | .unary op inId =>
    let inBuf := bufferName inId
    let opExpr := match op with
      | .neg => s!"-{inBuf}[idx]"
      | .abs => s!"fabsf({inBuf}[idx])"
      | .exp => s!"expf({inBuf}[idx])"
      | .log => s!"logf({inBuf}[idx])"
      | .sqrt => s!"sqrtf({inBuf}[idx])"
      | .sin => s!"sinf({inBuf}[idx])"
      | .cos => s!"cosf({inBuf}[idx])"
      | .tanh => s!"tanhf({inBuf}[idx])"
      | .sigmoid => s!"1.0f / (1.0f + expf(-{inBuf}[idx]))"
      | .sign => s!"({inBuf}[idx] > 0 ? 1.0f : ({inBuf}[idx] < 0 ? -1.0f : 0.0f))"
      | .relu => s!"fmaxf({inBuf}[idx], 0.0f)"
    s!"__global__ void {kernelName nodeIdx}(const float* {inBuf}, float* {outBuf}) \{\n" ++
    s!"  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" ++
    s!"  if (idx >= {size}) return;\n" ++
    s!"  {outBuf}[idx] = {opExpr};\n" ++
    "}\n\n"
  | .binary op lId rId =>
    let lBuf := bufferName lId
    let rBuf := bufferName rId
    let opExpr := match op with
      | .add => s!"{lBuf}[idx] + {rBuf}[idx]"
      | .mul => s!"{lBuf}[idx] * {rBuf}[idx]"
      | .sub => s!"{lBuf}[idx] - {rBuf}[idx]"
      | .div => s!"{lBuf}[idx] / {rBuf}[idx]"
      | .pow => s!"powf({lBuf}[idx], {rBuf}[idx])"
      | .max => s!"fmaxf({lBuf}[idx], {rBuf}[idx])"
      | .min => s!"fminf({lBuf}[idx], {rBuf}[idx])"
    s!"__global__ void {kernelName nodeIdx}(const float* {lBuf}, const float* {rBuf}, float* {outBuf}) \{\n" ++
    s!"  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" ++
    s!"  if (idx >= {size}) return;\n" ++
    s!"  {outBuf}[idx] = {opExpr};\n" ++
    "}\n\n"
  | .smul c inId =>
    let inBuf := bufferName inId
    s!"__global__ void {kernelName nodeIdx}(const float* {inBuf}, float* {outBuf}) \{\n" ++
    s!"  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" ++
    s!"  if (idx >= {size}) return;\n" ++
    s!"  {outBuf}[idx] = {repr c}f * {inBuf}[idx];\n" ++
    "}\n\n"
  | .reshape _ => ""
  | _ => s!"// TODO: kernel for node {nodeIdx}\n"

/-- Generate the host execute function. -/
private def genHostFunction (plan : ExecPlan) : String :=
  let bufDecls := plan.nodes.foldl (fun (acc : String × Nat) node =>
    let (code, idx) := acc
    let sz := node.shape.product
    (code ++ s!"  float* d_{bufferName idx};\n  cudaMalloc(&d_{bufferName idx}, {sz} * sizeof(float));\n", idx + 1)
  ) ("", 0) |>.1
  let freeAll := plan.nodes.foldl (fun (acc : String × Nat) _ =>
    let (code, idx) := acc
    (code ++ s!"  cudaFree(d_{bufferName idx});\n", idx + 1)
  ) ("", 0) |>.1
  "void execute() {\n" ++
  "  // Allocate device buffers\n" ++
  bufDecls ++
  "  // TODO: Upload inputs, launch kernels in order, read back output\n\n" ++
  "  // Free\n" ++
  freeAll ++
  "}\n"

/-- Generate complete CUDA C source code for an execution plan, including
    `__global__` kernels and a host `execute()` function that allocates
    device buffers, dispatches kernels, and frees resources. -/
def generateCUDA (plan : ExecPlan) : String :=
  let header := "#include <cuda_runtime.h>\n#include <stdio.h>\n#include <math.h>\n\n"
  let kernels := plan.nodes.foldl (fun (acc : String × Nat) node =>
    let (code, idx) := acc
    (code ++ genKernelBody idx node, idx + 1)
  ) ("", 0) |>.1
  let hostFn := genHostFunction plan
  header ++ kernels ++ "\n" ++ hostFn

end Lithe.Backend.Codegen.CUDA

namespace TensorExpr

/-- Generate CUDA C source code for this tensor expression. Flattens the expression
    into a DAG execution plan and emits corresponding CUDA kernels. -/
def toCUDA (e : TensorExpr Float s) : String :=
  let plan := Lithe.Backend.TensorExpr.toExecPlan e
  Lithe.Backend.Codegen.CUDA.generateCUDA plan

end TensorExpr
