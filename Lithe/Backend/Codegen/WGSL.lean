/-
  Lithe/Backend/Codegen/WGSL.lean — WebGPU WGSL code generation + self-contained HTML
-/
import Lithe.Backend.CPU
import Lithe.Backend.Codegen.Common
import Lithe.Env

namespace Lithe.Backend.Codegen.WGSL

open Lithe.Backend Lithe.Backend.Codegen

/-- Generate WGSL shader code for a unary op. -/
private def unaryWGSL (op : UnaryOp) (input : String) : String :=
  match op with
  | .neg     => s!"-{input}"
  | .abs     => s!"abs({input})"
  | .exp     => s!"exp({input})"
  | .log     => s!"log({input})"
  | .sqrt    => s!"sqrt({input})"
  | .sin     => s!"sin({input})"
  | .cos     => s!"cos({input})"
  | .tanh    => s!"tanh({input})"
  | .sigmoid => s!"(1.0 / (1.0 + exp(-{input})))"
  | .sign    => s!"sign({input})"
  | .relu    => s!"max({input}, 0.0)"

/-- Generate WGSL shader code for a binary op. -/
private def binaryWGSL (op : BinaryOp) (left right : String) : String :=
  match op with
  | .add => s!"({left} + {right})"
  | .mul => s!"({left} * {right})"
  | .sub => s!"({left} - {right})"
  | .div => s!"({left} / {right})"
  | .pow => s!"pow({left}, {right})"
  | .max => s!"max({left}, {right})"
  | .min => s!"min({left}, {right})"

/-- Generate a WGSL kernel for a single DAG node. Returns (shaderCode, hadKernel). -/
private def genNodeWGSL (nodeIdx : Nat) (node : DagNode) (kernelIdx : Nat) : (String × Bool) :=
  let sz := node.shape.product
  let outBuf := bufferName nodeIdx
  match node.op with
  | .literal _ | .var _ | .reshape _ => ("", false)
  | .fill val _ =>
    let body := "@compute @workgroup_size(64)\n" ++
      s!"fn {kernelName kernelIdx}(@builtin(global_invocation_id) gid : vec3<u32>) " ++ "{\n" ++
      "    let idx = gid.x;\n" ++
      s!"    if (idx >= {sz}u) " ++ "{ return; }\n" ++
      s!"    {outBuf}[idx] = {repr val};\n" ++
      "}\n\n"
    (body, true)
  | .unary op inId =>
    let expr := unaryWGSL op s!"{bufferName inId}[idx]"
    let body := "@compute @workgroup_size(64)\n" ++
      s!"fn {kernelName kernelIdx}(@builtin(global_invocation_id) gid : vec3<u32>) " ++ "{\n" ++
      "    let idx = gid.x;\n" ++
      s!"    if (idx >= {sz}u) " ++ "{ return; }\n" ++
      s!"    {outBuf}[idx] = {expr};\n" ++
      "}\n\n"
    (body, true)
  | .binary op lId rId =>
    let expr := binaryWGSL op s!"{bufferName lId}[idx]" s!"{bufferName rId}[idx]"
    let body := "@compute @workgroup_size(64)\n" ++
      s!"fn {kernelName kernelIdx}(@builtin(global_invocation_id) gid : vec3<u32>) " ++ "{\n" ++
      "    let idx = gid.x;\n" ++
      s!"    if (idx >= {sz}u) " ++ "{ return; }\n" ++
      s!"    {outBuf}[idx] = {expr};\n" ++
      "}\n\n"
    (body, true)
  | .smul c inId =>
    let body := "@compute @workgroup_size(64)\n" ++
      s!"fn {kernelName kernelIdx}(@builtin(global_invocation_id) gid : vec3<u32>) " ++ "{\n" ++
      "    let idx = gid.x;\n" ++
      s!"    if (idx >= {sz}u) " ++ "{ return; }\n" ++
      s!"    {outBuf}[idx] = {repr c} * {bufferName inId}[idx];\n" ++
      "}\n\n"
    (body, true)
  | _ => (s!"// TODO: WGSL kernel for node {nodeIdx}\n", true)

/-- Generate WGSL compute shader for an execution plan. -/
def generateWGSL (plan : ExecPlan) : String := Id.run do
  -- Declare all buffers
  let mut bufferDecls := ""
  for i in [:plan.nodes.size] do
    let access := if i == plan.output then "read_write" else "read_write"
    bufferDecls := bufferDecls ++
      s!"@group(0) @binding({i}) var<storage, {access}> {bufferName i} : array<f32>;\n"

  -- Generate kernels
  let mut shaderCode := ""
  let mut kIdx := 0
  for i in [:plan.nodes.size] do
    if h : i < plan.nodes.size then
      let node := plan.nodes[i]
      let (code, hadKernel) := genNodeWGSL i node kIdx
      shaderCode := shaderCode ++ code
      if hadKernel then kIdx := kIdx + 1
  return bufferDecls ++ "\n" ++ shaderCode

/-- Generate a self-contained HTML file for WebGPU compute. -/
def toWebGPUHTML (plan : ExecPlan) (env : Env Float) (outputShape : Shape) : String :=
  let wgslCode := generateWGSL plan
  let outSize := outputShape.product
  let shapeStr := "[" ++ ",".intercalate (outputShape.map toString) ++ "]"

  -- Build buffer sizes JSON array
  let bufSizesStr := ",".intercalate (plan.nodes.toList.map fun n => toString (n.shape.product * 4))

  -- Build input uploads
  let inputUploads := Id.run do
    let mut code := ""
    for i in [:plan.nodes.size] do
      if h : i < plan.nodes.size then
        let node := plan.nodes[i]
        match node.op with
        | .literal data =>
          let dataStr := ",".intercalate (data.toList.map Float.toString)
          code := code ++ s!"    device.queue.writeBuffer(buffers[{i}], 0, new Float32Array([{dataStr}]));\n"
        | .var name =>
          match env.find? (·.1 == name) with
          | some (_, td) =>
            let dataStr := ",".intercalate (td.data.toList.map Float.toString)
            code := code ++ s!"    device.queue.writeBuffer(buffers[{i}], 0, new Float32Array([{dataStr}]));\n"
          | none =>
            code := code ++ s!"    // WARNING: variable '{name}' not found in env\n"
        | _ => pure ()
    return code

  -- Build dispatch sequence
  let dispatches := Id.run do
    let mut code := ""
    let mut kIdx := 0
    for i in [:plan.nodes.size] do
      if h : i < plan.nodes.size then
        let node := plan.nodes[i]
        let needsKernel := match node.op with
          | .literal _ | .var _ | .reshape _ => false
          | _ => true
        if needsKernel then
          let sz := node.shape.product
          let wgCount := numWorkgroups sz
          code := code ++ s!"    dispatchKernel(device, commandEncoder, pipelines[{kIdx}], bindGroup, {wgCount});\n"
          kIdx := kIdx + 1
    return code

  -- Build HTML (avoiding problematic string interpolation with braces)
  let html := "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n" ++
    "<title>Lithe - WebGPU Compute</title>\n" ++
    "<style>\n" ++
    "  body { font-family: 'SF Mono', Menlo, monospace; margin: 20px; background: #0d1117; color: #c9d1d9; }\n" ++
    "  h1 { color: #58a6ff; }\n" ++
    "  .result { background: #161b22; padding: 16px; border-radius: 8px; border: 1px solid #30363d; margin: 12px 0; }\n" ++
    "  .error { color: #f85149; }\n" ++
    "  .success { color: #3fb950; }\n" ++
    "  pre { background: #161b22; padding: 12px; border-radius: 6px; overflow-x: auto; }\n" ++
    "</style>\n</head>\n<body>\n" ++
    "<h1>Lithe - WebGPU Compute</h1>\n" ++
    "<div id=\"status\">Initializing WebGPU...</div>\n" ++
    "<div id=\"output\" class=\"result\"></div>\n" ++
    "<h2>WGSL Shader</h2>\n" ++
    "<pre id=\"shader\"></pre>\n" ++
    "<script>\n" ++
    "const wgslCode = `" ++ wgslCode ++ "`;\n" ++
    "document.getElementById('shader').textContent = wgslCode;\n\n" ++
    "function dispatchKernel(device, encoder, pipeline, bindGroup, workgroups) {\n" ++
    "  const pass = encoder.beginComputePass();\n" ++
    "  pass.setPipeline(pipeline);\n" ++
    "  pass.setBindGroup(0, bindGroup);\n" ++
    "  pass.dispatchWorkgroups(workgroups);\n" ++
    "  pass.end();\n" ++
    "}\n\n" ++
    "async function main() {\n" ++
    "  const status = document.getElementById('status');\n" ++
    "  const output = document.getElementById('output');\n" ++
    "  if (!navigator.gpu) { status.innerHTML = '<span class=\"error\">WebGPU not supported.</span>'; return; }\n" ++
    "  const adapter = await navigator.gpu.requestAdapter();\n" ++
    "  if (!adapter) { status.innerHTML = '<span class=\"error\">No GPU adapter.</span>'; return; }\n" ++
    "  const device = await adapter.requestDevice();\n" ++
    "  status.innerHTML = '<span class=\"success\">WebGPU initialized</span>';\n\n" ++
    s!"  const bufferSizes = [{bufSizesStr}];\n" ++
    "  const buffers = bufferSizes.map(size =>\n" ++
    "    device.createBuffer({ size: Math.max(size, 4),\n" ++
    "      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })\n" ++
    "  );\n\n" ++
    inputUploads ++ "\n" ++
    "  const shaderModule = device.createShaderModule({ code: wgslCode });\n" ++
    "  const layoutEntries = buffers.map((_, i) => ({\n" ++
    "    binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }\n" ++
    "  }));\n" ++
    "  const bindGroupLayout = device.createBindGroupLayout({ entries: layoutEntries });\n" ++
    "  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });\n" ++
    "  const bindGroupEntries = buffers.map((buf, i) => ({ binding: i, resource: { buffer: buf } }));\n" ++
    "  const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: bindGroupEntries });\n\n" ++
    "  const kernelNames = wgslCode.match(/fn (kernel_\\\\d+)/g)?.map(s => s.replace('fn ', '')) || [];\n" ++
    "  const pipelines = kernelNames.map(name =>\n" ++
    "    device.createComputePipeline({ layout: pipelineLayout,\n" ++
    "      compute: { module: shaderModule, entryPoint: name } })\n" ++
    "  );\n\n" ++
    "  const commandEncoder = device.createCommandEncoder();\n" ++
    dispatches ++ "\n" ++
    "  const outputBuffer = buffers[" ++ toString plan.output ++ "];\n" ++
    "  const readBuffer = device.createBuffer({ size: " ++ toString (outSize * 4) ++ ",\n" ++
    "    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });\n" ++
    "  commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, " ++ toString (outSize * 4) ++ ");\n" ++
    "  device.queue.submit([commandEncoder.finish()]);\n" ++
    "  await readBuffer.mapAsync(GPUMapMode.READ);\n" ++
    "  const result = new Float32Array(readBuffer.getMappedRange());\n" ++
    "  output.innerHTML = '<strong>Output shape:</strong> " ++ shapeStr ++ "<br>' +\n" ++
    "    '<strong>Result:</strong> [' + Array.from(result).map(function(v){ return v.toFixed(4); }).join(', ') + ']';\n" ++
    "  readBuffer.unmap();\n" ++
    "}\n\n" ++
    "main().catch(err => {\n" ++
    "  document.getElementById('status').innerHTML = '<span class=\"error\">' + err + '</span>';\n" ++
    "});\n" ++
    "</script>\n</body>\n</html>"
  html

end Lithe.Backend.Codegen.WGSL

namespace TensorExpr

/-- Generate a self-contained HTML file that runs this expression on the GPU via WebGPU. -/
def toWebGPUHTML (e : TensorExpr Float s) (env : Env Float) : String :=
  let plan := Lithe.Backend.TensorExpr.toExecPlan e
  Lithe.Backend.Codegen.WGSL.toWebGPUHTML plan env s

end TensorExpr
