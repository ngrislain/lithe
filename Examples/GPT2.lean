/-
  Examples/GPT2.lean — GPT-2 inference and training demo
-/
import Lithe

open TensorExpr Tensor

namespace Examples.GPT2

/-- Run GPT-2 inference on a small token sequence. -/
def inference (envPath : System.FilePath) : IO Unit := do
  IO.println "Loading safetensors weights..."
  let env ← Safetensors.loadSafetensors envPath
  IO.println s!"Loaded {env.length} tensors"

  -- "Hello, this is a" → tokenized
  let tokens : List Nat := [15496, 11, 616, 318, 257]
  IO.println s!"Input tokens: {tokens}"

  -- Build computation graph
  let cfg := NN.defaultConfig
  let logits := NN.gpt2 cfg tokens

  -- Evaluate using CPU backend
  let plan := Lithe.Backend.TensorExpr.toExecPlan logits
  match plan.execute env with
  | .ok result =>
    -- Get logits for last position
    let vocabSize := cfg.vocabSize
    let lastStart := (tokens.length - 1) * vocabSize
    -- Argmax over last position's logits
    let mut maxIdx := 0
    let mut maxVal := result.getD lastStart (-1.0e30)
    for i in [:vocabSize] do
      let v := result.getD (lastStart + i) (-1.0e30)
      if v > maxVal then
        maxVal := v
        maxIdx := i
    IO.println s!"Next token prediction: {maxIdx} (logit: {maxVal})"
  | .error e =>
    IO.println s!"Error during inference: {e}"

/-- Run a few training steps on a tiny sequence. -/
def train : IO Unit := do
  IO.println "=== GPT-2 Training Demo (tiny) ==="

  -- Use a very small config for demo
  let cfg : NN.GPT2Config := {
    vocabSize := 100, maxLen := 32, embedDim := 16,
    nHeads := 2, nLayers := 1, ffDim := 64
  }

  let tokens : List Nat := [1, 2, 3, 4, 5]
  let targets : List Nat := [2, 3, 4, 5, 6]

  -- Build loss graph
  let logits := NN.gpt2 cfg tokens
  let loss := NN.crossEntropyLoss tokens.length cfg.vocabSize logits targets

  -- Initialize random-ish parameters
  let params := NN.gpt2Params cfg
  let mut env : Env Float := params.map fun (name, shape) =>
    let size := Shape.product shape
    -- Initialize with small random values (using a simple hash)
    let data := Id.run do
      let mut arr := Array.replicate size 0.0
      for i in [:size] do
        let hash := (name.hash + i * 2654435761) % 1000000
        arr := arr.set! i ((hash.toFloat / 1000000.0 - 0.5) * 0.02)
      return arr
    (name, ⟨shape, ⟨data, by sorry⟩⟩)

  -- Training loop
  let adamConfig : Optim.AdamConfig := { lr := 0.001 }
  let mut adamState := Optim.AdamState.init params

  for step in [:3] do
    match Optim.trainStep loss env adamConfig adamState with
    | .ok (lossVal, newEnv, newState) =>
      IO.println s!"Step {step}: loss = {lossVal}"
      env := newEnv
      adamState := newState
    | .error e =>
      IO.println s!"Step {step}: error = {e}"
      break

  IO.println "Training demo complete."

end Examples.GPT2

def main (args : List String) : IO Unit := do
  match args with
  | ["--train"] => Examples.GPT2.train
  | [path] => Examples.GPT2.inference path
  | _ =>
    IO.println "Usage:"
    IO.println "  gpt2 model.safetensors   -- Run inference"
    IO.println "  gpt2 --train             -- Run training demo"
