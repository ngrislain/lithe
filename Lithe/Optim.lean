/-
  Lithe/Optim.lean — Adam optimizer and training utilities
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Tensor
import Lithe.Env
import Lithe.Eval
import Lithe.Autodiff

namespace Optim

/-! ### Adam Optimizer -/

/-- Per-parameter optimizer state: first and second moment estimates. -/
structure ParamState where
  m : Array Float  -- first moment
  v : Array Float  -- second moment

/-- Adam optimizer state for all parameters. -/
structure AdamState where
  paramStates : List (String × ParamState)
  step : Nat := 0

/-- Adam optimizer hyperparameters. -/
structure AdamConfig where
  lr    : Float := 0.001
  beta1 : Float := 0.9
  beta2 : Float := 0.999
  eps   : Float := 1e-8

/-- Initialize Adam state for a set of parameters (all zeros). -/
def AdamState.init (params : List (String × Shape)) : AdamState :=
  let states := params.map fun (name, shape) =>
    let size := Shape.product shape
    (name, { m := Array.replicate size 0.0, v := Array.replicate size 0.0 : ParamState })
  { paramStates := states, step := 0 }

/-- Look up a parameter's state by name. -/
private def lookupParamState (states : List (String × ParamState)) (name : String)
    : Option ParamState :=
  match states.find? (fun p => p.1 == name) with
  | some (_, ps) => some ps
  | none => none

/-- Update a parameter's state by name. -/
private def updateParamState (states : List (String × ParamState))
    (name : String) (newState : ParamState) : List (String × ParamState) :=
  states.map fun (n, ps) => if n == name then (n, newState) else (n, ps)

/-- Perform one Adam step, updating parameters and optimizer state. -/
def adamStep (config : AdamConfig) (env : Env Float)
    (grads : Env Float) (state : AdamState) : (Env Float × AdamState) :=
  let t := state.step + 1
  let tFloat := t.toFloat
  let beta1PowT := config.beta1 ^ tFloat
  let beta2PowT := config.beta2 ^ tFloat
  Id.run do
    let mut newEnv := env
    let mut newStates := state.paramStates
    for (name, td) in env do
      -- Look up gradient for this parameter
      match grads.find? (fun p => p.1 == name) with
      | none => pure ()  -- no gradient, skip
      | some (_, gradTd) =>
        -- Look up optimizer state
        let ps := match lookupParamState newStates name with
          | some ps => ps
          | none => { m := Array.replicate td.shape.product 0.0,
                      v := Array.replicate td.shape.product 0.0 }
        -- Update moments and parameters (element-wise, each index is independent)
        let paramArr := td.data.toArray
        let gradArr := gradTd.data.toArray
        let n' := td.shape.product
        let newM := Array.ofFn fun (idx : Fin n') =>
          let i := idx.val
          let g := gradArr.getD i 0.0
          config.beta1 * ps.m.getD i 0.0 + (1.0 - config.beta1) * g
        let newV := Array.ofFn fun (idx : Fin n') =>
          let i := idx.val
          let g := gradArr.getD i 0.0
          config.beta2 * ps.v.getD i 0.0 + (1.0 - config.beta2) * g * g
        let newParam := Array.ofFn fun (idx : Fin n') =>
          let i := idx.val
          let g := gradArr.getD i 0.0
          let mi := config.beta1 * ps.m.getD i 0.0 + (1.0 - config.beta1) * g
          let vi := config.beta2 * ps.v.getD i 0.0 + (1.0 - config.beta2) * g * g
          -- Bias correction
          let mHat := mi / (1.0 - beta1PowT)
          let vHat := vi / (1.0 - beta2PowT)
          -- Parameter update
          paramArr.getD i 0.0 - config.lr * mHat / (vHat.sqrt + config.eps)
        -- Store updated state
        newStates := updateParamState newStates name { m := newM, v := newV }
        -- Store updated parameter
        newEnv := newEnv.map fun (n, td') =>
          if n == name then
            (n, ⟨td.shape, ⟨newParam, Array.size_ofFn⟩⟩)
          else (n, td')
    return (newEnv, { paramStates := newStates, step := t })

/-! ### Gradient Evaluation -/

/-- Evaluate symbolic gradients to concrete values using an environment.
    Sums multiple gradient contributions for the same variable. -/
def evalGrads (grads : Grads) (env : Env Float) : Except String (Env Float) := do
  -- Group by variable name and sum contributions
  let mut result : Env Float := []
  for (name, ⟨shape, expr⟩) in grads do
    let val ← expr.evalWith env
    match result.find? (fun p => p.1 == name) with
    | none =>
      result := result ++ [(name, ⟨shape, val⟩)]
    | some (_, existing) =>
      -- Sum with existing gradient
      if h : existing.shape.product = shape.product then
        let existingData := h ▸ existing.data
        let summed := existingData.zipWith (· + ·) val
        result := result.map fun (n, td) =>
          if n == name then (n, ⟨shape, summed⟩) else (n, td)
      else
        result := result ++ [(name, ⟨shape, val⟩)]
  return result

/-! ### Training Loop Helper -/

/-- Single training step: forward → backward → evaluate gradients → Adam update.
    Returns (loss value, updated env, updated optimizer state). -/
def trainStep (loss : TensorExpr Float []) (env : Env Float)
    (config : AdamConfig) (state : AdamState)
    : Except String (Float × Env Float × AdamState) := do
  -- Forward: evaluate loss
  let lossVal ← loss.evalWith env
  let lossFloat := lossVal.get ⟨0, by simp [Shape.product]⟩
  -- Backward: compute symbolic gradients
  let symbolicGrads := loss.grad
  -- Evaluate gradients with current env
  let gradEnv ← evalGrads symbolicGrads env
  -- Adam step
  let (newEnv, newState) := adamStep config env gradEnv state
  return (lossFloat, newEnv, newState)

end Optim
