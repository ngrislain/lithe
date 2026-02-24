/-
  Lithe/NN.lean — Neural network layers for GPT-2
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Ops
import Lithe.Tensor
import Lithe.Env
import Lithe.Eval
import Lithe.Smart
import Lithe.Module

open Shape

namespace NN

/-! ### Broadcast helpers -/

/-- Reshape [d] → [1, d] and broadcast to [n, d]. -/
def broadcastLastAxis (n d : Nat) (v : TensorExpr Float [d]) : TensorExpr Float [n, d] :=
  let reshaped := TensorExpr.reshape (s₂ := [1, d]) v (by simp [Shape.product, Nat.one_mul])
  TensorExpr.broadcast reshaped [n, d] ⟨rfl, fun ⟨i, hi⟩ => by
    match i, hi with
    | 0, _ => right; simp [List.getD]
    | 1, _ => left; simp [List.getD]⟩

/-- Reshape [n] → [n, 1] and broadcast to [n, d]. -/
def broadcastFirstAxis (n d : Nat) (v : TensorExpr Float [n]) : TensorExpr Float [n, d] :=
  let reshaped := TensorExpr.reshape (s₂ := [n, 1]) v (by simp [Shape.product, Nat.mul_one])
  TensorExpr.broadcast reshaped [n, d] ⟨rfl, fun ⟨i, hi⟩ => by
    match i, hi with
    | 0, _ => left; simp [List.getD]
    | 1, _ => right; simp [List.getD]⟩

/-! ### Activations -/

/-- GELU activation (GPT-2 variant):
    $\operatorname{gelu}(x) = 0.5 x (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$ -/
def gelu (x : TensorExpr Float s) : TensorExpr Float s :=
  let half := TensorExpr.fill s 0.5
  let one := TensorExpr.fill s 1.0
  let coeff := TensorExpr.fill s 0.7978845608  -- sqrt(2/π)
  let cubic := TensorExpr.fill s 0.044715
  let x3 := TensorExpr.binary .mul x (TensorExpr.binary .mul x x)
  let inner := TensorExpr.binary .mul coeff
    (TensorExpr.binary .add x (TensorExpr.binary .mul cubic x3))
  TensorExpr.binary .mul half
    (TensorExpr.binary .mul x
      (TensorExpr.binary .add one (TensorExpr.unary .tanh inner)))

/-- GELU as a Module (no parameters). -/
def geluLayer (s : Shape) : Module Float s s where
  forward x := gelu x
  params := []

/-! ### Softmax -/

/-- Numerically stable softmax along last axis for [n, d]:
    $\operatorname{softmax}(x)_{ij} = \frac{e^{x_{ij} - \max_j x_{ij}}}{\sum_j e^{x_{ij} - \max_j x_{ij}}}$ -/
def softmax2D (n d : Nat) (x : TensorExpr Float [n, d]) : TensorExpr Float [n, d] :=
  -- max along axis 1: [n, d] → [n]
  let xmax := TensorExpr.reduce .max ⟨1, by show 1 < 2; omega⟩ x
  -- broadcast max back to [n, d]
  let xmaxBcast := broadcastFirstAxis n d xmax
  -- x - max (for numerical stability)
  let shifted := x - xmaxBcast
  -- exp(shifted)
  let expX := TensorExpr.unary .exp shifted
  -- sum along axis 1: [n, d] → [n]
  let sumExp := TensorExpr.reduce .sum ⟨1, by show 1 < 2; omega⟩ expX
  -- broadcast sum back to [n, d]
  let sumBcast := broadcastFirstAxis n d sumExp
  -- exp / sum
  TensorExpr.binary .div expX sumBcast

/-- Softmax for 1D vector [d]:
    Reshapes to [1, d], applies softmax, reshapes back. -/
def softmax1D (d : Nat) (x : TensorExpr Float [d]) : TensorExpr Float [d] :=
  let x2d := TensorExpr.reshape (s₂ := [1, d]) x (by simp [Shape.product, Nat.one_mul])
  let sm := softmax2D 1 d x2d
  TensorExpr.reshape (s₂ := [d]) sm (by simp [Shape.product, Nat.one_mul, Nat.mul_one])

/-! ### Layer Normalization -/

/-- Layer normalization over the last axis [n, d]:
    $\operatorname{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$
    Parameters: `name.weight` [d] (gamma), `name.bias` [d] (beta). -/
def layerNorm (n d : Nat) (name : String) (x : TensorExpr Float [n, d])
    : TensorExpr Float [n, d] :=
  let gamma := TensorExpr.var (name ++ ".weight") [d]
  let beta := TensorExpr.var (name ++ ".bias") [d]
  let eps := TensorExpr.fill [n, d] 1e-5
  -- mean along last axis: [n, d] → [n]
  let dFloat := d.toFloat
  let sum := TensorExpr.reduce .sum ⟨1, by show 1 < 2; omega⟩ x
  let mean := TensorExpr.smul (1.0 / dFloat) sum
  -- broadcast mean back: [n] → [n, d]
  let meanBcast := broadcastFirstAxis n d mean
  -- x - mean
  let centered := x - meanBcast
  -- variance: mean of (x - mean)^2
  let sq := centered * centered
  let varSum := TensorExpr.reduce .sum ⟨1, by show 1 < 2; omega⟩ sq
  let var := TensorExpr.smul (1.0 / dFloat) varSum
  let varBcast := broadcastFirstAxis n d var
  -- normalize: (x - mean) / sqrt(var + eps)
  let normed := TensorExpr.binary .div centered (TensorExpr.unary .sqrt (varBcast + eps))
  -- scale and shift: gamma * normed + beta
  let gammaBcast := broadcastLastAxis n d gamma
  let betaBcast := broadcastLastAxis n d beta
  gammaBcast * normed + betaBcast

/-- LayerNorm as a Module. -/
def layerNormModule (n d : Nat) (name : String) : Module Float [n, d] [n, d] where
  forward x := layerNorm n d name x
  params := [(name ++ ".weight", [d]), (name ++ ".bias", [d])]

/-! ### Batched Linear -/

/-- Batched linear layer: `[n, inDim] → [n, outDim]` via einsum + bias.
    $y_{ni} = \sum_j x_{nj} W_{ji} + b_i$
    Parameters: `name.weight` [inDim, outDim], `name.bias` [outDim]. -/
def linearBatched (n inDim outDim : Nat) (name : String)
    (x : TensorExpr Float [n, inDim]) : TensorExpr Float [n, outDim] :=
  let w := TensorExpr.var (name ++ ".weight") [inDim, outDim]
  let b := TensorExpr.var (name ++ ".bias") [outDim]
  -- einsum "ni,io->no": [n, inDim] @ [inDim, outDim] → [n, outDim]
  let y := TensorExpr.einsum [0, 1] [1, 2] [0, 2] x w (Tensor.matmul_einsum_valid n inDim outDim)
  -- add bias broadcast to [n, outDim]
  let bBcast := broadcastLastAxis n outDim b
  y + bBcast

/-- Batched linear as a Module. -/
def linearBatchedModule (n inDim outDim : Nat) (name : String)
    : Module Float [n, inDim] [n, outDim] where
  forward x := linearBatched n inDim outDim name x
  params := [(name ++ ".weight", [inDim, outDim]), (name ++ ".bias", [outDim])]

/-! ### Embedding -/

/-- Embedding lookup: gather rows from weight matrix.
    `indices` is a compile-time list of token IDs.
    Parameters: `name.weight` [vocabSize, dim]. -/
def embedding (vocabSize dim : Nat) (name : String) (indices : List Nat)
    : TensorExpr Float [indices.length, dim] :=
  let w := TensorExpr.var (name ++ ".weight") [vocabSize, dim]
  -- Gather along axis 0 with the given indices
  let idxVec : Vector Nat indices.length := ⟨indices.toArray, by simp [List.size_toArray]⟩
  TensorExpr.gather w ⟨0, by show 0 < 2; omega⟩ idxVec

/-- Positional embedding: gather rows from pos embedding matrix. -/
def posEmbedding (maxLen dim : Nat) (name : String) (seqLen : Nat)
    : TensorExpr Float [seqLen, dim] :=
  let w := TensorExpr.var (name ++ ".weight") [maxLen, dim]
  let indices := List.range seqLen
  let idxVec : Vector Nat seqLen := ⟨indices.toArray, by simp [indices]⟩
  TensorExpr.gather w ⟨0, by show 0 < 2; omega⟩ idxVec

/-! ### Safe shape-checked constructors for NN -/

private def safeSliceNN {s : Shape} (e : TensorExpr Float s) (starts sizes : List Nat) :
    TensorExpr Float sizes :=
  if h : Shape.ValidSlice s starts sizes then TensorExpr.slice e starts sizes h
  else TensorExpr.fill sizes 0.0

private def safeReshapeNN {s₁ : Shape} (e : TensorExpr Float s₁) (s₂ : Shape) :
    TensorExpr Float s₂ :=
  if h : Shape.product s₁ = Shape.product s₂ then TensorExpr.reshape e h
  else TensorExpr.fill s₂ 0.0

private def safeConcatNN {s₁ s₂ : Shape} (e₁ : TensorExpr Float s₁) (e₂ : TensorExpr Float s₂)
    (axis : Fin s₁.length) : TensorExpr Float (Shape.concatShape s₁ s₂ axis) :=
  if h : Shape.ConcatCompatible s₁ s₂ axis.val then TensorExpr.concat e₁ e₂ axis h
  else TensorExpr.fill (Shape.concatShape s₁ s₂ axis) 0.0

/-! ### Multi-Head Attention -/

/-- Single attention head operating on 2D slices.
    Q, K, V each have shape [seqLen, headDim].
    Returns [seqLen, headDim]. -/
def attentionHead (seqLen headDim : Nat) (q k v : TensorExpr Float [seqLen, headDim])
    (mask : TensorExpr Float [seqLen, seqLen])
    : TensorExpr Float [seqLen, headDim] :=
  let scale := TensorExpr.fill [seqLen, seqLen] (1.0 / Float.sqrt headDim.toFloat)
  -- QK^T: [seqLen, headDim] @ [headDim, seqLen] → [seqLen, seqLen]
  let kT := Tensor.transpose2D k
  let scores := Tensor.matmul q kT
  -- Scale and mask
  let scaled := scores * scale
  let masked := scaled + mask
  -- Softmax along last axis
  let attnWeights := softmax2D seqLen seqLen masked
  -- Weighted sum: [seqLen, seqLen] @ [seqLen, headDim] → [seqLen, headDim]
  Tensor.matmul attnWeights v

/-- Build a causal (lower-triangular) attention mask [seqLen, seqLen].
    Upper triangle filled with -1e10, lower triangle + diagonal with 0. -/
def causalMask (seqLen : Nat) : TensorExpr Float [seqLen, seqLen] :=
  let size := seqLen * seqLen
  let data := Array.ofFn fun (idx : Fin size) =>
    let i := idx.val / seqLen
    let j := idx.val % seqLen
    if j > i then (-1.0e10 : Float) else 0.0
  TensorExpr.literal [seqLen, seqLen] ⟨data, by simp [Shape.product, Nat.mul_one]; exact Array.size_ofFn⟩

/-- Multi-head attention (loop over heads with 2D ops).
    Input: [seqLen, embedDim], Output: [seqLen, embedDim].
    Uses combined QKV projection, splits into heads via slicing.
    Parameters: `name.c_attn.weight`, `name.c_attn.bias`,
                `name.c_proj.weight`, `name.c_proj.bias`. -/
def multiHeadAttention (seqLen embedDim nHeads : Nat) (name : String)
    (x : TensorExpr Float [seqLen, embedDim])
    : TensorExpr Float [seqLen, embedDim] :=
  let headDim := embedDim / nHeads
  -- Combined QKV projection: [seqLen, embedDim] → [seqLen, 3*embedDim]
  let qkvDim := 3 * embedDim
  let qkv := linearBatched seqLen embedDim qkvDim (name ++ ".c_attn") x
  -- Build causal mask
  let mask := causalMask seqLen
  -- Process each head via List.range (avoids for-loop bound limitation)
  let headOutputs : List (TensorExpr Float [seqLen, headDim]) :=
    (List.range nHeads).map fun h =>
      let qStart := h * headDim
      let kStart := embedDim + h * headDim
      let vStart := 2 * embedDim + h * headDim
      let qh := safeSliceNN qkv [0, qStart] [seqLen, headDim]
      let kh := safeSliceNN qkv [0, kStart] [seqLen, headDim]
      let vh := safeSliceNN qkv [0, vStart] [seqLen, headDim]
      attentionHead seqLen headDim qh kh vh mask
  -- Concatenate head outputs using existential accumulator (width grows each step)
  let combined := Id.run do
    let mut acc : Σ w : Nat, TensorExpr Float [seqLen, w] :=
      match headOutputs.head? with
      | some h => ⟨headDim, h⟩
      | none => ⟨0, TensorExpr.fill [seqLen, 0] 0.0⟩
    for entry in headOutputs.drop 1 do
      let ⟨w, expr⟩ := acc
      let cat := safeConcatNN expr entry ⟨1, by show 1 < 2; omega⟩
      acc := ⟨w + headDim, safeReshapeNN cat [seqLen, w + headDim]⟩
    return acc
  let combined' := safeReshapeNN combined.2 [seqLen, embedDim]
  -- Output projection: [seqLen, embedDim] → [seqLen, embedDim]
  linearBatched seqLen embedDim embedDim (name ++ ".c_proj") combined'

/-! ### Transformer Block -/

/-- GPT-2 transformer block:
    LN1 → MHA → residual → LN2 → MLP(linear→GELU→linear) → residual -/
def transformerBlock (seqLen embedDim nHeads ffDim : Nat) (name : String)
    (x : TensorExpr Float [seqLen, embedDim])
    : TensorExpr Float [seqLen, embedDim] :=
  -- LN1 → MHA → residual
  let ln1Out := layerNorm seqLen embedDim (name ++ ".ln_1") x
  let attnOut := multiHeadAttention seqLen embedDim nHeads (name ++ ".attn") ln1Out
  let residual1 := x + attnOut
  -- LN2 → MLP → residual
  let ln2Out := layerNorm seqLen embedDim (name ++ ".ln_2") residual1
  let mlpUp := linearBatched seqLen embedDim ffDim (name ++ ".mlp.c_fc") ln2Out
  let mlpAct := gelu mlpUp
  let mlpDown := linearBatched seqLen ffDim embedDim (name ++ ".mlp.c_proj") mlpAct
  residual1 + mlpDown

/-! ### GPT-2 Model -/

/-- GPT-2 configuration. -/
structure GPT2Config where
  vocabSize : Nat := 50257
  maxLen    : Nat := 1024
  embedDim  : Nat := 768
  nHeads    : Nat := 12
  nLayers   : Nat := 12
  ffDim     : Nat := 3072  -- 4 * embedDim

/-- Default GPT-2 small configuration. -/
def defaultConfig : GPT2Config := {}

/-- Build the GPT-2 computation graph.
    Input: list of token IDs. Output: logits [seqLen, vocabSize]. -/
def gpt2 (cfg : GPT2Config) (tokens : List Nat)
    : TensorExpr Float [tokens.length, cfg.vocabSize] :=
  let seqLen := tokens.length
  -- Token + positional embeddings
  let tokEmb := embedding cfg.vocabSize cfg.embedDim "wte" tokens
  let posEmb := posEmbedding cfg.maxLen cfg.embedDim "wpe" seqLen
  let x := tokEmb + posEmb
  -- Transformer blocks
  let x' := applyBlocks seqLen cfg.embedDim cfg.nHeads cfg.ffDim cfg.nLayers 0 x
  -- Final layer norm
  let xNorm := layerNorm seqLen cfg.embedDim "ln_f" x'
  -- LM head: project to vocab (tied weights with wte)
  let wte := TensorExpr.var "wte.weight" [cfg.vocabSize, cfg.embedDim]
  let wteT := Tensor.transpose2D wte  -- [embedDim, vocabSize]
  -- [seqLen, embedDim] @ [embedDim, vocabSize] → [seqLen, vocabSize]
  TensorExpr.einsum [0, 1] [1, 2] [0, 2] xNorm wteT (Tensor.matmul_einsum_valid seqLen cfg.embedDim cfg.vocabSize)
where
  /-- Apply N transformer blocks sequentially. -/
  applyBlocks (seqLen embedDim nHeads ffDim nLayers idx : Nat)
      (x : TensorExpr Float [seqLen, embedDim])
      : TensorExpr Float [seqLen, embedDim] :=
    if idx >= nLayers then x
    else
      let blockName := "h." ++ toString idx
      let x' := transformerBlock seqLen embedDim nHeads ffDim blockName x
      applyBlocks seqLen embedDim nHeads ffDim nLayers (idx + 1) x'

/-- List all GPT-2 parameter names and shapes. -/
def gpt2Params (cfg : GPT2Config) : List (String × Shape) :=
  let embedParams := [
    ("wte.weight", [cfg.vocabSize, cfg.embedDim]),
    ("wpe.weight", [cfg.maxLen, cfg.embedDim])
  ]
  let blockParams := ((List.range cfg.nLayers).map fun i =>
    let name := "h." ++ toString i
    [ (name ++ ".ln_1.weight", [cfg.embedDim]),
      (name ++ ".ln_1.bias", [cfg.embedDim]),
      (name ++ ".attn.c_attn.weight", [cfg.embedDim, 3 * cfg.embedDim]),
      (name ++ ".attn.c_attn.bias", [3 * cfg.embedDim]),
      (name ++ ".attn.c_proj.weight", [cfg.embedDim, cfg.embedDim]),
      (name ++ ".attn.c_proj.bias", [cfg.embedDim]),
      (name ++ ".ln_2.weight", [cfg.embedDim]),
      (name ++ ".ln_2.bias", [cfg.embedDim]),
      (name ++ ".mlp.c_fc.weight", [cfg.embedDim, cfg.ffDim]),
      (name ++ ".mlp.c_fc.bias", [cfg.ffDim]),
      (name ++ ".mlp.c_proj.weight", [cfg.ffDim, cfg.embedDim]),
      (name ++ ".mlp.c_proj.bias", [cfg.embedDim])
    ]).flatten
  let finalParams := [
    ("ln_f.weight", [cfg.embedDim]),
    ("ln_f.bias", [cfg.embedDim])
  ]
  embedParams ++ blockParams ++ finalParams

/-! ### Cross-Entropy Loss -/

/-- One-hot encode target indices for [seqLen] tokens into [seqLen, vocabSize]. -/
def oneHot (seqLen vocabSize : Nat) (targets : List Nat) : TensorExpr Float [seqLen, vocabSize] :=
  let size := seqLen * vocabSize
  let data := Array.ofFn fun (idx : Fin size) =>
    let i := idx.val / vocabSize
    let j := idx.val % vocabSize
    let t := targets.getD i 0
    if j == t then (1.0 : Float) else 0.0
  TensorExpr.literal [seqLen, vocabSize] ⟨data, by simp [Shape.product, Nat.mul_one]; exact Array.size_ofFn⟩

/-- Cross-entropy loss: $-\frac{1}{n}\sum_i \sum_c y_{ic} \log(\text{softmax}(x)_{ic})$ -/
def crossEntropyLoss (seqLen vocabSize : Nat)
    (logits : TensorExpr Float [seqLen, vocabSize])
    (targets : List Nat) : TensorExpr Float [] :=
  let sm := softmax2D seqLen vocabSize logits
  -- log(softmax) — clamp to avoid log(0)
  let eps := TensorExpr.fill [seqLen, vocabSize] 1e-10
  let logSm := TensorExpr.unary .log (sm + eps)
  -- one-hot targets
  let oh := oneHot seqLen vocabSize targets
  -- element-wise multiply and sum
  let prod := oh * logSm
  -- sum over vocab: [seqLen, vocabSize] → [seqLen]
  let sumVocab := TensorExpr.reduce .sum ⟨1, by show 1 < 2; omega⟩ prod
  -- sum over seq: [seqLen] → []
  let sumSeq := TensorExpr.reduce .sum ⟨0, by show 0 < 1; omega⟩ sumVocab
  -- negate and mean
  TensorExpr.smul (-1.0 / seqLen.toFloat) sumSeq

end NN
