/-
  Lithe/Backend/CPU.lean — DAG-based evaluation with subexpression sharing
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Ops
import Lithe.Index
import Lithe.Tensor
import Lithe.Env
import Lithe.Eval

namespace Lithe.Backend

/-- A node operation in the flattened DAG, representing one computational step
    with references to input node indices. Each variant carries the operation
    type and the integer indices of its operand nodes. -/
inductive DagOp where
  | literal (data : Array Float)
  | fill (val : Float) (size : Nat)
  | var (name : String)
  | unary (op : UnaryOp) (input : Nat)
  | binary (op : BinaryOp) (left right : Nat)
  | smul (scalar : Float) (input : Nat)
  | select (cond ifTrue ifFalse : Nat)
  | reshape (input : Nat)
  | transpose (input : Nat) (perm : List Nat)
  | broadcast (input : Nat) (srcShape tgtShape : Shape)
  | slice (input : Nat) (starts sizes : List Nat)
  | pad (input : Nat) (padding : List (Nat × Nat)) (fillVal : Float)
  | concat (left right : Nat) (axis : Nat)
  | gather (input : Nat) (axis : Nat) (indices : Array Nat)
  | reduce (op : ReduceOp) (axis : Nat) (input : Nat)
  | scan (op : ReduceOp) (axis : Nat) (input : Nat)
  | einsum (subsA subsB subsOut : List Nat) (left right : Nat)
  deriving Repr

/-- A DAG node: an operation paired with its output shape. -/
structure DagNode where
  op    : DagOp
  shape : Shape
  deriving Repr

/-- An execution plan: a topologically sorted array of `DagNode`s with a
    designated output node index. Nodes are ordered so that every operand
    appears before the node that references it. -/
structure ExecPlan where
  nodes  : Array DagNode
  output : Nat  -- index of the output node
  deriving Repr

/-- Mutable state for DAG construction, accumulating nodes during flattening. -/
structure FlattenState where
  nodes : Array DagNode := #[]

namespace FlattenState

/-- Append a new node to the DAG and return its index. -/
def addNode (st : FlattenState) (node : DagNode) : (Nat × FlattenState) :=
  (st.nodes.size, { nodes := st.nodes.push node })

end FlattenState

private def getBuffer (buffers : Array (Array Float)) (i : Nat) : Array Float :=
  if h : i < buffers.size then buffers[i] else #[]

private def getNodeShape (nodes : Array DagNode) (i : Nat) : Shape :=
  if h : i < nodes.size then nodes[i].shape else []

private def setAt (l : List Nat) (pos val : Nat) : List Nat :=
  match l, pos with
  | [], _ => []
  | _ :: xs, 0 => val :: xs
  | x :: xs, n + 1 => x :: setAt xs n val

private def insertAt (l : List Nat) (pos val : Nat) : List Nat :=
  match l, pos with
  | xs, 0 => val :: xs
  | [], _ => [val]
  | x :: xs, n + 1 => x :: insertAt xs n val

private def findIdx (l : List Nat) (x : Nat) : Option Nat :=
  go l x 0
where
  go : List Nat → Nat → Nat → Option Nat
    | [], _, _ => none
    | y :: ys, x, i => if x == y then some i else go ys x (i + 1)

/-- Flatten a `TensorExpr` tree into a linear DAG, converting tree references
    to integer node indices. Returns the root node index and the accumulated state. -/
partial def flatten (st : FlattenState) : TensorExpr Float s → (Nat × FlattenState)
  | .literal s v =>
    st.addNode { op := .literal v.toArray, shape := s }
  | .fill s val =>
    st.addNode { op := .fill val s.product, shape := s }
  | .var name s =>
    st.addNode { op := .var name, shape := s }
  | .unary op e =>
    let (inId, st') := flatten st e
    st'.addNode { op := .unary op inId, shape := s }
  | .binary op e₁ e₂ =>
    let (id1, st1) := flatten st e₁
    let (id2, st2) := flatten st1 e₂
    st2.addNode { op := .binary op id1 id2, shape := s }
  | .smul c e =>
    let (inId, st') := flatten st e
    st'.addNode { op := .smul c inId, shape := s }
  | .select c t f =>
    let (cId, st1) := flatten st c
    let (tId, st2) := flatten st1 t
    let (fId, st3) := flatten st2 f
    st3.addNode { op := .select cId tId fId, shape := s }
  | .reshape e _ =>
    let (inId, st') := flatten st e
    st'.addNode { op := .reshape inId, shape := s }
  | .transpose e perm =>
    let (inId, st') := flatten st e
    st'.addNode { op := .transpose inId (perm.toList.map (·.val)), shape := s }
  | .broadcast e tgt _ =>
    let (inId, st') := flatten st e
    let srcShape := if h : inId < st'.nodes.size then st'.nodes[inId].shape else []
    st'.addNode { op := .broadcast inId srcShape tgt, shape := s }
  | .slice e starts sizes _ =>
    let (inId, st') := flatten st e
    st'.addNode { op := .slice inId starts sizes, shape := s }
  | .pad e padding fv _ =>
    let (inId, st') := flatten st e
    st'.addNode { op := .pad inId padding fv, shape := s }
  | .concat e₁ e₂ axis _ =>
    let (id1, st1) := flatten st e₁
    let (id2, st2) := flatten st1 e₂
    st2.addNode { op := .concat id1 id2 axis.val, shape := s }
  | .gather e axis indices =>
    let (inId, st') := flatten st e
    st'.addNode { op := .gather inId axis.val indices.toArray, shape := s }
  | .reduce op axis e =>
    let (inId, st') := flatten st e
    st'.addNode { op := .reduce op axis.val inId, shape := s }
  | .scan op axis e =>
    let (inId, st') := flatten st e
    st'.addNode { op := .scan op axis.val inId, shape := s }
  | .einsum subsA subsB subsOut eA eB _ =>
    let (idA, st1) := flatten st eA
    let (idB, st2) := flatten st1 eB
    st2.addNode { op := .einsum subsA subsB subsOut idA idB, shape := s }

/-- Convert a tensor expression into an executable DAG plan.
    Flattens the expression tree and records the output node index. -/
def TensorExpr.toExecPlan (e : TensorExpr Float s) : ExecPlan :=
  let (outId, st) := flatten {} e
  { nodes := st.nodes, output := outId }

/-- Execute a single DAG node given already-computed buffers. -/
private def executeNode (node : DagNode) (buffers : Array (Array Float))
    (nodes : Array DagNode) (env : Env Float) : Except String (Array Float) :=
  match node.op with
  | .literal data => .ok data
  | .fill val size => .ok (Array.replicate size val)
  | .var name =>
    match env.lookup name node.shape with
    | some v => .ok v.toArray
    | none => .error s!"Variable '{name}' not found"
  | .unary op inId =>
    let buf := getBuffer buffers inId
    .ok (buf.map op.evalFloat)
  | .binary op l r =>
    let bl := getBuffer buffers l
    let br := getBuffer buffers r
    .ok (Array.zipWith op.evalFloat bl br)
  | .smul c inId =>
    let buf := getBuffer buffers inId
    .ok (buf.map (c * ·))
  | .select cId tId fId =>
    let bc := getBuffer buffers cId
    let bt := getBuffer buffers tId
    let bf := getBuffer buffers fId
    .ok (bc.mapFinIdx fun i v _ => if v != 0.0 then bt.getD i 0.0 else bf.getD i 0.0)
  | .reshape inId => .ok (getBuffer buffers inId)
  | .transpose inId perm =>
    let inBuf := getBuffer buffers inId
    let inShape := getNodeShape nodes inId
    let outShape := node.shape
    let outSize := Shape.product outShape
    .ok <| Id.run do
      let mut out := Array.replicate outSize 0.0
      for idx in [:outSize] do
        let outMulti := Lithe.linearToMulti outShape idx
        -- Build input multi-index: input[perm[i]] = outMulti[i]
        let mut inMulti := List.replicate inShape.length 0
        for i in [:perm.length] do
          let p := perm.getD i 0
          inMulti := setAt inMulti p (outMulti.getD i 0)
        let inLin := Lithe.multiToLinear inShape inMulti
        out := out.set! idx (inBuf.getD inLin 0.0)
      return out
  | .broadcast inId srcShape tgtShape =>
    let inBuf := getBuffer buffers inId
    let outSize := Shape.product tgtShape
    .ok <| Id.run do
      let mut out := Array.replicate outSize 0.0
      for idx in [:outSize] do
        let outMulti := Lithe.linearToMulti tgtShape idx
        let inMulti := List.zipWith (fun d o => if d == 1 then 0 else o) srcShape outMulti
        let inLin := Lithe.multiToLinear srcShape inMulti
        out := out.set! idx (inBuf.getD inLin 0.0)
      return out
  | .slice inId starts sizes =>
    let inBuf := getBuffer buffers inId
    let inShape := getNodeShape nodes inId
    let outSize := Shape.product sizes
    .ok <| Id.run do
      let mut out := Array.replicate outSize 0.0
      for idx in [:outSize] do
        let outMulti := Lithe.linearToMulti sizes idx
        let inMulti := List.zipWith (· + ·) starts outMulti
        let inLin := Lithe.multiToLinear inShape inMulti
        out := out.set! idx (inBuf.getD inLin 0.0)
      return out
  | .pad inId padding fillVal =>
    let inBuf := getBuffer buffers inId
    let inShape := getNodeShape nodes inId
    let outShape := node.shape
    let outSize := Shape.product outShape
    .ok <| Id.run do
      let mut out := Array.replicate outSize 0.0
      for idx in [:outSize] do
        let outMulti := Lithe.linearToMulti outShape idx
        match goCheckPad inShape padding outMulti with
        | some inMulti =>
          let inLin := Lithe.multiToLinear inShape inMulti
          out := out.set! idx (inBuf.getD inLin fillVal)
        | none => out := out.set! idx fillVal
      return out
  | .concat lId rId axis =>
    let lBuf := getBuffer buffers lId
    let rBuf := getBuffer buffers rId
    let lShape := getNodeShape nodes lId
    let rShape := getNodeShape nodes rId
    let outShape := node.shape
    let outSize := Shape.product outShape
    let dim₁ := lShape.getD axis 0
    .ok <| Id.run do
      let mut out := Array.replicate outSize 0.0
      for idx in [:outSize] do
        let outMulti := Lithe.linearToMulti outShape idx
        let axisIdx := outMulti.getD axis 0
        if axisIdx < dim₁ then
          let inLin := Lithe.multiToLinear lShape outMulti
          out := out.set! idx (lBuf.getD inLin 0.0)
        else
          let inMulti := mapAtList outMulti axis (· - dim₁)
          let inLin := Lithe.multiToLinear rShape inMulti
          out := out.set! idx (rBuf.getD inLin 0.0)
      return out
  | .gather inId axis indices =>
    let inBuf := getBuffer buffers inId
    let inShape := getNodeShape nodes inId
    let outShape := node.shape
    let outSize := Shape.product outShape
    .ok <| Id.run do
      let mut out := Array.replicate outSize 0.0
      for idx in [:outSize] do
        let outMulti := Lithe.linearToMulti outShape idx
        let gatherIdx := outMulti.getD axis 0
        let srcIdx := indices.getD gatherIdx 0
        let inMulti := setAt outMulti axis srcIdx
        let inLin := Lithe.multiToLinear inShape inMulti
        out := out.set! idx (inBuf.getD inLin 0.0)
      return out
  | .reduce op axis inId =>
    let inBuf := getBuffer buffers inId
    let inShape := getNodeShape nodes inId
    let outShape := node.shape
    let outSize := Shape.product outShape
    let axisSize := inShape.getD axis 1
    .ok <| Id.run do
      let mut out := Array.replicate outSize 0.0
      for idx in [:outSize] do
        let outMulti := Lithe.linearToMulti outShape idx
        let baseMulti := insertAt outMulti axis 0
        let mut acc := op.identityFloat
        for k in [:axisSize] do
          let inMulti := setAt baseMulti axis k
          let inLin := Lithe.multiToLinear inShape inMulti
          acc := op.combineFloat acc (inBuf.getD inLin 0.0)
        out := out.set! idx acc
      return out
  | .scan op axis inId =>
    let inBuf := getBuffer buffers inId
    let inShape := getNodeShape nodes inId
    let outSize := Shape.product inShape
    .ok <| Id.run do
      let mut out := Array.replicate outSize 0.0
      for idx in [:outSize] do
        let outMulti := Lithe.linearToMulti inShape idx
        let pos := outMulti.getD axis 0
        let mut acc := op.identityFloat
        for k in [:pos + 1] do
          let inMulti := setAt outMulti axis k
          let inLin := Lithe.multiToLinear inShape inMulti
          acc := op.combineFloat acc (inBuf.getD inLin 0.0)
        out := out.set! idx acc
      return out
  | .einsum subsA subsB subsOut lId rId =>
    let lBuf := getBuffer buffers lId
    let rBuf := getBuffer buffers rId
    let sA := getNodeShape nodes lId
    let sB := getNodeShape nodes rId
    let sOut := node.shape
    let outSize := Shape.product sOut
    let allLabels := (subsA ++ subsB).eraseDups
    let contractedLabels := allLabels.filter (!subsOut.contains ·)
    let contractedDims := contractedLabels.map fun label =>
      match findIdx subsA label with
      | some idx => sA.getD idx 1
      | none =>
        match findIdx subsB label with
        | some idx => sB.getD idx 1
        | none => 1
    let contractedProduct := contractedDims.foldl (· * ·) 1
    .ok <| Id.run do
      let mut out := Array.replicate outSize 0.0
      for idx in [:outSize] do
        let outMulti := Lithe.linearToMulti sOut idx
        let labelMap := List.zip subsOut outMulti
        let mut sum : Float := 0.0
        for cIdx in [:contractedProduct] do
          let cMulti := Lithe.linearToMulti contractedDims cIdx
          let fullMap := labelMap ++ List.zip contractedLabels cMulti
          let aMulti := subsA.map fun label =>
            match fullMap.find? (fun p => p.1 == label) with
            | some (_, v) => v
            | none => 0
          let bMulti := subsB.map fun label =>
            match fullMap.find? (fun p => p.1 == label) with
            | some (_, v) => v
            | none => 0
          let aLin := Lithe.multiToLinear sA aMulti
          let bLin := Lithe.multiToLinear sB bMulti
          let aVal := lBuf.getD aLin 0.0
          let bVal := rBuf.getD bLin 0.0
          sum := sum + aVal * bVal
        out := out.set! idx sum
      return out
where
  goCheckPad : List Nat → List (Nat × Nat) → List Nat → Option (List Nat)
    | [], _, _ => some []
    | _, [], _ => some []
    | _, _, [] => some []
    | d :: ds, (lo, _) :: ps, o :: os =>
      if o < lo || o >= lo + d then none
      else match goCheckPad ds ps os with
        | some rest => some ((o - lo) :: rest)
        | none => none
  mapAtList (l : List Nat) (pos : Nat) (f : Nat → Nat) : List Nat :=
    match l, pos with
    | [], _ => []
    | x :: xs, 0 => f x :: xs
    | x :: xs, n + 1 => x :: mapAtList xs n f

/-- Execute the DAG plan sequentially, computing each node's output buffer
    in topological order. Returns the final output buffer. -/
def ExecPlan.execute (plan : ExecPlan) (env : Env Float := Env.empty)
    : Except String (Array Float) := do
  let mut buffers : Array (Array Float) := #[]
  for node in plan.nodes do
    let result ← executeNode node buffers plan.nodes env
    buffers := buffers.push result
  if h : plan.output < buffers.size then
    .ok buffers[plan.output]
  else
    .error "Invalid output node"

/-- IO wrapper for `execute` that converts `Except` errors to `IO.Error`. -/
def ExecPlan.executeIO (plan : ExecPlan) (env : Env Float := Env.empty) : IO (Array Float) := do
  match plan.execute env with
  | .ok result => pure result
  | .error e => throw (IO.Error.userError e)

end Lithe.Backend
