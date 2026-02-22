/-
  Lithe/Backend/CPU.lean — DAG-based evaluation with subexpression sharing
-/
import Lithe.Shape
import Lithe.Scalar
import Lithe.Ops
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
    (env : Env Float) : Except String (Array Float) :=
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
  | _ =>
    .error "CPU backend: op not yet implemented"

/-- Execute the DAG plan sequentially, computing each node's output buffer
    in topological order. Returns the final output buffer. -/
def ExecPlan.execute (plan : ExecPlan) (env : Env Float := Env.empty)
    : Except String (Array Float) := do
  let mut buffers : Array (Array Float) := #[]
  for node in plan.nodes do
    let result ← executeNode node buffers env
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
