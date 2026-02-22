import Lithe

open TensorExpr Tensor

namespace Examples

def examples : IO Unit := do
  IO.println "=== A few more examples ==="

  let a: Tensor Float [3] := .literal [3] #v[1, 2, 3]
  let b: Tensor Float [3] := .literal [3] #v[4, 5, 6]
  let c: Tensor Float [3] := a + b
  IO.println s!"a + b = {repr c.eval.toList}"

  IO.println "=== Done ==="

end Examples
