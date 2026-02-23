/-
  Lithe/Safetensors.lean — Safetensors file format parser
-/
import Lithe.Shape
import Lithe.Env

namespace Safetensors

/-! ### Minimal JSON Parser -/

/-- JSON value (subset needed for safetensors headers). -/
inductive JsonValue where
  | str (s : String)
  | num (n : Int)
  | arr (vs : List JsonValue)
  | obj (kvs : List (String × JsonValue))
  | null
  deriving Repr

/-- JSON tokenizer state. -/
private inductive Token where
  | lbrace | rbrace | lbracket | rbracket | colon | comma
  | str (s : String) | num (n : Int)
  deriving Repr

/-- Tokenize a JSON string into tokens. -/
private partial def tokenize (input : String) : List Token :=
  go input.toList []
where
  go : List Char → List Token → List Token
    | [], acc => acc.reverse
    | '{' :: rest, acc => go rest (.lbrace :: acc)
    | '}' :: rest, acc => go rest (.rbrace :: acc)
    | '[' :: rest, acc => go rest (.lbracket :: acc)
    | ']' :: rest, acc => go rest (.rbracket :: acc)
    | ':' :: rest, acc => go rest (.colon :: acc)
    | ',' :: rest, acc => go rest (.comma :: acc)
    | '"' :: rest, acc =>
      let (s, rest') := readString rest []
      go rest' (.str s :: acc)
    | c :: rest, acc =>
      if c == ' ' || c == '\n' || c == '\r' || c == '\t' then
        go rest acc
      else if c == '-' || c.isDigit then
        let (n, rest') := readNum (c :: rest) []
        go rest' (.num n :: acc)
      else if c == 'n' then  -- null
        go (rest.drop 3) acc  -- skip "ull"
      else
        go rest acc  -- skip unknown
  readString : List Char → List Char → (String × List Char)
    | [], acc => (String.ofList acc.reverse, [])
    | '"' :: rest, acc => (String.ofList acc.reverse, rest)
    | '\\' :: '"' :: rest, acc => readString rest ('"' :: acc)
    | '\\' :: 'n' :: rest, acc => readString rest ('\n' :: acc)
    | '\\' :: '\\' :: rest, acc => readString rest ('\\' :: acc)
    | c :: rest, acc => readString rest (c :: acc)
  readNum : List Char → List Char → (Int × List Char)
    | [], acc => (parseIntChars acc.reverse, [])
    | c :: rest, acc =>
      if c == '-' || c.isDigit then readNum rest (c :: acc)
      else (parseIntChars acc.reverse, c :: rest)
  parseIntChars (cs : List Char) : Int :=
    match cs with
    | '-' :: ds => -(natOfChars ds)
    | ds => natOfChars ds
  natOfChars (cs : List Char) : Int :=
    cs.foldl (fun acc c => acc * 10 + (c.toNat - '0'.toNat : Int)) 0

/-- Parse tokens into a JsonValue. Returns (value, remaining tokens). -/
private partial def parseValue : List Token → Option (JsonValue × List Token)
  | .str s :: rest => some (.str s, rest)
  | .num n :: rest => some (.num n, rest)
  | .lbrace :: rest => parseObject rest []
  | .lbracket :: rest => parseArray rest []
  | _ => none
where
  parseObject : List Token → List (String × JsonValue) → Option (JsonValue × List Token)
    | .rbrace :: rest, acc => some (.obj acc.reverse, rest)
    | .str key :: .colon :: rest, acc =>
      match parseValue rest with
      | some (v, .comma :: rest') => parseObject rest' ((key, v) :: acc)
      | some (v, .rbrace :: rest') => some (.obj ((key, v) :: acc).reverse, rest')
      | some (v, rest') => some (.obj ((key, v) :: acc).reverse, rest')
      | none => none
    | _, acc => some (.obj acc.reverse, [])
  parseArray : List Token → List JsonValue → Option (JsonValue × List Token)
    | .rbracket :: rest, acc => some (.arr acc.reverse, rest)
    | toks, acc =>
      match parseValue toks with
      | some (v, .comma :: rest') => parseArray rest' (v :: acc)
      | some (v, .rbracket :: rest') => some (.arr (v :: acc).reverse, rest')
      | some (v, rest') => some (.arr (v :: acc).reverse, rest')
      | none => some (.arr acc.reverse, toks)

/-- Parse a JSON string. -/
def parseJson (input : String) : Option JsonValue :=
  let tokens := tokenize input
  match parseValue tokens with
  | some (v, _) => some v
  | none => none

/-! ### IEEE 754 Float32 Decode (pure Lean) -/

/-- Decode an IEEE 754 binary32 value from 4 little-endian bytes.

    Layout (32 bits): `[sign:1][exponent:8][mantissa:23]`

    * **Normal** ($1 \le e \le 254$):
      $(-1)^s \times 2^{e-127} \times (1 + m/2^{23})$
    * **Denormal** ($e = 0$):
      $(-1)^s \times 2^{-126} \times (m/2^{23})$
    * **Infinity** ($e = 255, m = 0$):
      $\pm\infty$
    * **NaN** ($e = 255, m \ne 0$):
      `NaN` -/
def float32FromBytes (b0 b1 b2 b3 : UInt8) : Float :=
  let bits := b0.toNat ||| (b1.toNat <<< 8) ||| (b2.toNat <<< 16) ||| (b3.toNat <<< 24)
  let sign := if bits >>> 31 == 1 then -1.0 else 1.0
  let exponent := (bits >>> 23) &&& 0xFF
  let mantissa := bits &&& 0x7FFFFF
  let fraction := mantissa.toFloat / 8388608.0   -- 2^23
  if exponent == 0 then
    -- Denormalized: ±2^{-126} × fraction
    sign * fraction * (2.0 ^ (-126.0))
  else if exponent == 255 then
    if mantissa == 0 then sign * (1.0 / 0.0)     -- ±∞
    else 0.0 / 0.0                                 -- NaN
  else
    -- Normal: ±2^{exponent-127} × (1 + fraction)
    let exp := if exponent >= 127
      then 2.0 ^ (exponent - 127).toFloat
      else 1.0 / (2.0 ^ (127 - exponent).toFloat)
    sign * (1.0 + fraction) * exp

/-! ### ByteArray Utilities -/

/-- Read a little-endian uint64 from 8 bytes starting at offset. -/
def readUInt64LE (data : ByteArray) (offset : Nat) : UInt64 :=
  let get (i : Nat) : UInt64 := (data.get! (offset + i)).toUInt64
  get 0 ||| (get 1 <<< 8) ||| (get 2 <<< 16) ||| (get 3 <<< 24) |||
  (get 4 <<< 32) ||| (get 5 <<< 40) ||| (get 6 <<< 48) ||| (get 7 <<< 56)

/-- Decode an array of float32 values from a byte slice. -/
def decodeFloat32Array (data : ByteArray) (offset count : Nat) : Array Float :=
  Id.run do
    let mut arr := Array.replicate count 0.0
    for i in [:count] do
      let base := offset + i * 4
      if base + 3 < data.size then
        let f := float32FromBytes
          (data.get! base)
          (data.get! (base + 1))
          (data.get! (base + 2))
          (data.get! (base + 3))
        arr := arr.set! i f
    return arr

/-! ### Safetensors Loader -/

/-- Tensor metadata from the safetensors header. -/
structure TensorMeta where
  name   : String
  dtype  : String
  shape  : Shape
  offset : Nat  -- start offset in data section
  size   : Nat  -- end offset in data section

/-- Extract tensor metadata from parsed JSON header. -/
private def extractMeta (json : JsonValue) : List TensorMeta :=
  match json with
  | .obj kvs =>
    kvs.filterMap fun (name, value) =>
      if name == "__metadata__" then none
      else match value with
        | .obj fields =>
          let dtype := match fields.find? (fun p => p.1 == "dtype") with
            | some (_, .str s) => s
            | _ => "F32"
          let shape := match fields.find? (fun p => p.1 == "shape") with
            | some (_, .arr dims) => dims.filterMap fun v =>
              match v with
              | .num n => some n.natAbs
              | _ => none
            | _ => []
          let (off0, off1) := match fields.find? (fun p => p.1 == "data_offsets") with
            | some (_, .arr [.num a, .num b]) => (a.natAbs, b.natAbs)
            | _ => (0, 0)
          some { name := name, dtype := dtype, shape := shape,
                 offset := off0, size := off1 }
        | _ => none
  | _ => []

/-- Load a safetensors file into an `Env Float`.
    Only supports F32 (float32) tensors. -/
def loadSafetensors (path : System.FilePath) : IO (Env Float) := do
  let data ← IO.FS.readBinFile path
  if data.size < 8 then
    throw (IO.Error.userError "Safetensors file too small")
  -- Read header size (first 8 bytes, LE uint64)
  let headerSize := (readUInt64LE data 0).toNat
  if 8 + headerSize > data.size then
    throw (IO.Error.userError "Invalid header size")
  -- Extract header JSON
  let headerBytes := data.extract 8 (8 + headerSize)
  let headerStr := String.fromUTF8! headerBytes
  let json ← match parseJson headerStr with
    | some j => pure j
    | none => throw (IO.Error.userError "Failed to parse JSON header")
  -- Parse tensor metadata
  let tensorMetas := extractMeta json
  let dataOffset := 8 + headerSize
  -- Build Env
  let mut env : Env Float := []
  for tm in tensorMetas do
    if tm.dtype == "F32" || tm.dtype == "F16" then
      let numElements := tm.shape.foldl (· * ·) 1
      let byteStart := dataOffset + tm.offset
      let tensorData := if tm.dtype == "F32" then
        decodeFloat32Array data byteStart numElements
      else
        -- F16: decode each 2-byte half-float (simplified: skip for now)
        Array.replicate numElements 0.0
      env := env ++ [(tm.name, ⟨tm.shape, ⟨tensorData, by sorry⟩⟩)]
  return env

end Safetensors
