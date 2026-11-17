//! Read what matters in an `.onnx` model: operators, input/output shapes and the
//! parameter size - the facts that decide whether it can become a microWakeWord
//! `.tflite` for the device (see [`crate::wake_word`]).
//!
//! The protobuf wire format is parsed by hand, so this adds no dependencies and
//! no `prost` codegen. It reads the fields of `ModelProto`/`GraphProto` it needs
//! and skips the rest, so it copes with any opset.
//!
//! ```no_run
//! # fn demo() -> anyhow::Result<()> {
//! let bytes = std::fs::read("model.onnx")?;
//! let info = havoice::onnx::inspect_onnx(&bytes)?;
//! println!("ops: {}", info.ops.iter().map(|(op, n)| format!("{op}x{n}")).collect::<Vec<_>>().join(" "));
//! for c in havoice::onnx::checks(&info) {
//!     println!("{} {}", if c.ok { "ok  " } else { "FAIL" }, c.what);
//! }
//! # Ok(()) }
//! ```

use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::wake_word::Check;

/// One graph input or output.
#[derive(Debug, Clone, PartialEq)]
pub struct OnnxTensor {
    /// tensor name in the graph
    pub name: String,
    /// ONNX element type, e.g. `float32`
    pub elem_type: &'static str,
    /// dimensions; `-1` for a symbolic/unknown dimension
    pub shape: Vec<i64>,
}

/// What [`inspect_onnx`] read from an `.onnx` file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OnnxInfo {
    /// ONNX IR version
    pub ir_version: u64,
    /// what exported it, e.g. `pytorch`
    pub producer: String,
    /// `(domain, version)` of every opset import (empty domain = `ai.onnx`)
    pub opsets: Vec<(String, i64)>,
    /// operator types and how often each occurs, sorted by name
    pub ops: Vec<(String, usize)>,
    /// graph inputs
    pub inputs: Vec<OnnxTensor>,
    /// graph outputs
    pub outputs: Vec<OnnxTensor>,
    /// number of weight tensors
    pub initializers: usize,
    /// total size of the weights in bytes (the device's models are tens of kB)
    pub parameter_bytes: u64,
}

/// Read the parts of an `.onnx` file that matter for a microWakeWord conversion.
pub fn inspect_onnx(bytes: &[u8]) -> Result<OnnxInfo> {
    ensure_onnx(bytes)?;
    let mut top = Reader::new(bytes);
    let mut info = OnnxInfo::default();
    let mut graph_seen = false;
    while let Some((field, wt)) = top.tag()? {
        match (field, wt) {
            (1, 0) => info.ir_version = top.varint()?,
            (2, 2) => info.producer = text(top.bytes()?),
            (3, 2) => {
                let _ = top.bytes()?; // producer_version
            }
            (7, 2) => {
                graph_seen = true;
                let mut g = top.message()?;
                let mut ops: HashMap<String, usize> = HashMap::new();
                while let Some((f, w)) = g.tag()? {
                    match (f, w) {
                        (1, 2) => {
                            let mut n = g.message()?;
                            while let Some((nf, nw)) = n.tag()? {
                                match (nf, nw) {
                                    (4, 2) => *ops.entry(text(n.bytes()?)).or_default() += 1,
                                    _ => n.skip(nw)?,
                                }
                            }
                        }
                        (5, 2) => {
                            let t = g.message()?;
                            read_initializer(&mut info, t)?;
                        }
                        (11, 2) => {
                            let v = g.message()?;
                            if let Some(t) = value_info(v)? {
                                info.inputs.push(t);
                            }
                        }
                        (12, 2) => {
                            let v = g.message()?;
                            if let Some(t) = value_info(v)? {
                                info.outputs.push(t);
                            }
                        }
                        _ => g.skip(w)?,
                    }
                }
                info.ops = ops.into_iter().collect();
                info.ops.sort();
            }
            (8, 2) => {
                let mut m = top.message()?;
                let mut domain = String::new();
                let mut version = 0;
                while let Some((f, w)) = m.tag()? {
                    match (f, w) {
                        (1, 2) => domain = text(m.bytes()?),
                        (2, 0) => version = m.varint()? as i64,
                        _ => m.skip(w)?,
                    }
                }
                info.opsets.push((domain, version));
            }
            _ => top.skip(wt)?,
        }
    }
    if !graph_seen {
        bail!("no ONNX graph in this file");
    }
    Ok(info)
}

/// Heuristics for "can this become a microWakeWord model?". Failing checks here mean
/// conversion is pointless; warnings mean the converted model will likely not detect
/// well, or not at all, without extra work (e.g. calibration data for quantization).
pub fn checks(info: &OnnxInfo) -> Vec<Check> {
    let mut checks = Vec::new();
    let mut check = |ok: bool, what: String| {
        checks.push(Check {
            ok,
            warning: false,
            what,
        })
    };
    let opset = info
        .opsets
        .iter()
        .find(|(d, _)| d.is_empty() || d == "ai.onnx")
        .map(|(_, v)| *v);
    check(
        opset.is_some(),
        match opset {
            Some(v) => format!("ONNX opset {v}"),
            None => "has an ai.onnx opset import".into(),
        },
    );
    check(
        info.inputs.len() == 1,
        format!("exactly one graph input (has {})", info.inputs.len()),
    );
    let shape = info.inputs.first().map(|t| &t.shape);
    match shape {
        Some(s) if s.len() == 3 => check(
            *s.last().unwrap() == 40,
            format!(
                "input ends in the 40 mel bins (got {s:?}) - microWakeWord eats [1, stride, 40] melspectrogram frames"
            ),
        ),
        Some(s) => check(
            false,
            format!("input is rank 3 [1, stride, 40] (got {s:?})"),
        ),
        None => check(false, "graph declares an input".into()),
    }
    check(
        info.outputs.len() == 1,
        format!("exactly one graph output (has {})", info.outputs.len()),
    );
    const WARN_BYTES: u64 = 500_000;
    const FAIL_BYTES: u64 = 5_000_000;
    check(
        info.parameter_bytes <= FAIL_BYTES,
        format!(
            "weights within device reach ({} kB) - official models are 20-100 kB, the tensor arena may not exceed 200 kB",
            info.parameter_bytes / 1000
        ),
    );
    if info.parameter_bytes > WARN_BYTES && info.parameter_bytes <= FAIL_BYTES {
        checks.push(Check {
            ok: false,
            warning: true,
            what: format!(
                "{} kB of weights is large for the device - expect a tight or failed fit",
                info.parameter_bytes / 1000
            ),
        });
    }
    let awkward = ["Loop", "If", "Scan", "NonZero", "Unique", "SparseToDense"]
        .iter()
        .filter(|op| info.ops.iter().any(|(o, _)| o == **op))
        .map(|op| op.to_string())
        .collect::<Vec<_>>();
    if !awkward.is_empty() {
        checks.push(Check {
            ok: false,
            warning: true,
            what: format!(
                "control-flow-ish ops ({}) often can't be lowered to TFLite",
                awkward.join(", ")
            ),
        });
    }
    checks
}

fn read_initializer(info: &mut OnnxInfo, mut t: Reader) -> Result<()> {
    info.initializers += 1;
    let mut dims = Vec::new();
    let mut elem_type = 0i64;
    let mut raw = 0u64;
    while let Some((f, w)) = t.tag()? {
        match (f, w) {
            (1, 2) => {
                let packed = t.bytes()?;
                let mut r = Reader::new(packed);
                while r.pos < r.b.len() {
                    dims.push(r.varint()?);
                }
            }
            (1, 0) => dims.push(t.varint()?),
            (2, 0) => elem_type = t.varint()? as i64,
            (9, 2) => raw += t.bytes()?.len() as u64,
            _ => t.skip(w)?,
        }
    }
    info.parameter_bytes += if raw > 0 {
        raw
    } else {
        dims.iter()
            .product::<u64>()
            .saturating_mul(type_size(elem_type))
    };
    Ok(())
}

fn type_size(elem_type: i64) -> u64 {
    match elem_type {
        2 | 3 => 1,       // uint8, int8
        4 | 5 | 9 => 2,   // uint16, int16, bool(ish)
        1 | 6 | 12 => 4,  // float32, int32, uint32
        7 | 11 | 13 => 8, // int64, double, uint64
        10 => 2,          // float16
        16 => 2,          // bfloat16
        _ => 4,
    }
}

/// `(name, type)` of a ValueInfoProto; `None` when the type is a non-tensor.
fn value_info(mut v: Reader) -> Result<Option<OnnxTensor>> {
    let mut name = String::new();
    let mut tensor = None;
    while let Some((f, w)) = v.tag()? {
        match (f, w) {
            (1, 2) => name = text(v.bytes()?),
            (2, 2) => {
                let mut t = v.message()?;
                while let Some((tf, tw)) = t.tag()? {
                    if tf == 1 && tw == 2 {
                        tensor = Some(t.message()?);
                    } else {
                        t.skip(tw)?;
                    }
                }
            }
            _ => v.skip(w)?,
        }
    }
    let Some(mut t) = tensor else {
        return Ok(None);
    };
    let mut elem_type = 0i64;
    let mut shape = Vec::new();
    while let Some((f, w)) = t.tag()? {
        match (f, w) {
            (1, 0) => elem_type = t.varint()? as i64,
            (2, 2) => {
                let mut s = t.message()?;
                while let Some((sf, sw)) = s.tag()? {
                    if sf == 1 && sw == 2 {
                        let mut d = s.message()?;
                        let mut value = -1i64;
                        while let Some((df, dw)) = d.tag()? {
                            match (df, dw) {
                                (1, 0) => value = d.varint()? as i64,
                                (2, 2) => {
                                    let _ = d.bytes()?;
                                    value = -1;
                                }
                                _ => d.skip(dw)?,
                            }
                        }
                        shape.push(value);
                    } else {
                        s.skip(sw)?;
                    }
                }
            }
            _ => t.skip(w)?,
        }
    }
    Ok(Some(OnnxTensor {
        name,
        elem_type: elem_name(elem_type),
        shape,
    }))
}

fn elem_name(t: i64) -> &'static str {
    match t {
        1 => "float32",
        2 => "uint8",
        3 => "int8",
        4 => "uint16",
        5 => "int16",
        6 => "int32",
        7 => "int64",
        9 => "bool",
        10 => "float16",
        11 => "float64",
        12 => "uint32",
        13 => "uint64",
        16 => "bfloat16",
        _ => "unknown",
    }
}

fn ensure_onnx(bytes: &[u8]) -> Result<()> {
    anyhow::ensure!(
        bytes.len() > 16,
        "not an ONNX file (smaller than any model)"
    );
    Ok(())
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// Minimal protobuf wire-format cursor over a fixed byte slice.
struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let (mut v, mut shift, mut n) = (0u64, 0u32, 0);
        loop {
            let byte = *self
                .b
                .get(self.pos)
                .ok_or_else(|| anyhow::anyhow!("truncated ONNX"))?;
            self.pos += 1;
            n += 1;
            v |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
            anyhow::ensure!(n < 10, "bad varint in ONNX");
            shift += 7;
        }
    }

    /// `(field number, wire type)` of the next field, `None` at the end.
    fn tag(&mut self) -> Result<Option<(u32, u8)>> {
        if self.pos >= self.b.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        Ok(Some(((key >> 3) as u32, (key & 7) as u8)))
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()? as usize;
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| anyhow::anyhow!("bad length in ONNX"))?;
        let out = self
            .b
            .get(self.pos..end)
            .ok_or_else(|| anyhow::anyhow!("truncated ONNX"))?;
        self.pos = end;
        Ok(out)
    }

    fn message(&mut self) -> Result<Reader<'a>> {
        Ok(Reader::new(self.bytes()?))
    }

    fn skip(&mut self, wire_type: u8) -> Result<()> {
        match wire_type {
            0 => {
                self.varint()?;
            }
            1 => {
                self.pos = self
                    .pos
                    .checked_add(8)
                    .ok_or_else(|| anyhow::anyhow!("bad length in ONNX"))?;
                anyhow::ensure!(self.pos <= self.b.len(), "truncated ONNX");
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.pos = self
                    .pos
                    .checked_add(4)
                    .ok_or_else(|| anyhow::anyhow!("bad length in ONNX"))?;
                anyhow::ensure!(self.pos <= self.b.len(), "truncated ONNX");
            }
            3 => {
                let mut depth = 1;
                while depth > 0 {
                    let Some((_, wt)) = self.tag()? else {
                        bail!("unterminated group in ONNX");
                    };
                    match wt {
                        3 => depth += 1,
                        4 => depth -= 1,
                        _ => self.skip(wt)?,
                    }
                }
            }
            4 => {}
            wt => bail!("ONNX uses protobuf wire type {wt}, which this reader skips"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn vi(field: u32, v: u64, out: &mut Vec<u8>) {
        varint(u64::from(field) << 3, out);
        varint(v, out);
    }

    fn ld(field: u32, payload: &[u8], out: &mut Vec<u8>) {
        varint((u64::from(field) << 3) | 2, out);
        varint(payload.len() as u64, out);
        out.extend_from_slice(payload);
    }

    /// A tiny but valid ModelProto: one Conv node, input float32 [1, 16, 40],
    /// output float32 [1, 1, 1], one 16-byte weight, opset 17.
    fn mini_onnx() -> Vec<u8> {
        let dim = |v: i64, out: &mut Vec<u8>| {
            let mut d = Vec::new();
            vi(1, v as u64, &mut d); // Dimension.dim_value
            ld(1, &d, out); // TensorShapeProto.dim
        };
        let mut shape = Vec::new();
        dim(1, &mut shape);
        dim(16, &mut shape);
        dim(40, &mut shape);
        let mut out_shape = Vec::new();
        dim(1, &mut out_shape);
        dim(1, &mut out_shape);
        dim(1, &mut out_shape);

        let mut tensor_type = Vec::new();
        vi(1, 1, &mut tensor_type); // float32
        ld(2, &shape, &mut tensor_type);
        let mut out_type = Vec::new();
        vi(1, 1, &mut out_type);
        ld(2, &out_shape, &mut out_type);

        let mut input = Vec::new();
        ld(1, b"x", &mut input);
        let mut input_type = Vec::new();
        ld(1, &tensor_type, &mut input_type); // TypeProto.tensor_type
        ld(2, &input_type, &mut input);
        let mut output = Vec::new();
        ld(1, b"y", &mut output);
        let mut output_type = Vec::new();
        ld(1, &out_type, &mut output_type);
        ld(2, &output_type, &mut output);

        let mut node = Vec::new();
        ld(4, b"Conv", &mut node);

        let mut initializer = Vec::new();
        let mut dims = Vec::new();
        varint(2, &mut dims);
        varint(2, &mut dims);
        ld(1, &dims, &mut initializer);
        vi(2, 1, &mut initializer); // float32
        ld(8, b"W", &mut initializer);
        ld(9, &[0u8; 16], &mut initializer); // raw_data

        let mut opset = Vec::new();
        vi(2, 17, &mut opset);

        let mut graph = Vec::new();
        ld(1, &node, &mut graph);
        ld(5, &initializer, &mut graph);
        ld(11, &input, &mut graph);
        ld(12, &output, &mut graph);

        let mut model = Vec::new();
        vi(1, 8, &mut model); // ir_version
        ld(2, b"unittest", &mut model); // producer_name
        ld(3, b"1.0", &mut model); // producer_version
        ld(7, &graph, &mut model); // graph
        ld(8, &opset, &mut model); // opset_import
        model
    }

    #[test]
    fn reads_ops_shapes_and_weights() {
        let info = inspect_onnx(&mini_onnx()).unwrap();
        assert_eq!(info.ir_version, 8);
        assert_eq!(info.producer, "unittest");
        assert_eq!(info.opsets, vec![("".to_string(), 17)]);
        assert_eq!(info.ops, vec![("Conv".to_string(), 1)]);
        assert_eq!(info.inputs.len(), 1);
        assert_eq!(info.inputs[0].name, "x");
        assert_eq!(info.inputs[0].elem_type, "float32");
        assert_eq!(info.inputs[0].shape, vec![1, 16, 40]);
        assert_eq!(info.outputs[0].shape, vec![1, 1, 1]);
        assert_eq!(info.initializers, 1);
        assert_eq!(info.parameter_bytes, 16);
    }

    #[test]
    fn mini_model_passes_the_checks() {
        let info = inspect_onnx(&mini_onnx()).unwrap();
        let checks = checks(&info);
        assert!(checks.iter().all(|c| c.ok), "{checks:?}");
    }

    /// Reads a real ONNX model when one is lying around (HAVOICE_TEST_ONNX=path).
    #[test]
    fn real_onnx_if_available() {
        let Ok(path) = std::env::var("HAVOICE_TEST_ONNX") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let info = inspect_onnx(&bytes).unwrap();
        println!("{info:#?}");
        for c in checks(&info) {
            println!(
                "{} {}",
                if c.ok {
                    "ok  "
                } else if c.warning {
                    "warn"
                } else {
                    "FAIL"
                },
                c.what
            );
        }
    }

    #[test]
    fn garbage_is_rejected() {
        assert!(inspect_onnx(b"not onnx at all").is_err());
        assert!(inspect_onnx(&[]).is_err());
        let mut truncated = mini_onnx();
        truncated.truncate(truncated.len() - 3);
        assert!(inspect_onnx(&truncated).is_err());
    }

    #[test]
    fn wrong_mel_count_fails_the_check() {
        let mut m = mini_onnx();
        // the only 40 (0x28) byte in the file is the input shape's mel count
        let pos = m.iter().position(|&b| b == 40).unwrap();
        m[pos] = 64;
        let info = inspect_onnx(&m).unwrap();
        assert_eq!(info.inputs[0].shape, vec![1, 16, 64]);
        assert!(checks(&info).iter().any(|c| !c.ok && !c.warning));
    }
}
