//! Run a streaming microWakeWord `.tflite` on this computer, the way the Voice PE runs it.
//!
//! [`StreamingModel`] is a small int8 interpreter for exactly the operators the firmware
//! registers for wake word models (convolutions, depthwise convolutions, fully connected,
//! logistic, slicing, concatenation and the resource variables that hold the streaming
//! state). Requantization follows the TFLite Micro reference kernels: integer
//! multipliers, rounding shifts and the fixed-point logistic. [`Detector`] adds the
//! firmware's decision rule (`streaming_model.cpp`): a sliding-window average of the
//! uint8 probabilities against `probability_cutoff`, nothing during the first second, and
//! a cool-off after each detection.
//!
//! ```no_run
//! use havoice::micro_model::{Detector, StreamingModel};
//!
//! # fn demo() -> anyhow::Result<()> {
//! let model = StreamingModel::load(&std::fs::read("ahoj_hugo_model/ahoj_hugo.tflite")?)?;
//! let mut detector = Detector::new(model, 0.97, 5);
//! let pcm = havoice::Pcm::from_wav(&std::fs::read("ahoj_hugo.wav")?)?;
//! let result = detector.run(&pcm.samples)?;
//! println!("detected: {}, peak {:.2}", result.detected(), result.max_probability);
//! # Ok(()) }
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::micro_frontend::{CHANNELS, MicroFrontend, to_model_input};
use crate::wake_word::Fb;

// ------------------------------------------------------------------------------- reading

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DType {
    F32,
    I32,
    U8,
    I64,
    I8,
    I16,
    Bool,
    Resource,
    Other(i8),
}

impl DType {
    fn from_code(code: i8) -> Self {
        match code {
            0 => Self::F32,
            2 => Self::I32,
            3 => Self::U8,
            4 => Self::I64,
            6 => Self::Bool,
            7 => Self::I16,
            9 => Self::I8,
            20 => Self::Resource,
            other => Self::Other(other),
        }
    }
}

#[derive(Debug, Clone)]
struct Tensor {
    shape: Vec<usize>,
    dtype: DType,
    scale: Vec<f32>,
    zero_point: Vec<i32>,
    /// every element widened to i32 (int8, uint8, int32 data; resource ids)
    data: Vec<i32>,
}

impl Tensor {
    fn len(&self) -> usize {
        self.shape.iter().product()
    }
    fn scale(&self) -> f32 {
        self.scale.first().copied().unwrap_or(1.0)
    }
    fn zero_point(&self) -> i32 {
        self.zero_point.first().copied().unwrap_or(0)
    }
    fn qrange(&self) -> (i32, i32) {
        match self.dtype {
            DType::U8 => (0, 255),
            DType::I16 => (-32768, 32767),
            _ => (-128, 127),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Activation {
    None,
    Relu,
    ReluN1To1,
    Relu6,
}

#[derive(Debug, Clone, Copy)]
struct ConvParams {
    same_padding: bool,
    stride_w: usize,
    stride_h: usize,
    dilation_w: usize,
    dilation_h: usize,
    activation: Activation,
}

#[derive(Debug, Clone)]
enum Kind {
    Conv(ConvParams),
    Depthwise(ConvParams),
    FullyConnected(Activation),
    Concatenation(i32),
    Reshape,
    StridedSlice {
        begin_mask: i32,
        end_mask: i32,
        shrink_axis_mask: i32,
    },
    SplitV,
    Logistic,
    Quantize,
    CallOnce(usize),
    VarHandle(String),
    ReadVariable,
    AssignVariable,
}

#[derive(Debug, Clone)]
struct Op {
    kind: Kind,
    inputs: Vec<i32>,
    outputs: Vec<i32>,
}

#[derive(Debug, Clone)]
struct Graph {
    tensors: Vec<Tensor>,
    ops: Vec<Op>,
    inputs: Vec<i32>,
    outputs: Vec<i32>,
}

impl Fb<'_> {
    fn opt_u8(&self, table: usize, n: usize, default: u8) -> Result<u8> {
        Ok(match self.field(table, n)? {
            Some(f) => self.i8(f)? as u8,
            None => default,
        })
    }
    fn opt_i32(&self, table: usize, n: usize, default: i32) -> Result<i32> {
        Ok(match self.field(table, n)? {
            Some(f) => self.i32(f)?,
            None => default,
        })
    }
    fn bytes(&self, table: usize, n: usize) -> Result<&[u8]> {
        Ok(match self.vector(table, n)? {
            Some((start, len)) => self
                .0
                .get(start..start + len)
                .ok_or_else(|| anyhow!("truncated file"))?,
            None => &[],
        })
    }
    fn string(&self, table: usize, n: usize) -> Result<String> {
        Ok(String::from_utf8_lossy(self.bytes(table, n)?).into_owned())
    }
    fn floats(&self, table: usize, n: usize) -> Result<Vec<f32>> {
        Ok(self
            .bytes_n(table, n, 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect())
    }
    fn longs(&self, table: usize, n: usize) -> Result<Vec<i64>> {
        Ok(self
            .bytes_n(table, n, 8)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| i64::from_le_bytes(b))
            .collect())
    }
    /// The bytes of a vector of `width`-byte scalars.
    fn bytes_n(&self, table: usize, n: usize, width: usize) -> Result<&[u8]> {
        Ok(match self.vector(table, n)? {
            Some((start, len)) => self
                .0
                .get(start..start + len * width)
                .ok_or_else(|| anyhow!("truncated file"))?,
            None => &[],
        })
    }
}

fn activation(code: u8) -> Result<Activation> {
    Ok(match code {
        0 => Activation::None,
        1 => Activation::Relu,
        2 => Activation::ReluN1To1,
        3 => Activation::Relu6,
        other => bail!("unsupported fused activation {other}"),
    })
}

fn conv_params(fb: &Fb, t: usize, depthwise: bool) -> Result<ConvParams> {
    // Conv2DOptions: 0 padding, 1 stride_w, 2 stride_h, 3 activation, 4 dilation_w, 5 dilation_h
    // DepthwiseConv2DOptions: 0 padding, 1 stride_w, 2 stride_h, 3 depth_multiplier,
    //                         4 activation, 5 dilation_w, 6 dilation_h
    let d = depthwise as usize;
    Ok(ConvParams {
        same_padding: fb.opt_u8(t, 0, 0)? == 0,
        stride_w: fb.opt_i32(t, 1, 1)?.max(1) as usize,
        stride_h: fb.opt_i32(t, 2, 1)?.max(1) as usize,
        activation: activation(fb.opt_u8(t, 3 + d, 0)?)?,
        dilation_w: fb.opt_i32(t, 4 + d, 1)?.max(1) as usize,
        dilation_h: fb.opt_i32(t, 5 + d, 1)?.max(1) as usize,
    })
}

fn read_model(bytes: &[u8]) -> Result<Vec<Graph>> {
    ensure!(
        bytes.len() > 8 && &bytes[4..8] == b"TFL3",
        "not a TensorFlow Lite file (no TFL3 identifier)"
    );
    let fb = Fb(bytes);
    let model = fb.deref(0)?;
    // Model: 1 operator_codes, 2 subgraphs, 4 buffers
    let mut codes = Vec::new();
    for oc in fb.tables(model, 1)? {
        let deprecated = fb.opt_u8(oc, 0, 0)? as i8 as i32;
        codes.push(deprecated.max(fb.opt_i32(oc, 3, 0)?));
    }
    let buffers = fb.tables(model, 4)?;
    let mut graphs = Vec::new();
    for sg in fb.tables(model, 2)? {
        // SubGraph: 0 tensors, 1 inputs, 2 outputs, 3 operators
        let mut tensors = Vec::new();
        for t in fb.tables(sg, 0)? {
            // Tensor: 0 shape, 1 type, 2 buffer, 4 quantization
            let shape: Vec<usize> = fb.ints(t, 0)?.iter().map(|&d| d.max(0) as usize).collect();
            let dtype = DType::from_code(fb.opt_u8(t, 1, 0)? as i8);
            let (mut scale, mut zero_point) = (vec![], vec![]);
            if let Some(q) = fb.field(t, 4)? {
                // QuantizationParameters: 2 scale, 3 zero_point
                let q = fb.deref(q)?;
                scale = fb.floats(q, 2)?;
                zero_point = fb.longs(q, 3)?.iter().map(|&z| z as i32).collect();
            }
            let buffer = fb.opt_i32(t, 2, 0)? as usize;
            let raw = match buffers.get(buffer) {
                Some(&b) if buffer > 0 => fb.bytes(b, 0)?,
                _ => &[],
            };
            let data: Vec<i32> = match dtype {
                DType::I8 => raw.iter().map(|&b| b as i8 as i32).collect(),
                DType::U8 | DType::Bool => raw.iter().map(|&b| b as i32).collect(),
                DType::I16 => raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&b| i16::from_le_bytes(b) as i32)
                    .collect(),
                DType::I32 => raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| i32::from_le_bytes(b))
                    .collect(),
                DType::I64 => raw
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|&b| i64::from_le_bytes(b) as i32)
                    .collect(),
                _ => vec![],
            };
            tensors.push(Tensor {
                shape,
                dtype,
                scale,
                zero_point,
                data,
            });
        }
        let mut ops = Vec::new();
        for o in fb.tables(sg, 3)? {
            // Operator: 0 opcode_index, 1 inputs, 2 outputs, 4 builtin_options
            let code = *codes
                .get(fb.opt_i32(o, 0, 0)? as usize)
                .context("bad opcode index")?;
            let opts = match fb.field(o, 4)? {
                Some(f) => Some(fb.deref(f)?),
                None => None,
            };
            let opt = || opts.context("operator without options");
            let kind = match code {
                3 => Kind::Conv(conv_params(&fb, opt()?, false)?),
                4 => Kind::Depthwise(conv_params(&fb, opt()?, true)?),
                9 => Kind::FullyConnected(match opts {
                    Some(t) => activation(fb.opt_u8(t, 0, 0)?)?,
                    None => Activation::None,
                }),
                2 => Kind::Concatenation(match opts {
                    Some(t) => fb.opt_i32(t, 0, 0)?,
                    None => 0,
                }),
                22 => Kind::Reshape,
                45 => {
                    let t = opt()?;
                    // StridedSliceOptions: 0 begin_mask, 1 end_mask, 2 ellipsis_mask,
                    //                      3 new_axis_mask, 4 shrink_axis_mask
                    ensure!(
                        fb.opt_i32(t, 2, 0)? == 0 && fb.opt_i32(t, 3, 0)? == 0,
                        "STRIDED_SLICE with ellipsis or new axis masks isn't supported"
                    );
                    Kind::StridedSlice {
                        begin_mask: fb.opt_i32(t, 0, 0)?,
                        end_mask: fb.opt_i32(t, 1, 0)?,
                        shrink_axis_mask: fb.opt_i32(t, 4, 0)?,
                    }
                }
                102 => Kind::SplitV,
                14 => Kind::Logistic,
                114 => Kind::Quantize,
                129 => Kind::CallOnce(fb.opt_i32(opt()?, 0, 0)? as usize),
                142 => {
                    // VarHandleOptions: 0 container, 1 shared_name
                    let t = opt()?;
                    Kind::VarHandle(format!("{}/{}", fb.string(t, 0)?, fb.string(t, 1)?))
                }
                143 => Kind::ReadVariable,
                144 => Kind::AssignVariable,
                other => bail!(
                    "operator {other} isn't supported here (the device registers it only if listed in wake_word::SUPPORTED_OPS)"
                ),
            };
            ops.push(Op {
                kind,
                inputs: fb.ints(o, 1)?,
                outputs: fb.ints(o, 2)?,
            });
        }
        graphs.push(Graph {
            tensors,
            ops,
            inputs: fb.ints(sg, 1)?,
            outputs: fb.ints(sg, 2)?,
        });
    }
    ensure!(!graphs.is_empty(), "model has no subgraph");
    Ok(graphs)
}

// ------------------------------------------------------------- integer arithmetic (TFLM)

/// `QuantizeMultiplier`: a real multiplier as a Q31 value and a power-of-two shift.
fn quantize_multiplier(m: f64) -> (i32, i32) {
    if m == 0.0 {
        return (0, 0);
    }
    let (q, mut shift) = frexp(m);
    let mut q_fixed = (q * (1i64 << 31) as f64).round() as i64;
    if q_fixed == 1i64 << 31 {
        q_fixed /= 2;
        shift += 1;
    }
    if shift < -31 {
        shift = 0;
        q_fixed = 0;
    }
    if shift > 30 {
        shift = 30;
        q_fixed = (1i64 << 31) - 1;
    }
    (q_fixed as i32, shift)
}

/// `std::frexp`: `x = q * 2^e` with `0.5 <= |q| < 1`.
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i32;
    if exp == 0 {
        let (q, e) = frexp(x * (1u64 << 54) as f64);
        return (q, e - 54);
    }
    let q = f64::from_bits((bits & !(0x7FFu64 << 52)) | (1022u64 << 52));
    (q, exp - 1022)
}

fn saturating_rounding_doubling_high_mul(a: i32, b: i32) -> i32 {
    if a == b && a == i32::MIN {
        return i32::MAX;
    }
    let ab = a as i64 * b as i64;
    let nudge = if ab >= 0 {
        1i64 << 30
    } else {
        1 - (1i64 << 30)
    };
    ((ab + nudge) / (1i64 << 31)) as i32
}

fn rounding_divide_by_pot(x: i32, exponent: i32) -> i32 {
    if exponent <= 0 {
        return x;
    }
    let mask = (1i64 << exponent) - 1;
    let remainder = x as i64 & mask;
    let threshold = (mask >> 1) + (x < 0) as i64;
    (x >> exponent) + (remainder > threshold) as i32
}

fn multiply_by_quantized_multiplier(x: i32, multiplier: i32, shift: i32) -> i32 {
    let left = shift.max(0);
    let right = (-shift).max(0);
    rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(x.wrapping_mul(1 << left), multiplier),
        right,
    )
}

fn saturating_shift(x: i32, exponent: i32) -> i32 {
    if exponent > 0 {
        (x as i64)
            .checked_shl(exponent as u32)
            .unwrap_or(if x < 0 { i64::MIN } else { i64::MAX })
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32
    } else {
        rounding_divide_by_pot(x, -exponent)
    }
}

/// gemmlowp fixed-point arithmetic, as TFLite Micro's int8 LOGISTIC uses it.
mod gemmlowp {
    use super::{
        rounding_divide_by_pot, saturating_rounding_doubling_high_mul as mul, saturating_shift,
    };

    /// `1 / (1 + x)` for `x` in [0, 1), Q0.31 in, Q0.31 out.
    fn one_over_one_plus_x(a: i32) -> i32 {
        // RoundingHalfSum(a, F0::One())
        let sum = a as i64 + i32::MAX as i64;
        let half_denominator = ((sum + if sum >= 0 { 1 } else { -1 }) / 2) as i32;
        let c48_17: i32 = 1515870810; // Q2.29
        let cneg32_17: i32 = -1010580540;
        let one_q2: i32 = 1 << 29;
        let mut x = c48_17.wrapping_add(mul(half_denominator, cneg32_17));
        for _ in 0..3 {
            let half_denominator_times_x = mul(half_denominator, x); // Q2
            let one_minus = one_q2.wrapping_sub(half_denominator_times_x); // Q2
            x = x.wrapping_add(saturating_shift(mul(x, one_minus), 2)); // Q4 -> Q2
        }
        saturating_shift(x, 1) // ExactMulByPot<-1> (Q2 -> Q1), Rescale<0>
    }

    /// `exp(a)` for `a` in [-1/4, 0), Q0.31.
    fn exp_quarter(a: i32) -> i32 {
        let constant_term: i32 = 1895147668;
        let one_third: i32 = 715827883;
        let x = a.wrapping_add(1 << 28);
        let x2 = mul(x, x);
        let x3 = mul(x2, x);
        let x4 = mul(x2, x2);
        let x4_over_4 = rounding_divide_by_pot(x4, 2);
        let poly = rounding_divide_by_pot(
            mul(x4_over_4.wrapping_add(x3), one_third).wrapping_add(x2),
            1,
        );
        constant_term.wrapping_add(mul(constant_term, x.wrapping_add(poly)))
    }

    /// `exp(a)` for `a <= 0` given with 4 integer bits (Q4.27); result Q0.31.
    fn exp_on_negative_values_q4(a: i32) -> i32 {
        const FRACTIONAL: i32 = 27;
        let one_quarter = 1i32 << (FRACTIONAL - 2);
        let mask = one_quarter - 1;
        let a_mod_quarter_minus_quarter = (a & mask).wrapping_sub(one_quarter);
        let mut result = exp_quarter(saturating_shift(a_mod_quarter_minus_quarter, 4));
        let remainder = a_mod_quarter_minus_quarter.wrapping_sub(a);
        for (exponent, multiplier) in [
            (-2, 1672461947),
            (-1, 1302514674),
            (0, 790015084),
            (1, 290630308),
            (2, 39332535),
            (3, 720401),
        ] {
            if remainder & (1 << (FRACTIONAL + exponent)) != 0 {
                result = mul(result, multiplier);
            }
        }
        if a == 0 { i32::MAX } else { result }
    }

    /// `logistic(a)` for `a` in Q4.27; result Q0.31.
    pub(super) fn logistic_q4(a: i32) -> i32 {
        if a == 0 {
            return 1 << 30;
        }
        let positive = a > 0;
        let abs = if positive { a } else { a.wrapping_neg() };
        let on_positive = one_over_one_plus_x(exp_on_negative_values_q4(abs.wrapping_neg()));
        if positive {
            on_positive
        } else {
            i32::MAX.wrapping_sub(on_positive)
        }
    }
}

fn activation_range(act: Activation, t: &Tensor) -> (i32, i32) {
    let (qmin, qmax) = t.qrange();
    let q = |v: f32| t.zero_point() + (v / t.scale()).round() as i32;
    match act {
        Activation::None => (qmin, qmax),
        Activation::Relu => (qmin.max(q(0.0)), qmax),
        Activation::Relu6 => (qmin.max(q(0.0)), qmax.min(q(6.0))),
        Activation::ReluN1To1 => (qmin.max(q(-1.0)), qmax.min(q(1.0))),
    }
}

/// Per-output-channel (multiplier, shift), from input, filter and output scales.
fn channel_multipliers(
    input: &Tensor,
    filter: &Tensor,
    output: &Tensor,
    channels: usize,
) -> Vec<(i32, i32)> {
    (0..channels)
        .map(|c| {
            let fs = *filter.scale.get(c).or(filter.scale.first()).unwrap_or(&1.0);
            quantize_multiplier(input.scale() as f64 * fs as f64 / output.scale() as f64)
        })
        .collect()
}

fn padding(
    same: bool,
    in_size: usize,
    filter: usize,
    stride: usize,
    dilation: usize,
    out_size: usize,
) -> i64 {
    if !same {
        return 0;
    }
    let effective = (filter - 1) * dilation + 1;
    ((((out_size as i64 - 1) * stride as i64 + effective as i64) - in_size as i64) / 2).max(0)
}

// ------------------------------------------------------------------------------- kernels

fn conv(
    p: ConvParams,
    depthwise: bool,
    input: &Tensor,
    filter: &Tensor,
    bias: Option<&Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    ensure!(
        input.shape.len() == 4 && filter.shape.len() == 4 && output.shape.len() == 4,
        "conv needs 4-D tensors"
    );
    let [_, ih, iw, ic] = input.shape[..] else {
        unreachable!()
    };
    let [_, oh, ow, oc] = output.shape[..] else {
        unreachable!()
    };
    let (fh, fw) = (filter.shape[1], filter.shape[2]);
    let pad_h = padding(p.same_padding, ih, fh, p.stride_h, p.dilation_h, oh);
    let pad_w = padding(p.same_padding, iw, fw, p.stride_w, p.dilation_w, ow);
    let mults = channel_multipliers(input, filter, output, oc);
    let (act_min, act_max) = activation_range(p.activation, output);
    let in_off = -input.zero_point();
    let out_off = output.zero_point();
    let depth_multiplier = if depthwise { oc / ic.max(1) } else { 1 };
    let mut out = vec![0i32; oh * ow * oc];
    for oy in 0..oh {
        for ox in 0..ow {
            for c in 0..oc {
                let mut acc: i64 = 0;
                for fy in 0..fh {
                    let y = (oy * p.stride_h) as i64 - pad_h + (fy * p.dilation_h) as i64;
                    if y < 0 || y >= ih as i64 {
                        continue;
                    }
                    for fx in 0..fw {
                        let x = (ox * p.stride_w) as i64 - pad_w + (fx * p.dilation_w) as i64;
                        if x < 0 || x >= iw as i64 {
                            continue;
                        }
                        let base = (y as usize * iw + x as usize) * ic;
                        if depthwise {
                            let ich = c / depth_multiplier;
                            let w = filter.data[(fy * fw + fx) * oc + c];
                            acc += w as i64 * (input.data[base + ich] + in_off) as i64;
                        } else {
                            let fbase = ((c * fh + fy) * fw + fx) * ic;
                            for i in 0..ic {
                                acc += filter.data[fbase + i] as i64
                                    * (input.data[base + i] + in_off) as i64;
                            }
                        }
                    }
                }
                if let Some(b) = bias {
                    acc += b.data[c] as i64;
                }
                let (m, s) = mults[c];
                let v = multiply_by_quantized_multiplier(acc as i32, m, s) + out_off;
                out[(oy * ow + ox) * oc + c] = v.clamp(act_min, act_max);
            }
        }
    }
    output.data = out;
    Ok(())
}

fn fully_connected(
    act: Activation,
    input: &Tensor,
    filter: &Tensor,
    bias: Option<&Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let depth = *filter.shape.last().context("filter without shape")?;
    let units = filter.shape[0];
    let batches = input.len() / depth;
    let (act_min, act_max) = activation_range(act, output);
    let mults = channel_multipliers(input, filter, output, units);
    let (in_off, f_off, out_off) = (
        -input.zero_point(),
        -filter.zero_point(),
        output.zero_point(),
    );
    let mut out = vec![0i32; batches * units];
    for b in 0..batches {
        for u in 0..units {
            let mut acc: i64 = 0;
            for d in 0..depth {
                acc += (filter.data[u * depth + d] + f_off) as i64
                    * (input.data[b * depth + d] + in_off) as i64;
            }
            if let Some(bias) = bias {
                acc += bias.data[u] as i64;
            }
            let (m, s) = mults[u];
            let v = multiply_by_quantized_multiplier(acc as i32, m, s) + out_off;
            out[b * units + u] = v.clamp(act_min, act_max);
        }
    }
    output.data = out;
    Ok(())
}

fn logistic(input: &Tensor, output: &mut Tensor) -> Result<()> {
    ensure!(
        input.dtype == DType::I8,
        "LOGISTIC is only supported for int8"
    );
    const INPUT_INTEGER_BITS: i32 = 4;
    let real = input.scale() as f64 * (1i64 << (31 - INPUT_INTEGER_BITS)) as f64;
    let (q, left_shift) = frexp(real);
    let multiplier = (q * (1i64 << 31) as f64).round() as i32;
    let radius = (((1i64 << INPUT_INTEGER_BITS) - 1) as f64
        * (1i64 << (31 - INPUT_INTEGER_BITS)) as f64
        / (1i64 << left_shift) as f64)
        .floor() as i32;
    output.data = input
        .data
        .iter()
        .map(|&v| {
            let x = v - input.zero_point();
            if x <= -radius {
                -128
            } else if x >= radius {
                127
            } else {
                let q4 = multiply_by_quantized_multiplier(x, multiplier, left_shift);
                let q0 = gemmlowp::logistic_q4(q4);
                (rounding_divide_by_pot(q0, 23) - 128).clamp(-128, 127)
            }
        })
        .collect();
    Ok(())
}

fn quantize(input: &Tensor, output: &mut Tensor) -> Result<()> {
    ensure!(
        matches!(input.dtype, DType::I8 | DType::U8 | DType::I16),
        "QUANTIZE from {:?} isn't supported",
        input.dtype
    );
    let (m, s) = quantize_multiplier(input.scale() as f64 / output.scale() as f64);
    let (qmin, qmax) = output.qrange();
    output.data = input
        .data
        .iter()
        .map(|&v| {
            (multiply_by_quantized_multiplier(v - input.zero_point(), m, s) + output.zero_point())
                .clamp(qmin, qmax)
        })
        .collect();
    Ok(())
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn concatenation(axis: i32, inputs: &[&Tensor], output: &mut Tensor) -> Result<()> {
    let rank = output.shape.len() as i32;
    let axis = if axis < 0 { axis + rank } else { axis } as usize;
    let outer: usize = output.shape[..axis].iter().product();
    let mut out = Vec::with_capacity(output.len());
    for o in 0..outer {
        for t in inputs {
            ensure!(
                t.scale() == output.scale() && t.zero_point() == output.zero_point(),
                "CONCATENATION with differently quantized inputs isn't supported"
            );
            let inner: usize = t.shape[axis..].iter().product();
            out.extend_from_slice(&t.data[o * inner..(o + 1) * inner]);
        }
    }
    output.data = out;
    Ok(())
}

#[allow(clippy::too_many_arguments)] // the operator's inputs and options, as named in TFLite
fn strided_slice(
    begin_mask: i32,
    end_mask: i32,
    shrink: i32,
    input: &Tensor,
    begin: &Tensor,
    end: &Tensor,
    step: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    let rank = input.shape.len();
    let mut starts = Vec::with_capacity(rank);
    let mut counts = Vec::with_capacity(rank);
    let mut steps = Vec::with_capacity(rank);
    for d in 0..rank {
        let dim = input.shape[d] as i64;
        let s = *step.data.get(d).unwrap_or(&1) as i64;
        ensure!(s != 0, "STRIDED_SLICE with stride 0");
        let fix = |v: i64| if v < 0 { v + dim } else { v };
        let mut b = if begin_mask & (1 << d) != 0 {
            if s > 0 { 0 } else { dim - 1 }
        } else {
            fix(*begin.data.get(d).unwrap_or(&0) as i64)
        };
        let mut e = if end_mask & (1 << d) != 0 {
            if s > 0 { dim } else { -1 }
        } else {
            fix(*end.data.get(d).unwrap_or(&(dim as i32)) as i64)
        };
        if s > 0 {
            b = b.clamp(0, dim);
            e = e.clamp(0, dim);
        } else {
            b = b.clamp(-1, dim - 1);
            e = e.clamp(-1, dim - 1);
        }
        if shrink & (1 << d) != 0 {
            e = b + 1;
        }
        let n = if s > 0 {
            (e - b + s - 1) / s
        } else {
            (b - e - s - 1) / -s
        }
        .max(0);
        starts.push(b);
        counts.push(n as usize);
        steps.push(s);
    }
    let in_strides = strides(&input.shape);
    let total: usize = counts.iter().product();
    ensure!(
        total == output.len(),
        "STRIDED_SLICE produced {total} values for a tensor of {}",
        output.len()
    );
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; rank];
    for _ in 0..total {
        let mut at = 0usize;
        for d in 0..rank {
            at += (starts[d] + idx[d] as i64 * steps[d]) as usize * in_strides[d];
        }
        out.push(input.data[at]);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < counts[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    output.data = out;
    Ok(())
}

fn split_v(input: &Tensor, axis: i32, outputs: &mut [&mut Tensor]) -> Result<()> {
    let rank = input.shape.len() as i32;
    let axis = if axis < 0 { axis + rank } else { axis } as usize;
    let outer: usize = input.shape[..axis].iter().product();
    let inner: usize = input.shape[axis + 1..].iter().product();
    let dim = input.shape[axis];
    let mut offset = 0;
    for out in outputs.iter_mut() {
        let size = out.shape[axis];
        let mut data = Vec::with_capacity(out.len());
        for o in 0..outer {
            let from = (o * dim + offset) * inner;
            data.extend_from_slice(&input.data[from..from + size * inner]);
        }
        out.data = data;
        offset += size;
    }
    ensure!(offset == dim, "SPLIT_V sizes don't add up to the input");
    Ok(())
}

// ----------------------------------------------------------------------- the interpreter

/// A streaming microWakeWord model: int8 features in, one uint8 probability out per call.
#[derive(Debug, Clone)]
pub struct StreamingModel {
    graphs: Vec<Graph>,
    variables: Vec<Option<Tensor>>,
    variable_ids: HashMap<String, usize>,
    called_once: Vec<bool>,
    stride: usize,
}

impl StreamingModel {
    /// Parse a `.tflite`; fails on operators or layouts the device can't run either.
    pub fn load(bytes: &[u8]) -> Result<Self> {
        let graphs = read_model(bytes)?;
        let main = &graphs[0];
        ensure!(
            main.inputs.len() == 1 && main.outputs.len() == 1,
            "model must have one input and one output"
        );
        let input = &main.tensors[main.inputs[0] as usize];
        let output = &main.tensors[main.outputs[0] as usize];
        ensure!(
            input.dtype == DType::I8 && input.shape.len() == 3 && input.shape[2] == CHANNELS,
            "input must be int8 [1, stride, 40], got {:?} {:?}",
            input.dtype,
            input.shape
        );
        ensure!(
            output.dtype == DType::U8 && output.len() == 1,
            "output must be one uint8 value"
        );
        let stride = input.shape[1];
        let n = graphs.len();
        Ok(Self {
            graphs,
            variables: Vec::new(),
            variable_ids: HashMap::new(),
            called_once: vec![false; n],
            stride,
        })
    }

    /// Feature slices consumed per call (the input's time dimension).
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Forget the streaming state, as if the model had just been loaded.
    pub fn reset(&mut self) {
        self.variables.clear();
        self.variable_ids.clear();
        self.called_once.iter_mut().for_each(|c| *c = false);
    }

    /// Run once on `stride` slices of int8 features (`stride * 40` values, oldest first);
    /// returns the probability as the device sees it, 0..=255.
    pub fn invoke(&mut self, features: &[i8]) -> Result<u8> {
        ensure!(
            features.len() == self.stride * CHANNELS,
            "expected {} feature values, got {}",
            self.stride * CHANNELS,
            features.len()
        );
        let main = &mut self.graphs[0];
        let input = main.inputs[0] as usize;
        main.tensors[input].data = features.iter().map(|&v| v as i32).collect();
        self.run(0)?;
        let main = &self.graphs[0];
        let out = &main.tensors[main.outputs[0] as usize];
        Ok(*out.data.first().context("model produced no output")? as u8)
    }

    fn run(&mut self, g: usize) -> Result<()> {
        for i in 0..self.graphs[g].ops.len() {
            let op = self.graphs[g].ops[i].clone();
            self.run_op(g, &op).with_context(|| {
                format!(
                    "subgraph {g}, operator {i} ({:?})",
                    std::mem::discriminant(&op.kind)
                )
            })?;
        }
        Ok(())
    }

    fn run_op(&mut self, g: usize, op: &Op) -> Result<()> {
        let tensor = |model: &Self, idx: i32| -> Option<Tensor> {
            (idx >= 0).then(|| model.graphs[g].tensors[idx as usize].clone())
        };
        let input = |n: usize| -> Result<Tensor> {
            tensor(self, *op.inputs.get(n).context("missing input")?)
                .context("missing input tensor")
        };
        let out_idx = *op.outputs.first().unwrap_or(&-1);
        match &op.kind {
            Kind::CallOnce(sub) => {
                if !self.called_once[*sub] {
                    self.called_once[*sub] = true;
                    self.run(*sub)?;
                }
                return Ok(());
            }
            Kind::VarHandle(name) => {
                let next = self.variable_ids.len();
                let id = *self.variable_ids.entry(name.clone()).or_insert(next);
                if self.variables.len() <= id {
                    self.variables.resize(id + 1, None);
                }
                self.graphs[g].tensors[out_idx as usize].data = vec![id as i32];
                return Ok(());
            }
            Kind::AssignVariable => {
                let id = input(0)?.data[0] as usize;
                let value = input(1)?;
                self.variables[id] = Some(value);
                return Ok(());
            }
            Kind::ReadVariable => {
                let id = input(0)?.data[0] as usize;
                let value = self.variables[id]
                    .clone()
                    .ok_or_else(|| anyhow!("variable read before it was assigned"))?;
                let out = &mut self.graphs[g].tensors[out_idx as usize];
                ensure!(
                    value.data.len() == out.len(),
                    "variable has {} values, tensor wants {}",
                    value.data.len(),
                    out.len()
                );
                out.data = value.data;
                return Ok(());
            }
            Kind::SplitV => {
                let src = input(0)?;
                let axis = input(2)?.data[0];
                let graph = &mut self.graphs[g];
                let mut outs: Vec<Tensor> = op
                    .outputs
                    .iter()
                    .map(|&o| graph.tensors[o as usize].clone())
                    .collect();
                let mut refs: Vec<&mut Tensor> = outs.iter_mut().collect();
                split_v(&src, axis, &mut refs)?;
                for (&o, t) in op.outputs.iter().zip(outs) {
                    graph.tensors[o as usize].data = t.data;
                }
                return Ok(());
            }
            _ => {}
        }
        let mut out = tensor(self, out_idx).context("operator without output")?;
        match &op.kind {
            Kind::Conv(p) => conv(
                *p,
                false,
                &input(0)?,
                &input(1)?,
                tensor(self, *op.inputs.get(2).unwrap_or(&-1)).as_ref(),
                &mut out,
            )?,
            Kind::Depthwise(p) => conv(
                *p,
                true,
                &input(0)?,
                &input(1)?,
                tensor(self, *op.inputs.get(2).unwrap_or(&-1)).as_ref(),
                &mut out,
            )?,
            Kind::FullyConnected(a) => fully_connected(
                *a,
                &input(0)?,
                &input(1)?,
                tensor(self, *op.inputs.get(2).unwrap_or(&-1)).as_ref(),
                &mut out,
            )?,
            Kind::Concatenation(axis) => {
                let ins: Vec<Tensor> = op
                    .inputs
                    .iter()
                    .map(|&i| tensor(self, i).context("missing input"))
                    .collect::<Result<_>>()?;
                let refs: Vec<&Tensor> = ins.iter().collect();
                concatenation(*axis, &refs, &mut out)?
            }
            Kind::Reshape => out.data = input(0)?.data,
            Kind::StridedSlice {
                begin_mask,
                end_mask,
                shrink_axis_mask,
            } => strided_slice(
                *begin_mask,
                *end_mask,
                *shrink_axis_mask,
                &input(0)?,
                &input(1)?,
                &input(2)?,
                &input(3)?,
                &mut out,
            )?,
            Kind::Logistic => logistic(&input(0)?, &mut out)?,
            Kind::Quantize => quantize(&input(0)?, &mut out)?,
            _ => unreachable!(),
        }
        self.graphs[g].tensors[out_idx as usize].data = out.data;
        Ok(())
    }
}

// ------------------------------------------------------------------- the firmware's rule

/// Feature slices (10 ms each) after a reset during which nothing is detected.
pub const MIN_SLICES_BEFORE_DETECTION: i32 = 100;

/// What one audio clip did to a model.
#[derive(Debug, Clone, Default)]
pub struct ClipResult {
    /// seconds into the clip at which each detection fired
    pub detections: Vec<f32>,
    /// highest single probability, 0..=1
    pub max_probability: f32,
    /// highest sliding-window average, 0..=1 (what is compared with the cutoff)
    pub max_average: f32,
    /// model calls made
    pub invocations: usize,
}

impl ClipResult {
    /// Whether the wake word fired at least once.
    pub fn detected(&self) -> bool {
        !self.detections.is_empty()
    }
}

/// A [`StreamingModel`] plus the firmware's frontend and detection rule.
#[derive(Debug, Clone)]
pub struct Detector {
    model: StreamingModel,
    cutoff: u8,
    window: usize,
}

impl Detector {
    /// `probability_cutoff` 0..=1 and `sliding_window_size`, as in the model's manifest.
    pub fn new(model: StreamingModel, probability_cutoff: f32, sliding_window_size: usize) -> Self {
        Self {
            model,
            // __init__.py: int(probability_cutoff * 255)
            cutoff: (probability_cutoff.clamp(0.0, 1.0) * 255.0) as u8,
            window: sliding_window_size.max(1),
        }
    }

    /// Stream 16 kHz mono audio through a fresh frontend and model state, the way the
    /// device processes it right after the model loads.
    pub fn run(&mut self, samples: &[i16]) -> Result<ClipResult> {
        self.model.reset();
        let stride = self.model.stride();
        let mut frontend = MicroFrontend::new();
        let mut result = ClipResult::default();
        let mut ring = vec![0u8; self.window];
        let mut last = 0usize;
        let mut ignore = -MIN_SLICES_BEFORE_DETECTION;
        let mut input = vec![0i8; stride * CHANNELS];
        let mut step = 0usize;
        for (n, slice) in frontend.process(samples).iter().enumerate() {
            input[step * CHANNELS..(step + 1) * CHANNELS].copy_from_slice(&to_model_input(slice));
            step += 1;
            let mut fresh = false;
            if step >= stride {
                step = 0;
                let p = self.model.invoke(&input)?;
                result.invocations += 1;
                result.max_probability = result.max_probability.max(p as f32 / 255.0);
                last = (last + 1) % self.window;
                ring[last] = p;
                fresh = true;
            }
            if ring[last] < self.cutoff {
                ignore = (ignore + 1).min(0);
            }
            // determine_detected(), run after each new probability
            if fresh && ignore >= 0 {
                let sum: u32 = ring.iter().map(|&p| p as u32).sum();
                result.max_average = result
                    .max_average
                    .max(sum as f32 / self.window as f32 / 255.0);
                if sum > self.cutoff as u32 * self.window as u32 {
                    // the slice ends 30 ms after its start, every 10 ms
                    result.detections.push((n * 10 + 30) as f32 / 1000.0);
                    ring.iter_mut().for_each(|p| *p = 0);
                    ignore = -MIN_SLICES_BEFORE_DETECTION;
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipliers_match_tflite() {
        assert_eq!(quantize_multiplier(1.0), (1 << 30, 1));
        assert_eq!(quantize_multiplier(0.5), (1 << 30, 0));
        assert_eq!(frexp(8.0), (0.5, 4));
        assert_eq!(multiply_by_quantized_multiplier(100, 1 << 30, 1), 100);
        assert_eq!(multiply_by_quantized_multiplier(3, 1 << 30, -1), 1); // 0.75 rounds to 1
        // halves round away from zero
        assert_eq!(rounding_divide_by_pot(5, 1), 3);
        assert_eq!(rounding_divide_by_pot(-5, 1), -3);
    }

    #[test]
    fn logistic_is_a_sigmoid() {
        let input = Tensor {
            shape: vec![1, 5],
            dtype: DType::I8,
            scale: vec![0.1],
            zero_point: vec![0],
            data: vec![-128, -20, 0, 20, 127],
        };
        let mut out = Tensor {
            shape: vec![1, 5],
            dtype: DType::I8,
            scale: vec![1.0 / 256.0],
            zero_point: vec![-128],
            data: vec![],
        };
        logistic(&input, &mut out).unwrap();
        for (&x, &y) in input.data.iter().zip(&out.data) {
            let want = 1.0 / (1.0 + (-(x as f32) * 0.1).exp());
            let got = (y + 128) as f32 / 256.0;
            assert!((want - got).abs() < 1.5 / 256.0, "x={x}: {got} vs {want}");
        }
    }
}
