//! In-memory ONNX graph rewrites that work around slow kernels in some execution providers.
//!
//! HTDemucs implements its STFT as a pair of strided 1-D `Conv`s and its iSTFT as a strided
//! `ConvTranspose`, all with dense 4096-point DFT matrices. ONNX Runtime's WebGPU kernels for those
//! shapes are slow (the `ConvTranspose` alone took ~85% of the inference time), so
//! [`optimize_for_webgpu`] replaces them with FFTs built from small MatMuls (~15x fewer FLOPs),
//! after checking that the weights really are a windowed DFT basis.
//!
//! Only the protobuf messages that need to change are decoded; everything else (including the
//! ~300MB of weights) is copied through as raw bytes.

use std::borrow::Cow;

type Result<T> = std::result::Result<T, String>;

// --- Minimal protobuf wire format -------------------------------------------------------------

const WT_VARINT: u8 = 0;
const WT_I64: u8 = 1;
const WT_LEN: u8 = 2;
const WT_I32: u8 = 5;

#[derive(Clone, Copy)]
struct Field<'a> {
    num: u32,
    wire_type: u8,
    /// Varint value (only for `WT_VARINT`).
    varint: u64,
    /// Payload (only for `WT_LEN`).
    payload: &'a [u8],
    /// The complete encoded field (tag included), for copying through unchanged.
    raw: &'a [u8],
}

fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *buf.get(*pos).ok_or("truncated varint")?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("varint too long".into())
}

fn parse(buf: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut fields = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let start = pos;
        let tag = read_varint(buf, &mut pos)?;
        let (num, wire_type) = ((tag >> 3) as u32, (tag & 7) as u8);
        let mut varint = 0;
        let mut payload: &[u8] = &[];
        match wire_type {
            WT_VARINT => varint = read_varint(buf, &mut pos)?,
            WT_I64 => pos += 8,
            WT_I32 => pos += 4,
            WT_LEN => {
                let len = read_varint(buf, &mut pos)? as usize;
                let end = pos.checked_add(len).filter(|end| *end <= buf.len()).ok_or("truncated field")?;
                payload = &buf[pos..end];
                pos = end;
            }
            _ => return Err(format!("unsupported wire type {wire_type}")),
        }
        if pos > buf.len() {
            return Err("truncated field".into());
        }
        fields.push(Field { num, wire_type, varint, payload, raw: &buf[start..pos] });
    }
    Ok(fields)
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn put_tag(out: &mut Vec<u8>, num: u32, wire_type: u8) {
    put_varint(out, (u64::from(num) << 3) | u64::from(wire_type));
}

fn put_bytes(out: &mut Vec<u8>, num: u32, bytes: &[u8]) {
    put_tag(out, num, WT_LEN);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn put_int(out: &mut Vec<u8>, num: u32, value: i64) {
    put_tag(out, num, WT_VARINT);
    put_varint(out, value as u64);
}

/// Repeated int64 fields may be encoded packed or unpacked; accept both.
fn read_ints(fields: &[Field], num: u32) -> Result<Vec<i64>> {
    let mut values = Vec::new();
    for f in fields.iter().filter(|f| f.num == num) {
        match f.wire_type {
            WT_VARINT => values.push(f.varint as i64),
            WT_LEN => {
                let mut pos = 0;
                while pos < f.payload.len() {
                    values.push(read_varint(f.payload, &mut pos)? as i64);
                }
            }
            _ => return Err("unexpected wire type for int field".into()),
        }
    }
    Ok(values)
}

fn read_str<'a>(fields: &[Field<'a>], num: u32) -> &'a str {
    fields
        .iter()
        .rev()
        .find(|f| f.num == num && f.wire_type == WT_LEN)
        .and_then(|f| std::str::from_utf8(f.payload).ok())
        .unwrap_or("")
}

fn read_strs<'a>(fields: &[Field<'a>], num: u32) -> Vec<&'a str> {
    fields
        .iter()
        .filter(|f| f.num == num && f.wire_type == WT_LEN)
        .map(|f| std::str::from_utf8(f.payload).unwrap_or(""))
        .collect()
}

// --- ONNX messages (field numbers from onnx.proto) --------------------------------------------

mod model {
    pub const GRAPH: u32 = 7;
    pub const OPSET_IMPORT: u32 = 8;
}
mod opset {
    pub const DOMAIN: u32 = 1;
    pub const VERSION: u32 = 2;
}
mod graph {
    pub const NODE: u32 = 1;
    pub const INITIALIZER: u32 = 5;
}
mod node {
    pub const INPUT: u32 = 1;
    pub const OUTPUT: u32 = 2;
    pub const NAME: u32 = 3;
    pub const OP_TYPE: u32 = 4;
    pub const ATTRIBUTE: u32 = 5;
    pub const DOMAIN: u32 = 7;
}
mod attr {
    pub const NAME: u32 = 1;
    pub const F: u32 = 2;
    pub const I: u32 = 3;
    pub const S: u32 = 4;
    pub const T: u32 = 5;
    pub const INTS: u32 = 8;
    pub const TYPE: u32 = 20;
    pub const TYPE_INT: i64 = 2;
    pub const TYPE_INTS: i64 = 7;
}
mod tensor {
    pub const DIMS: u32 = 1;
    pub const DATA_TYPE: u32 = 2;
    pub const FLOAT_DATA: u32 = 4;
    pub const INT64_DATA: u32 = 7;
    pub const NAME: u32 = 8;
    pub const RAW_DATA: u32 = 9;
    pub const DATA_LOCATION: u32 = 14;
    pub const FLOAT: i64 = 1;
    pub const INT64: i64 = 7;
}

struct Node<'a> {
    inputs: Vec<&'a str>,
    outputs: Vec<&'a str>,
    op_type: &'a str,
    domain: &'a str,
    attributes: Vec<Vec<Field<'a>>>,
}

impl<'a> Node<'a> {
    fn parse(buf: &'a [u8]) -> Result<Self> {
        let fields = parse(buf)?;
        Ok(Self {
            inputs: read_strs(&fields, node::INPUT),
            outputs: read_strs(&fields, node::OUTPUT),
            op_type: read_str(&fields, node::OP_TYPE),
            domain: read_str(&fields, node::DOMAIN),
            attributes: fields
                .iter()
                .filter(|f| f.num == node::ATTRIBUTE)
                .map(|f| parse(f.payload))
                .collect::<Result<_>>()?,
        })
    }

    fn attr(&self, name: &str) -> Option<&[Field<'a>]> {
        self.attributes.iter().find(|a| read_str(a, attr::NAME) == name).map(Vec::as_slice)
    }

    fn attr_ints(&self, name: &str) -> Result<Option<Vec<i64>>> {
        self.attr(name).map(|a| read_ints(a, attr::INTS)).transpose()
    }

    fn attr_int(&self, name: &str) -> Option<i64> {
        self.attr(name).and_then(|a| a.iter().find(|f| f.num == attr::I)).map(|f| f.varint as i64)
    }

    fn attr_str(&self, name: &str) -> Option<&'a str> {
        self.attr(name).map(|a| read_str(a, attr::S))
    }

    fn attr_float(&self, name: &str) -> Option<f32> {
        let f = self.attr(name)?.iter().find(|f| f.num == attr::F && f.wire_type == WT_I32)?;
        Some(f32::from_le_bytes(f.raw[f.raw.len() - 4..].try_into().unwrap()))
    }
}

/// The values of an inline int64 tensor, or `None` for any other tensor.
fn parse_i64_tensor(buf: &[u8]) -> Result<Option<Vec<i64>>> {
    let fields = parse(buf)?;
    let data_type = fields.iter().find(|f| f.num == tensor::DATA_TYPE).map(|f| f.varint as i64);
    let external = fields.iter().any(|f| f.num == tensor::DATA_LOCATION && f.varint == 1);
    if data_type != Some(tensor::INT64) || external {
        return Ok(None);
    }
    Ok(Some(match fields.iter().find(|f| f.num == tensor::RAW_DATA) {
        Some(raw) => raw.payload.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect(),
        None => read_ints(&fields, tensor::INT64_DATA)?,
    }))
}

/// A float32 tensor whose data lives inline in the model.
struct FloatTensor<'a> {
    dims: Vec<i64>,
    data: Cow<'a, [u8]>,
}

impl<'a> FloatTensor<'a> {
    fn parse(buf: &'a [u8]) -> Result<Option<Self>> {
        let fields = parse(buf)?;
        let data_type = fields.iter().find(|f| f.num == tensor::DATA_TYPE).map(|f| f.varint as i64);
        let external = fields.iter().any(|f| f.num == tensor::DATA_LOCATION && f.varint == 1);
        if data_type != Some(tensor::FLOAT) || external {
            return Ok(None);
        }
        let dims = read_ints(&fields, tensor::DIMS)?;
        let data = if let Some(raw) = fields.iter().find(|f| f.num == tensor::RAW_DATA) {
            Cow::Borrowed(raw.payload)
        } else {
            let mut bytes = Vec::new();
            for f in fields.iter().filter(|f| f.num == tensor::FLOAT_DATA) {
                match f.wire_type {
                    WT_LEN => bytes.extend_from_slice(f.payload),
                    WT_I32 => bytes.extend_from_slice(&f.raw[f.raw.len() - 4..]),
                    _ => return Err("unexpected wire type for float_data".into()),
                }
            }
            Cow::Owned(bytes)
        };
        let expected = dims.iter().product::<i64>() as usize * 4;
        if data.len() != expected {
            return Err(format!("tensor data is {} bytes, expected {expected}", data.len()));
        }
        Ok(Some(Self { dims, data }))
    }
}

fn encode_tensor(name: &str, data_type: i64, dims: &[i64], raw_data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw_data.len() + name.len() + 32);
    for d in dims {
        put_int(&mut out, tensor::DIMS, *d);
    }
    put_int(&mut out, tensor::DATA_TYPE, data_type);
    put_bytes(&mut out, tensor::NAME, name.as_bytes());
    put_bytes(&mut out, tensor::RAW_DATA, raw_data);
    out
}

/// An attribute for [`encode_node`]: a list of ints, or a single int.
enum Attr<'a> {
    Ints(&'a [i64]),
    Int(i64),
}

fn encode_node(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, Attr)]) -> Vec<u8> {
    let mut out = Vec::new();
    for i in inputs {
        put_bytes(&mut out, node::INPUT, i.as_bytes());
    }
    for o in outputs {
        put_bytes(&mut out, node::OUTPUT, o.as_bytes());
    }
    put_bytes(&mut out, node::NAME, name.as_bytes());
    put_bytes(&mut out, node::OP_TYPE, op_type.as_bytes());
    for (attr_name, value) in attrs {
        let mut a = Vec::new();
        put_bytes(&mut a, attr::NAME, attr_name.as_bytes());
        match value {
            Attr::Ints(values) => {
                for v in *values {
                    put_int(&mut a, attr::INTS, *v);
                }
                put_int(&mut a, attr::TYPE, attr::TYPE_INTS);
            }
            Attr::Int(v) => {
                put_int(&mut a, attr::I, *v);
                put_int(&mut a, attr::TYPE, attr::TYPE_INT);
            }
        }
        put_bytes(&mut out, node::ATTRIBUTE, &a);
    }
    out
}


// --- Graph building -----------------------------------------------------------------------------

/// Accumulates the nodes and initializers of one replacement subgraph, with unique names.
struct Emitter<'a> {
    prefix: String,
    nodes: Vec<Vec<u8>>,
    initializers: &'a mut Vec<Vec<u8>>,
    counter: usize,
    opset: u64,
}

impl<'a> Emitter<'a> {
    fn new(prefix: String, initializers: &'a mut Vec<Vec<u8>>, opset: u64) -> Self {
        Self { prefix, nodes: Vec::new(), initializers, counter: 0, opset }
    }

    fn name(&mut self, what: &str) -> String {
        self.counter += 1;
        format!("{}/{what}_{}", self.prefix, self.counter)
    }

    fn i64s(&mut self, values: &[i64]) -> String {
        let name = self.name("const");
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.initializers.push(encode_tensor(&name, tensor::INT64, &[values.len() as i64], &bytes));
        name
    }

    fn f32s(&mut self, what: &str, dims: &[i64], values: &[f32]) -> String {
        let name = self.name(what);
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.initializers.push(encode_tensor(&name, tensor::FLOAT, dims, &bytes));
        name
    }

    fn raw_f32(&mut self, what: &str, dims: &[i64], bytes: &[u8]) -> String {
        let name = self.name(what);
        self.initializers.push(encode_tensor(&name, tensor::FLOAT, dims, bytes));
        name
    }

    /// Adds a node; `output` names its result, or `None` for a fresh internal name.
    fn op(&mut self, op_type: &str, inputs: &[&str], output: Option<&str>, attrs: &[(&str, Attr)]) -> String {
        let out = output.map(str::to_owned).unwrap_or_else(|| self.name(op_type));
        let name = self.name(op_type);
        self.nodes.push(encode_node(op_type, &name, inputs, &[&out], attrs));
        out
    }

    fn reshape(&mut self, x: &str, shape: &[i64]) -> String {
        let shape = self.i64s(shape);
        self.op("Reshape", &[x, &shape], None, &[])
    }

    fn transpose(&mut self, x: &str, perm: &[i64]) -> String {
        self.op("Transpose", &[x], None, &[("perm", Attr::Ints(perm))])
    }

    fn slice(&mut self, x: &str, axis: i64, start: i64, end: i64) -> String {
        let (starts, ends, axes) = (self.i64s(&[start]), self.i64s(&[end]), self.i64s(&[axis]));
        self.op("Slice", &[x, &starts, &ends, &axes], None, &[])
    }

    fn pad(&mut self, x: &str, pads: &[i64]) -> String {
        let pads = self.i64s(pads);
        self.op("Pad", &[x, &pads], None, &[])
    }

    /// `ReduceMean` over one axis, keeping it.
    fn reduce_mean(&mut self, x: &str, axis: i64) -> String {
        self.reduce_mean_to(x, axis, None)
    }

    fn reduce_mean_to(&mut self, x: &str, axis: i64, output: Option<&str>) -> String {
        // Opset 18 moved `axes` from an attribute to an input.
        if self.opset >= 18 {
            let axes = self.i64s(&[axis]);
            self.op("ReduceMean", &[x, &axes], output, &[])
        } else {
            self.op("ReduceMean", &[x], output, &[("axes", Attr::Ints(&[axis]))])
        }
    }

    /// Overlap-adds `segments` (each `[N, L, S]`, segment `j` delayed by `j*S` samples) into
    /// `output` (`[N, 1, (L + R - 1) * S]`).
    fn overlap_add(&mut self, segments: &[String], stride: i64, output: &str) {
        let r = segments.len() as i64;
        let mut acc: Option<String> = None;
        for (j, seg) in segments.iter().enumerate() {
            let j = j as i64;
            let flat = self.reshape(seg, &[0, 1, -1]); // [N, 1, L*S]
            let padded = self.pad(&flat, &[0, 0, j * stride, 0, 0, (r - 1 - j) * stride]);
            acc = Some(match acc {
                None => padded,
                Some(prev) => self.op("Add", &[&prev, &padded], (j == r - 1).then_some(output), &[]),
            });
        }
        if r == 1 {
            self.op("Identity", &[acc.as_deref().unwrap()], Some(output), &[]);
        }
    }
}

// --- Recognizing (windowed) DFT weights --------------------------------------------------------

/// Max deviation of the weights from the fitted DFT basis, relative to the largest weight.
/// HTDemucs' exported STFT weights are off by up to ~8e-4 (float32 rounding when they were
/// computed); anything that isn't a DFT basis is off by O(1).
const DFT_FIT_TOLERANCE: f64 = 5e-3;

/// `cos_rows[k][n] = cos_scale[k] * window[n] * cos(2πkn/K)` and
/// `sin_rows[k][n] = sin_scale[k] * window[n] * sin(2πkn/K)`, for `k` in `0..=K/2`.
struct DftFit {
    window: Vec<f64>,
    cos_scale: Vec<f64>,
    sin_scale: Vec<f64>,
}

fn trig_tables(k: usize) -> (Vec<f64>, Vec<f64>) {
    (0..k).map(|m| (std::f64::consts::TAU * m as f64 / k as f64).sin_cos()).map(|(s, c)| (c, s)).unzip()
}

fn fit_windowed_dft(cos_rows: &[f32], sin_rows: &[f32], k: usize) -> Option<DftFit> {
    let bins = k / 2 + 1;
    if cos_rows.len() != bins * k || sin_rows.len() != bins * k {
        return None;
    }
    let window: Vec<f64> = cos_rows[..k].iter().map(|v| f64::from(*v)).collect();
    let peak = cos_rows.iter().chain(sin_rows).fold(0f64, |m, v| m.max(f64::from(v.abs())));
    if peak == 0.0 {
        return None;
    }
    let (cos_t, sin_t) = trig_tables(k);
    let fit = |rows: &[f32], table: &[f64]| -> Option<Vec<f64>> {
        let mut scales = Vec::with_capacity(bins);
        let mut basis = vec![0f64; k];
        for (r, row) in rows.chunks_exact(k).enumerate() {
            // basis[n] = window[n] * table[r*n mod K], stepping the angle index by r.
            let mut m = 0;
            for (n, b) in basis.iter_mut().enumerate() {
                *b = window[n] * table[m];
                m += r;
                if m >= k {
                    m -= k;
                }
            }
            let (mut num, mut den) = (0.0, 0.0);
            for (w, b) in row.iter().zip(&basis) {
                num += f64::from(*w) * b;
                den += b * b;
            }
            // sin rows 0 and K/2 are identically zero.
            let scale = if den > 1e-12 * peak * peak { num / den } else { 0.0 };
            if row.iter().zip(&basis).any(|(w, b)| (f64::from(*w) - scale * b).abs() > DFT_FIT_TOLERANCE * peak) {
                return None;
            }
            scales.push(scale);
        }
        Some(scales)
    };
    Some(DftFit { cos_scale: fit(cos_rows, &cos_t)?, sin_scale: fit(sin_rows, &sin_t)?, window })
}

/// Splits a K-point DFT into P x Q (Cooley-Tukey) with P = Q = sqrt(K), both even.
fn fft_split(k: usize) -> Option<(usize, usize)> {
    let p = (k as f64).sqrt().round() as usize;
    (k >= 64 && p * p == k && p % 2 == 0).then_some((p, p))
}

fn even_ceil(v: usize) -> usize {
    v + v % 2
}

// --- STFT: a pair of strided Conv nodes -> FFT -------------------------------------------------

/// Emits the real/imaginary STFT (`Conv(x[N, 1, T], W[K/2+1, 1, K], stride S)` with windowed
/// cos/sin bases) as framing + a P x Q FFT made of two MatMuls.
///
/// With `n = Q*n1 + n2` and `k = k1 + P*k2`:
/// `X[k] = Σ_n2 e^(-2πi k2 n2/Q) · [e^(-2πi k1 n2/K) Σ_n1 w[n] x[n] e^(-2πi k1 n1/P)]`.
/// The window and the twiddle factors are folded into the first MatMul's (per-n2) weights.
fn emit_stft_fft(e: &mut Emitter, input: &str, cos_out: &str, sin_out: &str, k: usize, s: usize, fit: &DftFit) {
    let (p, q) = fft_split(k).unwrap();
    let (bins, r) = (k / 2 + 1, k / s);
    let (pi, qi) = (p as i64, q as i64);
    let (cos_t, sin_t) = trig_tables(k);

    // Frames [N, L, K]: frame l is blocks l..l+R of the signal cut into hop-sized blocks.
    let blocks = e.reshape(input, &[0, -1, s as i64]); // [N, L+R-1, S]
    let segments: Vec<String> = (0..r as i64)
        .map(|j| e.slice(&blocks, 1, j, if j == r as i64 - 1 { i64::MAX } else { j - (r as i64 - 1) }))
        .collect();
    let frames = if r == 1 {
        segments[0].clone()
    } else {
        let refs: Vec<&str> = segments.iter().map(String::as_str).collect();
        e.op("Concat", &refs, None, &[("axis", Attr::Int(2))])
    };

    // Stage 1, per n2: contract n1 with window * twiddle * e^(-2πi k1 n1/P) -> [Q, N, L, (re|im) k1].
    let mut w1 = vec![0f32; q * p * 2 * p];
    for n2 in 0..q {
        for n1 in 0..p {
            let window = fit.window[q * n1 + n2];
            for k1 in 0..p {
                let m = (q * k1 * n1 + k1 * n2) % k;
                let base = (n2 * p + n1) * 2 * p;
                w1[base + k1] = (window * cos_t[m]) as f32;
                w1[base + p + k1] = (-window * sin_t[m]) as f32;
            }
        }
    }
    let w1 = e.f32s("stft_stage1", &[qi, 1, pi, 2 * pi], &w1);
    let f4 = e.reshape(&frames, &[0, 0, pi, qi]); // [N, L, P(n1), Q(n2)]
    let ft = e.transpose(&f4, &[3, 0, 1, 2]); // [Q(n2), N, L, P(n1)]
    let b = e.op("MatMul", &[&ft, &w1], None, &[]); // [Q(n2), N, L, 2P]
    let b5 = e.reshape(&b, &[0, 0, 0, 2, pi]); // [Q(n2), N, L, 2, P(k1)]
    let bt = e.transpose(&b5, &[1, 2, 4, 3, 0]); // [N, L, P(k1), 2, Q(n2)]
    let b4 = e.reshape(&bt, &[0, 0, 0, -1]); // [N, L, P(k1), (re|im) n2]

    // Stage 2: complex contraction over n2 with e^(-2πi k2 n2/Q), only for the k2 that reach
    // k <= K/2 (padded to an even count so the MatMul stays vec4-aligned).
    let k2c = even_ceil(q / 2 + 1);
    let mut m3 = vec![0f32; 2 * q * 2 * k2c];
    for n2 in 0..q {
        for k2 in 0..k2c {
            let m = (p * k2 * n2) % k;
            let (gr, gi) = (cos_t[m] as f32, -sin_t[m] as f32);
            m3[n2 * 2 * k2c + k2] = gr;
            m3[n2 * 2 * k2c + k2c + k2] = gi;
            m3[(q + n2) * 2 * k2c + k2] = -gi;
            m3[(q + n2) * 2 * k2c + k2c + k2] = gr;
        }
    }
    let m3 = e.f32s("stft_stage2", &[2 * qi, 2 * k2c as i64], &m3);
    let c = e.op("MatMul", &[&b4, &m3], None, &[]); // [N, L, P(k1), (re|im) k2]
    let c5 = e.reshape(&c, &[0, 0, 0, 2, k2c as i64]); // [N, L, P(k1), 2, K2c(k2)]
    let ct = e.transpose(&c5, &[3, 0, 4, 2, 1]); // [2, N, K2c(k2), P(k1), L]
    let spectrum = e.reshape(&ct, &[0, 0, (k2c * p) as i64, -1]); // [2, N, k = k1 + P*k2, L]
    let spectrum = e.slice(&spectrum, 2, 0, bins as i64); // [2, N, K/2+1, L]

    // Re X = cos_scale * Re F, and the sin conv computes sin_scale * Σ w x sin = -sin_scale * Im F.
    let axis0 = e.i64s(&[0]);
    for (part, out, scale) in [(0, cos_out, &fit.cos_scale), (1, sin_out, &fit.sin_scale)] {
        let sign = if part == 0 { 1.0 } else { -1.0 };
        let scale: Vec<f32> = scale.iter().map(|v| (sign * v) as f32).collect();
        let scale = e.f32s("stft_scale", &[bins as i64, 1], &scale);
        let half = e.slice(&spectrum, 0, part, part + 1);
        let half = e.op("Squeeze", &[&half, &axis0], None, &[]); // [N, K/2+1, L]
        e.op("Mul", &[&half, &scale], Some(out), &[]);
    }
}

// --- iSTFT: a strided ConvTranspose -> inverse FFT + overlap-add -------------------------------

/// Emits the inverse STFT (`ConvTranspose(X[N, 2*(K/2+1), L], W, stride S)` where W holds the
/// windowed cos rows then the sin rows) as a P x Q inverse FFT made of two MatMuls, followed by
/// overlap-add.
///
/// `frame[n] = w[n] · Re Σ_k Z[k] e^(2πi kn/K)` with `Z[k] = cos_scale[k]·Xr[k] - i·sin_scale[k]·Xi[k]`;
/// with `k = Q*k1 + k2` and `n = n1 + P*n2` the sum factors into a contraction over k1 and a
/// (per-n1, twiddle- and window-folded) contraction over k2.
fn emit_istft_fft(e: &mut Emitter, input: &str, output: &str, k: usize, s: usize, fit: &DftFit) {
    let (p, q) = fft_split(k).unwrap();
    let (bins, r) = (k / 2 + 1, k / s);
    let (pi, qi) = (p as i64, q as i64);
    let (cos_t, sin_t) = trig_tables(k);
    let k1c = even_ceil(p / 2 + 1); // k1 values needed for k <= K/2, kept even for vec4

    let scale: Vec<f32> = fit.cos_scale.iter().map(|v| *v as f32).chain(fit.sin_scale.iter().map(|v| -*v as f32)).collect();
    let scale = e.f32s("istft_scale", &[2 * bins as i64, 1], &scale);
    let z = e.op("Mul", &[input, &scale], None, &[]); // [N, 2*bins, L]
    let z = e.reshape(&z, &[0, 2, bins as i64, -1]); // [N, 2, bins, L]
    let z = e.pad(&z, &[0, 0, 0, 0, 0, 0, (k1c * q - bins) as i64, 0]); // [N, 2, K1c*Q, L]
    let z = e.reshape(&z, &[0, 0, k1c as i64, qi, -1]); // [N, 2, K1c(k1), Q(k2), L]
    let z = e.transpose(&z, &[0, 4, 3, 1, 2]); // [N, L, Q(k2), 2, K1c(k1)]
    let z = e.reshape(&z, &[0, 0, 0, -1]); // [N, L, Q(k2), (re|im) k1]

    // Stage 1: complex contraction over k1 with e^(2πi k1 n1/P).
    let mut ma = vec![0f32; 2 * k1c * 2 * p];
    for k1 in 0..k1c {
        for n1 in 0..p {
            let m = (q * k1 * n1) % k;
            let (hr, hi) = (cos_t[m] as f32, sin_t[m] as f32);
            ma[k1 * 2 * p + n1] = hr;
            ma[k1 * 2 * p + p + n1] = hi;
            ma[(k1c + k1) * 2 * p + n1] = -hi;
            ma[(k1c + k1) * 2 * p + p + n1] = hr;
        }
    }
    let ma = e.f32s("istft_stage1", &[2 * k1c as i64, 2 * pi], &ma);
    let a = e.op("MatMul", &[&z, &ma], None, &[]); // [N, L, Q(k2), (re|im) n1]
    let a = e.reshape(&a, &[0, 0, 0, 2, pi]); // [N, L, Q(k2), 2, P(n1)]
    let a = e.transpose(&a, &[4, 0, 1, 3, 2]); // [P(n1), N, L, 2, Q(k2)]
    let a = e.reshape(&a, &[0, 0, 0, -1]); // [P(n1), N, L, (re|im) k2]

    // Stage 2, per n1: real part of the contraction over k2 with window * twiddle * e^(2πi k2 n2/Q).
    let mut wc = vec![0f32; p * 2 * q * q];
    for n1 in 0..p {
        for k2 in 0..q {
            for n2 in 0..q {
                let m = (k2 * n1 + p * k2 * n2) % k;
                let window = fit.window[n1 + p * n2];
                wc[(n1 * 2 * q + k2) * q + n2] = (window * cos_t[m]) as f32;
                wc[(n1 * 2 * q + q + k2) * q + n2] = (-window * sin_t[m]) as f32;
            }
        }
    }
    let wc = e.f32s("istft_stage2", &[pi, 1, 2 * qi, qi], &wc);
    let y = e.op("MatMul", &[&a, &wc], None, &[]); // [P(n1), N, L, Q(n2)]
    let y = e.transpose(&y, &[1, 2, 3, 0]); // [N, L, Q(n2), P(n1)]
    let frames = e.reshape(&y, &[0, 0, -1]); // [N, L, K], n = n1 + P*n2

    let segments: Vec<String> = (0..r as i64).map(|j| e.slice(&frames, 2, j * s as i64, (j + 1) * s as i64)).collect();
    e.overlap_add(&segments, s as i64, output);
}

// --- iSTFT fallback: ConvTranspose -> MatMul + overlap-add -------------------------------------

/// Emits `ConvTranspose(X[N, Cin, L], W[Cin, 1, K], stride S)` as
/// `overlap_add(Transpose(X) @ W[:, 0, :], hop = S)`, for weights that aren't a DFT basis.
fn emit_matmul_overlap_add(e: &mut Emitter, input: &str, output: &str, weight: &FloatTensor, k: i64, s: i64) {
    let cin = weight.dims[0];
    let r = k / s;

    // Zero-pad Cin to a multiple of 4 so WebGPU picks its vectorized (vec4) MatMul shader.
    let cpad = (4 - cin % 4) % 4;
    let x = if cpad != 0 { e.pad(input, &[0, 0, 0, 0, cpad, 0]) } else { input.to_owned() };
    let xt = e.transpose(&x, &[0, 2, 1]); // [N, L, Cin]

    // One MatMul per hop-sized slice of the kernel: slice j of every frame is exactly the part
    // that lands j*S samples later in the overlap-add. Splitting also keeps each weight small:
    // a single [Cin, K] weight (~67MB for HTDemucs) is sometimes placed where WebGPU reads it
    // ~15x slower, especially on GPUs without resizable BAR.
    let w = &weight.data; // [Cin, 1, K] == [Cin, K] in row-major order
    let row = (k * 4) as usize;
    let segments: Vec<String> = (0..r)
        .map(|j| {
            let mut wj = Vec::with_capacity(((cin + cpad) * s * 4) as usize);
            for c in 0..cin as usize {
                let start = c * row + (j * s * 4) as usize;
                wj.extend_from_slice(&w[start..start + (s * 4) as usize]);
            }
            wj.resize(((cin + cpad) * s * 4) as usize, 0);
            let wj = e.raw_f32("weight", &[cin + cpad, s], &wj);
            e.op("MatMul", &[&xt, &wj], None, &[]) // [N, L, S]
        })
        .collect();
    e.overlap_add(&segments, s, output);
}

// --- Single-group GroupNorm: InstanceNormalization over [N, 1, C*rest] -> two-stage reductions ----

/// Emits `InstanceNormalization(Reshape(x[N, C, ...], [0, 1, -1]))`, i.e. normalizing each sample
/// over all of its channels at once.
///
/// WebGPU computes each instance's statistics with a single workgroup, which takes milliseconds for
/// HTDemucs' multi-million-element instances. Reducing over the channels first and then over the
/// rest gives the same mean (every channel has the same size) with one workgroup per position.
/// Channels first matches the EP's channels-last (NHWC) activations, which avoids a transpose
/// before the first reduction (~2.5x less reduction time than rest-first on an RX 9070).
fn emit_group_norm(e: &mut Emitter, x: &str, scale: &str, bias: &str, output: &str, epsilon: f32, identity_affine: bool) {
    let x = e.reshape(x, &[0, 0, -1]); // [N, C, rest]
    let mean = e.reduce_mean(&x, 1);
    let mean = e.reduce_mean(&mean, 2); // [N, 1, 1]
    let centered = e.op("Sub", &[&x, &mean], None, &[]);
    let squared = e.op("Mul", &[&centered, &centered], None, &[]);
    let variance = e.reduce_mean(&squared, 1);
    let variance = e.reduce_mean(&variance, 2);
    let epsilon = e.f32s("epsilon", &[1], &[epsilon]);
    let variance = e.op("Add", &[&variance, &epsilon], None, &[]);
    let std = e.op("Sqrt", &[&variance], None, &[]);
    let normalized = e.op("Div", &[&centered, &std], None, &[]);
    let shape = e.i64s(&[0, 1, -1]);
    if identity_affine {
        e.op("Reshape", &[&normalized, &shape], Some(output), &[]);
    } else {
        let normalized = e.op("Reshape", &[&normalized, &shape], None, &[]);
        let scaled = e.op("Mul", &[&normalized, scale], None, &[]);
        e.op("Add", &[&scaled, bias], Some(output), &[]);
    }
}

// --- 1-D Conv -> shifted MatMuls ----------------------------------------------------------------

fn round_up4(v: usize) -> usize {
    v.div_ceil(4) * 4
}

/// A 1-D, stride-1, ungrouped `Conv` with constant weight (and bias).
struct Conv1d<'a> {
    input: &'a str,
    output: &'a str,
    /// `[Cout, Cin, K]`
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    cout: usize,
    cin: usize,
    kernel: usize,
    dilation: usize,
    pads: [usize; 2],
}

/// Emits `Conv1d(x[N, Cin, L])` as MatMuls on channels-last data, with channels zero-padded to
/// multiples of 4 for WebGPU's vec4 MatMul.
///
/// WebGPU's Conv shader is very slow for HTDemucs' narrow DConv layers (e.g. 48 -> 6 -> 96
/// channels: ~50-130 GFLOP/s, vs. 2-3 TFLOP/s for its MatMul). Same products, summed in a
/// different order. Kernel taps are applied by shifting whichever side is narrower:
/// - narrow output: one MatMul with all taps' weights side by side (`[Cin, K*Cout]`), then the
///   tap outputs are shifted and summed;
/// - otherwise: the (padded) input is shifted per tap and each shift multiplied by its tap.
fn emit_conv1d_matmul(e: &mut Emitter, conv: &Conv1d) {
    let (cin_p, cout_p, k) = (round_up4(conv.cin), round_up4(conv.cout), conv.kernel);
    let [p0, p1] = conv.pads.map(|p| p as i64);
    let span = (conv.dilation * (k - 1)) as i64;
    let tap_weight = |j: usize, w: &mut [f32], row: usize, col0: usize| {
        for co in 0..conv.cout {
            for ci in 0..conv.cin {
                w[ci * row + col0 + co] = conv.weight[(co * conv.cin + ci) * k + j];
            }
        }
    };
    // Rows [offset, offset + Lout) of axis 1 (from the end, as the length isn't known statically).
    let shifted_end = |offset: i64| if offset == span { i64::MAX } else { offset - span };

    let mut x = conv.input.to_owned();
    let output_side = k > 1 && k * cout_p <= cin_p;
    let (spatial_pad0, spatial_pad1) = if output_side { (0, 0) } else { (p0, p1) };
    if spatial_pad0 != 0 || spatial_pad1 != 0 || cin_p != conv.cin {
        x = e.pad(&x, &[0, 0, spatial_pad0, 0, (cin_p - conv.cin) as i64, spatial_pad1]);
    }
    let xt = e.transpose(&x, &[0, 2, 1]); // [N, L(+pads), Cin_p]

    let mut y = if output_side {
        let mut w = vec![0f32; cin_p * k * cout_p];
        for j in 0..k {
            tap_weight(j, &mut w, k * cout_p, j * cout_p);
        }
        let w = e.f32s("conv_weight", &[cin_p as i64, (k * cout_p) as i64], &w);
        let taps = e.op("MatMul", &[&xt, &w], None, &[]); // [N, L, K*Cout_p]
        let taps = if p0 != 0 || p1 != 0 { e.pad(&taps, &[0, p0, 0, 0, p1, 0]) } else { taps };
        let mut acc: Option<String> = None;
        for j in 0..k {
            let offset = (j * conv.dilation) as i64;
            let (starts, ends, axes) = (
                e.i64s(&[offset, (j * cout_p) as i64]),
                e.i64s(&[shifted_end(offset), ((j + 1) * cout_p) as i64]),
                e.i64s(&[1, 2]),
            );
            let tap = e.op("Slice", &[&taps, &starts, &ends, &axes], None, &[]); // [N, Lout, Cout_p]
            acc = Some(match acc {
                None => tap,
                Some(prev) => e.op("Add", &[&prev, &tap], None, &[]),
            });
        }
        acc.unwrap()
    } else {
        let mut acc: Option<String> = None;
        for j in 0..k {
            let mut w = vec![0f32; cin_p * cout_p];
            tap_weight(j, &mut w, cout_p, 0);
            let w = e.f32s("conv_weight", &[cin_p as i64, cout_p as i64], &w);
            let offset = (j * conv.dilation) as i64;
            let shifted = if k == 1 { xt.clone() } else { e.slice(&xt, 1, offset, shifted_end(offset)) };
            let product = e.op("MatMul", &[&shifted, &w], None, &[]); // [N, Lout, Cout_p]
            acc = Some(match acc {
                None => product,
                Some(prev) => e.op("Add", &[&prev, &product], None, &[]),
            });
        }
        acc.unwrap()
    };
    if let Some(bias) = &conv.bias {
        let mut b = bias.clone();
        b.resize(cout_p, 0.0);
        let b = e.f32s("conv_bias", &[cout_p as i64], &b);
        y = e.op("Add", &[&y, &b], None, &[]);
    }
    if cout_p == conv.cout {
        e.op("Transpose", &[&y], Some(conv.output), &[("perm", Attr::Ints(&[0, 2, 1]))]);
    } else {
        let y = e.transpose(&y, &[0, 2, 1]); // [N, Cout_p, Lout]
        let (starts, ends, axes) = (e.i64s(&[0]), e.i64s(&[conv.cout as i64]), e.i64s(&[1]));
        e.op("Slice", &[&y, &starts, &ends, &axes], Some(conv.output), &[]);
    }
}

// --- Matching ----------------------------------------------------------------------------------

/// A 1-D, ungrouped, unpadded strided (transposed) convolution with a constant weight.
struct StridedConv<'a> {
    input: &'a str,
    output: &'a str,
    weight: FloatTensor<'a>,
    kernel: i64,
    stride: i64,
}

fn match_strided_conv<'a>(
    node: &Node<'a>,
    op_type: &str,
    weights: &mut dyn FnMut(&str) -> Result<Option<FloatTensor<'a>>>,
) -> Result<Option<StridedConv<'a>>> {
    if node.op_type != op_type || !node.domain.is_empty() || node.inputs.len() != 2 || node.outputs.len() != 1 {
        return Ok(None);
    }
    let all_eq = |v: Option<Vec<i64>>, x: i64| v.is_none_or(|v| v.iter().all(|e| *e == x));
    let (Some(kernel), Some(stride)) = (node.attr_ints("kernel_shape")?, node.attr_ints("strides")?) else {
        return Ok(None);
    };
    let ([kernel], [stride]) = (kernel.as_slice(), stride.as_slice()) else {
        return Ok(None);
    };
    let (kernel, stride) = (*kernel, *stride);
    if node.attr_int("group").unwrap_or(1) != 1
        || !all_eq(node.attr_ints("pads")?, 0)
        || !all_eq(node.attr_ints("dilations")?, 1)
        || !all_eq(node.attr_ints("output_padding")?, 0)
        || node.attr("output_shape").is_some()
        || node.attr_str("auto_pad").is_some_and(|p| p != "NOTSET")
        // Only worth it for large strides (the STFT case), where frames barely overlap.
        || stride < 64
        || kernel % stride != 0
    {
        return Ok(None);
    }
    let Some(weight) = weights(node.inputs[1])? else {
        return Ok(None);
    };
    if weight.dims.len() != 3 || weight.dims[1] != 1 || weight.dims[2] != kernel {
        return Ok(None);
    }
    Ok(Some(StridedConv { input: node.inputs[0], output: node.outputs[0], weight, kernel, stride }))
}

fn match_conv1d<'a>(node: &Node<'a>, weights: &mut dyn FnMut(&str) -> Result<Option<FloatTensor<'a>>>) -> Result<Option<Conv1d<'a>>> {
    if node.op_type != "Conv" || !node.domain.is_empty() || !(2..=3).contains(&node.inputs.len()) || node.outputs.len() != 1 {
        return Ok(None);
    }
    let all_eq = |v: Option<Vec<i64>>, x: i64| v.is_none_or(|v| v.iter().all(|e| *e == x));
    let Some(kernel) = node.attr_ints("kernel_shape")? else { return Ok(None) };
    let [kernel] = kernel.as_slice() else { return Ok(None) };
    let dilation = match node.attr_ints("dilations")?.as_deref() {
        None => 1,
        Some([d]) if *d >= 1 => *d,
        _ => return Ok(None),
    };
    let pads = match node.attr_ints("pads")?.as_deref() {
        None => [0, 0],
        Some([a, b]) if *a >= 0 && *b >= 0 => [*a as usize, *b as usize],
        _ => return Ok(None),
    };
    if node.attr_int("group").unwrap_or(1) != 1 || !all_eq(node.attr_ints("strides")?, 1) || node.attr_str("auto_pad").is_some_and(|p| p != "NOTSET") {
        return Ok(None);
    }
    let Some(weight) = weights(node.inputs[1])? else { return Ok(None) };
    let [cout, cin, k] = weight.dims.as_slice() else { return Ok(None) };
    if *k != *kernel {
        return Ok(None);
    }
    let bias = match node.inputs.get(2).filter(|b| !b.is_empty()) {
        None => None,
        Some(b) => match weights(b)? {
            Some(t) if t.dims == [*cout] => Some(t.data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()),
            _ => return Ok(None),
        },
    };
    Ok(Some(Conv1d {
        input: node.inputs[0],
        output: node.outputs[0],
        weight: weight.data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
        bias,
        cout: *cout as usize,
        cin: *cin as usize,
        kernel: *k as usize,
        dilation: dilation as usize,
        pads,
    }))
}

/// What [`optimize_for_webgpu`] rewrote.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Rewrites {
    /// STFT `Conv` pairs replaced by an FFT.
    pub stft_fft: usize,
    /// iSTFT `ConvTranspose` nodes replaced by an inverse FFT + overlap-add.
    pub istft_fft: usize,
    /// Large-stride `ConvTranspose` nodes replaced by MatMul + overlap-add.
    pub istft_matmul: usize,
    /// Single-group GroupNorms (`InstanceNormalization` over `[N, 1, -1]`) replaced by two-stage
    /// reductions.
    pub group_norm: usize,
    /// Multi-axis `ReduceMean`s split into single-axis ones.
    pub reduce_mean: usize,
    /// Bottleneck 1-D Convs replaced by shifted MatMuls.
    pub conv1d: usize,
}

impl Rewrites {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

impl std::fmt::Display for Rewrites {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "STFT→FFT ×{}, iSTFT→FFT ×{}, iSTFT→MatMul ×{}, GroupNorm ×{}, ReduceMean ×{}, Conv1d ×{}",
            self.stft_fft, self.istft_fft, self.istft_matmul, self.group_norm, self.reduce_mean, self.conv1d
        )
    }
}

/// Rewrites HTDemucs' STFT (a pair of strided `Conv`s) and iSTFT (a strided `ConvTranspose`) into
/// FFTs built from small MatMuls, other large-stride `ConvTranspose`s into MatMul + overlap-add,
/// single-group GroupNorms into two-stage reductions, and multi-axis means into single-axis ones.
///
/// Returns `Ok(None)` when the model contains nothing to rewrite.
pub fn optimize_for_webgpu(model: &[u8]) -> Result<Option<(Vec<u8>, Rewrites)>> {
    let model_fields = parse(model)?;

    // Pad/Slice/Reshape/Squeeze with tensor inputs need opset >= 13.
    let opset = model_fields
        .iter()
        .filter(|f| f.num == model::OPSET_IMPORT)
        .map(|f| parse(f.payload))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .find(|o| read_str(o, opset::DOMAIN).is_empty())
        .and_then(|o| o.iter().find(|f| f.num == opset::VERSION).map(|f| f.varint))
        .unwrap_or(0);
    if opset < 13 {
        return Ok(None);
    }

    let graph_field = match model_fields.iter().filter(|f| f.num == model::GRAPH).collect::<Vec<_>>().as_slice() {
        [g] => **g,
        _ => return Err("expected exactly one graph".into()),
    };
    let graph_fields = parse(graph_field.payload)?;

    // Weights may be initializers or outputs of Constant nodes.
    let initializers: Vec<(&str, &[u8])> = graph_fields
        .iter()
        .filter(|f| f.num == graph::INITIALIZER)
        .map(|f| Ok((read_str(&parse(f.payload)?, tensor::NAME), f.payload)))
        .collect::<Result<_>>()?;
    let nodes: Vec<Node> = graph_fields.iter().filter(|f| f.num == graph::NODE).map(|f| Node::parse(f.payload)).collect::<Result<_>>()?;
    let constant = |name: &str| -> Option<&[u8]> {
        if let Some((_, buf)) = initializers.iter().find(|(n, _)| *n == name) {
            return Some(buf);
        }
        let node = nodes.iter().find(|n| n.op_type == "Constant" && n.domain.is_empty() && n.outputs == [name])?;
        node.attr("value")?.iter().find(|f| f.num == attr::T).map(|f| f.payload)
    };
    let mut weight = |name: &str| -> Result<Option<FloatTensor>> { constant(name).map(FloatTensor::parse).transpose().map(Option::flatten) };
    let to_f32 = |t: &FloatTensor| -> Vec<f32> { t.data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect() };

    // Node index -> replacement nodes (empty to remove the node).
    let mut replacements: std::collections::BTreeMap<usize, Vec<Vec<u8>>> = Default::default();
    let mut new_initializers = Vec::new();
    let mut rewrites = Rewrites::default();

    // STFT: two strided Convs on the same input, one with cos and one with sin DFT weights.
    let mut convs: Vec<(usize, StridedConv)> = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        if let Some(conv) = match_strided_conv(node, "Conv", &mut weight)? {
            convs.push((index, conv));
        }
    }
    for (i, (index_a, a)) in convs.iter().enumerate() {
        let pair: Vec<_> = convs.iter().filter(|(_, c)| c.input == a.input).collect();
        let [(_, first), (index_b, b)] = pair.as_slice() else { continue };
        let k = a.kernel as usize;
        // Handle each pair once, from its first Conv.
        if !std::ptr::eq(first, a) || a.kernel != b.kernel || a.stride != b.stride || a.weight.dims[0] as usize != k / 2 + 1 || fft_split(k).is_none() {
            continue;
        }
        let (wa, wb) = (to_f32(&a.weight), to_f32(&b.weight));
        // The cos conv has the window in its DC row; the sin conv's DC row is zero.
        let a_is_cos = wa[..k].iter().any(|v| *v != 0.0);
        let (cos_conv, sin_conv, cos_w, sin_w) = if a_is_cos { (a, b, &wa, &wb) } else { (b, a, &wb, &wa) };
        let Some(fit) = fit_windowed_dft(cos_w, sin_w, k) else { continue };
        let mut e = Emitter::new(format!("stemgen_stft{i}"), &mut new_initializers, opset);
        emit_stft_fft(&mut e, a.input, cos_conv.output, sin_conv.output, k, a.stride as usize, &fit);
        replacements.insert(*index_a, e.nodes);
        replacements.insert(*index_b, Vec::new());
        rewrites.stft_fft += 1;
    }

    // iSTFT: a strided ConvTranspose whose weight holds the cos then the sin DFT rows.
    for (index, node) in nodes.iter().enumerate() {
        let Some(conv) = match_strided_conv(node, "ConvTranspose", &mut weight)? else { continue };
        let k = conv.kernel as usize;
        let prefix = format!("stemgen_istft{}", rewrites.istft_fft + rewrites.istft_matmul);
        let mut e = Emitter::new(prefix, &mut new_initializers, opset);
        let fit = (conv.weight.dims[0] as usize == 2 * (k / 2 + 1) && fft_split(k).is_some())
            .then(|| {
                let w = to_f32(&conv.weight);
                let (cos_w, sin_w) = w.split_at(w.len() / 2);
                fit_windowed_dft(cos_w, sin_w, k)
            })
            .flatten();
        match fit {
            Some(fit) => {
                emit_istft_fft(&mut e, conv.input, conv.output, k, conv.stride as usize, &fit);
                rewrites.istft_fft += 1;
            }
            None => {
                emit_matmul_overlap_add(&mut e, conv.input, conv.output, &conv.weight, conv.kernel, conv.stride);
                rewrites.istft_matmul += 1;
            }
        }
        replacements.insert(index, e.nodes);
    }
    // Single-group GroupNorm: Reshape(x, [0, 1, -1]) -> InstanceNormalization.
    let producers: std::collections::HashMap<&str, &Node> = nodes.iter().flat_map(|n| n.outputs.iter().map(move |o| (*o, n))).collect();
    for (index, node) in nodes.iter().enumerate() {
        if node.op_type != "InstanceNormalization" || !node.domain.is_empty() || node.inputs.len() != 3 || node.outputs.len() != 1 {
            continue;
        }
        let Some(reshape) = producers.get(node.inputs[0]) else { continue };
        if reshape.op_type != "Reshape" || !reshape.domain.is_empty() || reshape.inputs.len() < 2 || reshape.attr_int("allowzero").unwrap_or(0) != 0 {
            continue;
        }
        if constant(reshape.inputs[1]).map(parse_i64_tensor).transpose()?.flatten() != Some(vec![0, 1, -1]) {
            continue;
        }
        let values = |name: &str| -> Result<Option<Vec<f32>>> { Ok(weight(name)?.map(|t| to_f32(&t))) };
        let identity_affine = values(node.inputs[1])? == Some(vec![1.0]) && values(node.inputs[2])? == Some(vec![0.0]);
        let epsilon = node.attr_float("epsilon").unwrap_or(1e-5);
        let mut e = Emitter::new(format!("stemgen_groupnorm{}", rewrites.group_norm), &mut new_initializers, opset);
        emit_group_norm(&mut e, reshape.inputs[0], node.inputs[1], node.inputs[2], node.outputs[0], epsilon, identity_affine);
        replacements.insert(index, e.nodes);
        rewrites.group_norm += 1;
    }

    // Multi-axis ReduceMean (e.g. normalizing the input over [C, F, T]): WebGPU reduces each output
    // in a single workgroup, so reduce one axis at a time, last first. Every group along an axis
    // has the same size, so the mean of means is exact.
    for (index, node) in nodes.iter().enumerate() {
        if node.op_type != "ReduceMean" || !node.domain.is_empty() || node.outputs.len() != 1 || replacements.contains_key(&index) {
            continue;
        }
        if node.attr_int("keepdims").unwrap_or(1) != 1 {
            continue;
        }
        let mut axes = match (opset >= 18, node.inputs.as_slice()) {
            (false, [_]) => node.attr_ints("axes")?.unwrap_or_default(),
            (true, [_, axes]) => constant(axes).map(parse_i64_tensor).transpose()?.flatten().unwrap_or_default(),
            _ => continue,
        };
        // Without the rank, axes can only be ordered if they all count from the same end.
        if axes.len() < 2 || !(axes.iter().all(|a| *a >= 0) || axes.iter().all(|a| *a < 0)) {
            continue;
        }
        axes.sort_unstable_by(|a, b| b.cmp(a));
        axes.dedup();
        let mut e = Emitter::new(format!("stemgen_reducemean{}", rewrites.reduce_mean), &mut new_initializers, opset);
        let mut x = node.inputs[0].to_owned();
        for (i, axis) in axes.iter().enumerate() {
            x = e.reduce_mean_to(&x, *axis, (i == axes.len() - 1).then_some(node.outputs[0]));
        }
        replacements.insert(index, e.nodes);
        rewrites.reduce_mean += 1;
    }

    // Bottleneck 1-D Convs (HTDemucs' DConv residual branches: C -> C/8 -> 2C channels) ->
    // shifted MatMuls. Convs with balanced channel counts run fine as Convs (no gain measured).
    for (index, node) in nodes.iter().enumerate() {
        if replacements.contains_key(&index) {
            continue;
        }
        let Some(conv) = match_conv1d(node, &mut weight)? else { continue };
        if conv.cin.max(conv.cout) < 4 * conv.cin.min(conv.cout) {
            continue;
        }
        let mut e = Emitter::new(format!("stemgen_conv1d{}", rewrites.conv1d), &mut new_initializers, opset);
        emit_conv1d_matmul(&mut e, &conv);
        replacements.insert(index, e.nodes);
        rewrites.conv1d += 1;
    }

    if rewrites.is_empty() {
        return Ok(None);
    }

    // Re-emit the graph with the replaced nodes swapped in place (keeps topological order).
    let mut new_graph = Vec::with_capacity(graph_field.payload.len() + new_initializers.iter().map(Vec::len).sum::<usize>() + 4096);
    let mut node_index = 0;
    for f in &graph_fields {
        if f.num == graph::NODE {
            match replacements.get(&node_index) {
                Some(new_nodes) => {
                    for n in new_nodes {
                        put_bytes(&mut new_graph, graph::NODE, n);
                    }
                }
                None => new_graph.extend_from_slice(f.raw),
            }
            node_index += 1;
        } else {
            new_graph.extend_from_slice(f.raw);
        }
    }
    for t in &new_initializers {
        put_bytes(&mut new_graph, graph::INITIALIZER, t);
    }

    let mut out = Vec::with_capacity(model.len() + new_graph.len() - graph_field.payload.len());
    for f in &model_fields {
        if f.num == model::GRAPH {
            put_bytes(&mut out, model::GRAPH, &new_graph);
        } else {
            out.extend_from_slice(f.raw);
        }
    }
    Ok(Some((out, rewrites)))
}

#[cfg(test)]
mod tests {
    use super::*;

    use ort::session::Session;
    use ort::value::Tensor;

    fn constant(name: &str, dims: &[i64], values: &[f32]) -> Vec<u8> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut value_attr = Vec::new();
        put_bytes(&mut value_attr, attr::NAME, b"value");
        put_bytes(&mut value_attr, attr::T, &encode_tensor(name, tensor::FLOAT, dims, &bytes));
        put_int(&mut value_attr, attr::TYPE, 4); // TENSOR
        let mut node = Vec::new();
        put_bytes(&mut node, node::OUTPUT, name.as_bytes());
        put_bytes(&mut node, node::NAME, format!("/Constant_{name}").as_bytes());
        put_bytes(&mut node, node::OP_TYPE, b"Constant");
        put_bytes(&mut node, node::ATTRIBUTE, &value_attr);
        node
    }

    /// ValueInfoProto { name = 1, type = 2 { tensor_type = 1 { elem_type = 1, shape = 2 } } }
    fn value_info(name: &str, dims: &[Option<i64>]) -> Vec<u8> {
        let mut shape = Vec::new();
        for d in dims {
            let mut dim = Vec::new();
            match d {
                Some(v) => put_int(&mut dim, 1, *v),
                None => put_bytes(&mut dim, 2, b"dyn"),
            }
            put_bytes(&mut shape, 1, &dim);
        }
        let mut tensor_type = Vec::new();
        put_int(&mut tensor_type, 1, tensor::FLOAT);
        put_bytes(&mut tensor_type, 2, &shape);
        let mut ty = Vec::new();
        put_bytes(&mut ty, 1, &tensor_type);
        let mut vi = Vec::new();
        put_bytes(&mut vi, 1, name.as_bytes());
        put_bytes(&mut vi, 2, &ty);
        vi
    }

    fn model(nodes: &[Vec<u8>], input: (&str, &[Option<i64>]), outputs: &[(&str, &[Option<i64>])]) -> Vec<u8> {
        let mut graph = Vec::new();
        for n in nodes {
            put_bytes(&mut graph, graph::NODE, n);
        }
        put_bytes(&mut graph, 2, b"test"); // name
        put_bytes(&mut graph, 11, &value_info(input.0, input.1));
        for (name, dims) in outputs {
            put_bytes(&mut graph, 12, &value_info(name, dims));
        }
        let mut opset_import = Vec::new();
        put_int(&mut opset_import, opset::VERSION, 17);
        let mut model = Vec::new();
        put_int(&mut model, 1, 8); // ir_version
        put_bytes(&mut model, model::GRAPH, &graph);
        put_bytes(&mut model, model::OPSET_IMPORT, &opset_import);
        model
    }

    fn strided(op: &str, input: &str, weight: &str, output: &str, kernel: i64, stride: i64) -> Vec<u8> {
        encode_node(op, &format!("/{output}"), &[input, weight], &[output], &[("kernel_shape", Attr::Ints(&[kernel])), ("strides", Attr::Ints(&[stride]))])
    }

    fn run(model: &[u8], input: &[f32], shape: &[usize], outputs: &[&str]) -> Vec<Vec<f32>> {
        let mut session = Session::builder().unwrap().commit_from_memory(model).unwrap();
        let x = Tensor::<f32>::from_array((shape.to_vec(), input.to_vec())).unwrap();
        let out = session.run(ort::inputs! { "x" => x }).unwrap();
        outputs.iter().map(|o| out[*o].try_extract_array::<f32>().unwrap().iter().copied().collect()).collect()
    }

    fn assert_close(expected: &[f32], actual: &[f32], tolerance: f32) {
        assert_eq!(expected.len(), actual.len());
        let peak = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
            assert!((e - a).abs() <= tolerance * peak, "at {i}: {e} != {a} (peak {peak})");
        }
    }

    fn pseudo(i: usize, seed: usize) -> f32 {
        ((i * 7919 + seed * 104729) % 1000) as f32 / 500.0 - 1.0
    }

    /// Hann-windowed DFT rows like HTDemucs': `[K/2+1, 1, K]` cos and sin weights.
    fn dft_weights(k: usize, cos_scale: impl Fn(usize) -> f64, sin_scale: impl Fn(usize) -> f64, window_scale: f64) -> (Vec<f32>, Vec<f32>) {
        let bins = k / 2 + 1;
        let window = |n: usize| window_scale * (0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / k as f64).cos());
        let (mut cos_w, mut sin_w) = (Vec::new(), Vec::new());
        for f in 0..bins {
            for n in 0..k {
                let angle = std::f64::consts::TAU * ((f * n) % k) as f64 / k as f64;
                cos_w.push((cos_scale(f) * window(n) * angle.cos()) as f32);
                sin_w.push((sin_scale(f) * window(n) * angle.sin()) as f32);
            }
        }
        (cos_w, sin_w)
    }

    #[test]
    fn test_stft_rewrite_matches_conv() {
        let (k, s, n, l) = (256usize, 64usize, 2usize, 6usize);
        let bins = k / 2 + 1;
        let t = (l - 1) * s + k;
        let (cos_w, sin_w) = dft_weights(k, |_| 1.0, |_| -1.0, 1.0 / 16.0);
        let dims = [bins as i64, 1, k as i64];
        let model = model(
            &[
                constant("wc", &dims, &cos_w),
                constant("ws", &dims, &sin_w),
                strided("Conv", "x", "wc", "re", k as i64, s as i64),
                strided("Conv", "x", "ws", "im", k as i64, s as i64),
            ],
            ("x", &[None, Some(1), Some(t as i64)]),
            &[("re", &[None, Some(bins as i64), None]), ("im", &[None, Some(bins as i64), None])],
        );
        let (patched, rewrites) = optimize_for_webgpu(&model).unwrap().expect("model should be rewritten");
        assert_eq!(rewrites, Rewrites { stft_fft: 1, ..Default::default() });

        let input: Vec<f32> = (0..n * t).map(|i| pseudo(i, 3)).collect();
        let expected = run(&model, &input, &[n, 1, t], &["re", "im"]);
        let actual = run(&patched, &input, &[n, 1, t], &["re", "im"]);
        assert_eq!(expected[0].len(), n * bins * l);
        assert_close(&expected[0], &actual[0], 1e-5);
        assert_close(&expected[1], &actual[1], 1e-5);
    }

    #[test]
    fn test_istft_fft_rewrite_matches_conv_transpose() {
        let (k, s, n, l) = (256usize, 64usize, 3usize, 5usize);
        let bins = k / 2 + 1;
        let doubled = |f: usize| if f == 0 || f == k / 2 { 1.0 } else { 2.0 };
        let (cos_w, sin_w) = dft_weights(k, doubled, |f| -doubled(f), 1.0 / 64.0);
        let weight: Vec<f32> = cos_w.into_iter().chain(sin_w).collect();
        let model = model(
            &[constant("w", &[2 * bins as i64, 1, k as i64], &weight), strided("ConvTranspose", "x", "w", "y", k as i64, s as i64)],
            ("x", &[None, Some(2 * bins as i64), None]),
            &[("y", &[None, Some(1), None])],
        );
        let (patched, rewrites) = optimize_for_webgpu(&model).unwrap().expect("model should be rewritten");
        assert_eq!(rewrites, Rewrites { istft_fft: 1, ..Default::default() });

        let input: Vec<f32> = (0..n * 2 * bins * l).map(|i| pseudo(i, 4)).collect();
        let expected = run(&model, &input, &[n, 2 * bins, l], &["y"]);
        let actual = run(&patched, &input, &[n, 2 * bins, l], &["y"]);
        assert_eq!(expected[0].len(), n * ((l - 1) * s + k));
        assert_close(&expected[0], &actual[0], 1e-5);
    }

    #[test]
    fn test_istft_matmul_rewrite_matches_conv_transpose() {
        // Arbitrary (non-DFT) weights fall back to MatMul + overlap-add; odd Cin exercises the
        // channel padding.
        let (cin, k, s, n, l) = (7usize, 256usize, 64usize, 2usize, 5usize);
        let weight: Vec<f32> = (0..cin * k).map(|i| pseudo(i, 1)).collect();
        let model = model(
            &[constant("w", &[cin as i64, 1, k as i64], &weight), strided("ConvTranspose", "x", "w", "y", k as i64, s as i64)],
            ("x", &[None, Some(cin as i64), None]),
            &[("y", &[None, Some(1), None])],
        );
        let (patched, rewrites) = optimize_for_webgpu(&model).unwrap().expect("model should be rewritten");
        assert_eq!(rewrites, Rewrites { istft_matmul: 1, ..Default::default() });

        let input: Vec<f32> = (0..n * cin * l).map(|i| pseudo(i, 2)).collect();
        let expected = run(&model, &input, &[n, cin, l], &["y"]);
        let actual = run(&patched, &input, &[n, cin, l], &["y"]);
        assert_eq!(expected[0].len(), n * ((l - 1) * s + k));
        assert_close(&expected[0], &actual[0], 1e-5);
    }

    fn constant_i64(name: &str, values: &[i64]) -> Vec<u8> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut value_attr = Vec::new();
        put_bytes(&mut value_attr, attr::NAME, b"value");
        put_bytes(&mut value_attr, attr::T, &encode_tensor(name, tensor::INT64, &[values.len() as i64], &bytes));
        put_int(&mut value_attr, attr::TYPE, 4); // TENSOR
        let mut node = Vec::new();
        put_bytes(&mut node, node::OUTPUT, name.as_bytes());
        put_bytes(&mut node, node::NAME, format!("/Constant_{name}").as_bytes());
        put_bytes(&mut node, node::OP_TYPE, b"Constant");
        put_bytes(&mut node, node::ATTRIBUTE, &value_attr);
        node
    }

    #[test]
    fn test_group_norm_rewrite_matches_instance_norm() {
        let (n, c, t) = (2usize, 6usize, 50usize);
        // Offset input, so a single-pass variance would lose precision.
        let input: Vec<f32> = (0..n * c * t).map(|i| 3.0 + pseudo(i, 5) * (1.0 + (i % 7) as f32)).collect();
        for (scale, bias) in [(1.0f32, 0.0f32), (1.5, -0.25)] {
            let model = model(
                &[
                    constant_i64("shape", &[0, 1, -1]),
                    constant("scale", &[1], &[scale]),
                    constant("bias", &[1], &[bias]),
                    encode_node("Reshape", "/Reshape", &["x", "shape"], &["r"], &[]),
                    encode_node("InstanceNormalization", "/InstanceNormalization", &["r", "scale", "bias"], &["y"], &[]),
                ],
                ("x", &[None, Some(c as i64), None]),
                &[("y", &[None, Some(1), None])],
            );
            let (patched, rewrites) = optimize_for_webgpu(&model).unwrap().expect("model should be rewritten");
            assert_eq!(rewrites, Rewrites { group_norm: 1, ..Default::default() });
            let expected = run(&model, &input, &[n, c, t], &["y"]);
            let actual = run(&patched, &input, &[n, c, t], &["y"]);
            assert_close(&expected[0], &actual[0], 1e-5);
        }
    }

    #[test]
    fn test_reduce_mean_rewrite_matches() {
        let (n, c, h, w) = (2usize, 3usize, 5usize, 7usize);
        let input: Vec<f32> = (0..n * c * h * w).map(|i| pseudo(i, 6)).collect();
        let model = model(
            &[encode_node("ReduceMean", "/ReduceMean", &["x"], &["y"], &[("axes", Attr::Ints(&[1, 2, 3])), ("keepdims", Attr::Int(1))])],
            ("x", &[None, Some(c as i64), Some(h as i64), Some(w as i64)]),
            &[("y", &[None, Some(1), Some(1), Some(1)])],
        );
        let (patched, rewrites) = optimize_for_webgpu(&model).unwrap().expect("model should be rewritten");
        assert_eq!(rewrites, Rewrites { reduce_mean: 1, ..Default::default() });
        let expected = run(&model, &input, &[n, c, h, w], &["y"]);
        let actual = run(&patched, &input, &[n, c, h, w], &["y"]);
        assert_eq!(expected[0].len(), n);
        assert_close(&expected[0], &actual[0], 1e-5);
    }

    #[test]
    fn test_conv1d_rewrite_matches_conv() {
        let (n, l) = (3usize, 11usize);
        // (cin, cout, kernel, dilation, pads, bias): DConv-like shapes plus odd channel counts.
        // (48 -> 6, k=3) and (96 -> 6, k=3, uneven pads) shift outputs, (5 -> 23, k=3) shifts inputs.
        for (cin, cout, k, d, pads, bias) in [
            (48usize, 6usize, 3usize, 2usize, [2i64, 2i64], true),
            (96, 6, 3, 1, [2, 0], true),
            (6, 48, 1, 1, [0, 0], true),
            (5, 23, 3, 1, [1, 0], false),
        ] {
            let weight: Vec<f32> = (0..cout * cin * k).map(|i| pseudo(i, 8)).collect();
            let b: Vec<f32> = (0..cout).map(|i| pseudo(i, 9)).collect();
            let mut inputs = vec!["x", "w"];
            let mut nodes = vec![constant("w", &[cout as i64, cin as i64, k as i64], &weight)];
            if bias {
                nodes.push(constant("b", &[cout as i64], &b));
                inputs.push("b");
            }
            nodes.push(encode_node(
                "Conv",
                "/Conv",
                &inputs,
                &["y"],
                &[("kernel_shape", Attr::Ints(&[k as i64])), ("dilations", Attr::Ints(&[d as i64])), ("pads", Attr::Ints(&pads))],
            ));
            let model = model(&nodes, ("x", &[None, Some(cin as i64), None]), &[("y", &[None, Some(cout as i64), None])]);
            let (patched, rewrites) = optimize_for_webgpu(&model).unwrap().expect("model should be rewritten");
            assert_eq!(rewrites, Rewrites { conv1d: 1, ..Default::default() });
            let input: Vec<f32> = (0..n * cin * l).map(|i| pseudo(i, 10)).collect();
            let expected = run(&model, &input, &[n, cin, l], &["y"]);
            let actual = run(&patched, &input, &[n, cin, l], &["y"]);
            assert_close(&expected[0], &actual[0], 1e-5);
        }
    }

    #[test]
    fn test_rewrite_skips_small_strides() {
        let weight = vec![0.5f32; 3 * 4];
        let model = model(
            &[constant("w", &[3, 1, 4], &weight), strided("ConvTranspose", "x", "w", "y", 4, 2)],
            ("x", &[None, Some(3), None]),
            &[("y", &[None, Some(1), None])],
        );
        assert!(optimize_for_webgpu(&model).unwrap().is_none());
    }

    #[test]
    fn test_fit_rejects_non_dft_weights() {
        let k = 256;
        let (cos_w, mut sin_w) = dft_weights(k, |_| 1.0, |_| -1.0, 1.0);
        assert!(fit_windowed_dft(&cos_w, &sin_w, k).is_some());
        sin_w[5 * k + 17] += 0.1;
        assert!(fit_windowed_dft(&cos_w, &sin_w, k).is_none());
    }
}
