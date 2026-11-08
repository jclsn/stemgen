//! In-memory ONNX graph rewrites that work around slow kernels in some execution providers.
//!
//! HTDemucs implements its iSTFT as a 1-D `ConvTranspose` with a huge kernel (4096) and stride
//! (1024). ONNX Runtime's WebGPU kernel for that shape is pathologically slow (~85% of the whole
//! inference time), so [`rewrite_istft_conv_transpose`] replaces it with the equivalent
//! `MatMul` + overlap-add, which maps onto WebGPU's fast matrix multiplication shader.
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
    pub const I: u32 = 3;
    pub const S: u32 = 4;
    pub const T: u32 = 5;
    pub const INTS: u32 = 8;
    pub const TYPE: u32 = 20;
    pub const TYPE_INTS: i64 = 7;
}
mod tensor {
    pub const DIMS: u32 = 1;
    pub const DATA_TYPE: u32 = 2;
    pub const FLOAT_DATA: u32 = 4;
    pub const NAME: u32 = 8;
    pub const RAW_DATA: u32 = 9;
    pub const DATA_LOCATION: u32 = 14;
    pub const FLOAT: i64 = 1;
    pub const INT64: i64 = 7;
}

struct Node<'a> {
    inputs: Vec<&'a str>,
    outputs: Vec<&'a str>,
    name: &'a str,
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
            name: read_str(&fields, node::NAME),
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

fn encode_node(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str], ints_attrs: &[(&str, &[i64])]) -> Vec<u8> {
    let mut out = Vec::new();
    for i in inputs {
        put_bytes(&mut out, node::INPUT, i.as_bytes());
    }
    for o in outputs {
        put_bytes(&mut out, node::OUTPUT, o.as_bytes());
    }
    put_bytes(&mut out, node::NAME, name.as_bytes());
    put_bytes(&mut out, node::OP_TYPE, op_type.as_bytes());
    for (attr_name, values) in ints_attrs {
        let mut a = Vec::new();
        put_bytes(&mut a, attr::NAME, attr_name.as_bytes());
        for v in *values {
            put_int(&mut a, attr::INTS, *v);
        }
        put_int(&mut a, attr::TYPE, attr::TYPE_INTS);
        put_bytes(&mut out, node::ATTRIBUTE, &a);
    }
    out
}

// --- The rewrite ------------------------------------------------------------------------------

/// The parts of a matching `ConvTranspose` needed to rewrite it.
struct IstftConv<'a> {
    input: &'a str,
    output: &'a str,
    name: &'a str,
    weight: FloatTensor<'a>,
    kernel: i64,
    stride: i64,
}

fn match_istft_conv<'a>(node: &Node<'a>, weights: &mut dyn FnMut(&str) -> Result<Option<FloatTensor<'a>>>) -> Result<Option<IstftConv<'a>>> {
    if node.op_type != "ConvTranspose" || !node.domain.is_empty() || node.inputs.len() != 2 || node.outputs.len() != 1 {
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
        // Only worth it for large strides (the iSTFT case), where the frames barely overlap.
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
    Ok(Some(IstftConv { input: node.inputs[0], output: node.outputs[0], name: node.name, weight, kernel, stride }))
}

/// Emits nodes and initializers computing
/// `ConvTranspose(X[N, Cin, L], W[Cin, 1, K], stride S)` as
/// `overlap_add(Transpose(X) @ W[:, 0, :], hop = S)`.
fn emit_matmul_overlap_add(conv: &IstftConv, prefix: &str, nodes: &mut Vec<Vec<u8>>, initializers: &mut Vec<Vec<u8>>) {
    let (cin, k, s) = (conv.weight.dims[0], conv.kernel, conv.stride);
    let r = k / s;
    let name = |suffix: &str| format!("{prefix}/{suffix}");
    let int64 = |initializers: &mut Vec<Vec<u8>>, suffix: &str, values: &[i64]| {
        let n = name(suffix);
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        initializers.push(encode_tensor(&n, tensor::INT64, &[values.len() as i64], &bytes));
        n
    };

    // Zero-pad Cin to a multiple of 4 so WebGPU picks its vectorized (vec4) MatMul shader.
    let cpad = (4 - cin % 4) % 4;
    let mut x = conv.input.to_owned();
    if cpad != 0 {
        let pads = int64(initializers, "pad_channels", &[0, 0, 0, 0, cpad, 0]);
        let out = name("x_padded");
        nodes.push(encode_node("Pad", &name("PadChannels"), &[&x, &pads], &[&out], &[]));
        x = out;
    }
    let xt = name("x_t");
    nodes.push(encode_node("Transpose", &name("TransposeX"), &[&x], &[&xt], &[("perm", &[0, 2, 1])])); // [N, L, Cin]
    let flat = int64(initializers, "shape_flat", &[0, 1, -1]);

    // One MatMul per hop-sized slice of the kernel: slice j of every frame is exactly the part
    // that lands j*S samples later in the overlap-add. Splitting also keeps each weight small:
    // a single [Cin, K] weight (~67MB for HTDemucs) is sometimes placed where WebGPU reads it
    // ~15x slower, especially on GPUs without resizable BAR.
    let w = &conv.weight.data; // [Cin, 1, K] == [Cin, K] in row-major order
    let row = (k * 4) as usize;
    let mut acc: Option<String> = None;
    for j in 0..r {
        let mut wj = Vec::with_capacity(((cin + cpad) * s * 4) as usize);
        for c in 0..cin as usize {
            let start = c * row + (j * s * 4) as usize;
            wj.extend_from_slice(&w[start..start + (s * 4) as usize]);
        }
        wj.resize(((cin + cpad) * s * 4) as usize, 0);
        let w_name = name(&format!("weight{j}"));
        initializers.push(encode_tensor(&w_name, tensor::FLOAT, &[cin + cpad, s], &wj));

        let pads = int64(initializers, &format!("pads{j}"), &[0, 0, j * s, 0, 0, (r - 1 - j) * s]);
        let (frames, seg, padded) = (name(&format!("frames{j}")), name(&format!("segment{j}")), name(&format!("segment{j}_padded")));
        nodes.push(encode_node("MatMul", &name(&format!("MatMul{j}")), &[&xt, &w_name], &[&frames], &[])); // [N, L, S]
        nodes.push(encode_node("Reshape", &name(&format!("Reshape{j}")), &[&frames, &flat], &[&seg], &[])); // [N, 1, L*S]
        nodes.push(encode_node("Pad", &name(&format!("Pad{j}")), &[&seg, &pads], &[&padded], &[]));
        acc = Some(match acc {
            None => padded,
            Some(prev) => {
                let out = if j == r - 1 { conv.output.to_owned() } else { name(&format!("sum{j}")) };
                nodes.push(encode_node("Add", &name(&format!("Add{j}")), &[&prev, &padded], &[&out], &[]));
                out
            }
        });
    }
    if r == 1 {
        // Single segment: nothing to add, just forward it.
        nodes.push(encode_node("Identity", &name("Identity"), &[acc.as_deref().unwrap()], &[conv.output], &[]));
    }
}

/// Rewrites large-stride 1-D `ConvTranspose` nodes (HTDemucs' iSTFT) into `MatMul` + overlap-add.
///
/// Returns `Ok(None)` when the model contains nothing to rewrite.
pub fn rewrite_istft_conv_transpose(model: &[u8]) -> Result<Option<Vec<u8>>> {
    let model_fields = parse(model)?;

    // Pad/Slice/Reshape with tensor inputs need opset >= 13.
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
    let nodes: Vec<(Field, Node)> = graph_fields
        .iter()
        .filter(|f| f.num == graph::NODE)
        .map(|f| Ok((*f, Node::parse(f.payload)?)))
        .collect::<Result<_>>()?;
    let mut weight = |name: &str| -> Result<Option<FloatTensor>> {
        if let Some((_, buf)) = initializers.iter().find(|(n, _)| *n == name) {
            return FloatTensor::parse(buf);
        }
        let constant = nodes.iter().find(|(_, n)| n.op_type == "Constant" && n.domain.is_empty() && n.outputs == [name]);
        match constant.and_then(|(_, n)| n.attr("value")).and_then(|a| a.iter().find(|f| f.num == attr::T)) {
            Some(t) => FloatTensor::parse(t.payload),
            None => Ok(None),
        }
    };

    let mut replacements: Vec<(usize, Vec<Vec<u8>>)> = Vec::new();
    let mut new_initializers = Vec::new();
    for (index, (_, node)) in nodes.iter().enumerate() {
        if let Some(conv) = match_istft_conv(node, &mut weight)? {
            let prefix = format!("stemgen_istft{}{}", replacements.len(), conv.name.replace('/', "_"));
            let mut new_nodes = Vec::new();
            emit_matmul_overlap_add(&conv, &prefix, &mut new_nodes, &mut new_initializers);
            replacements.push((index, new_nodes));
        }
    }
    if replacements.is_empty() {
        return Ok(None);
    }

    // Re-emit the graph with the replaced nodes swapped in place (keeps topological order).
    let mut new_graph = Vec::with_capacity(graph_field.payload.len() + new_initializers.iter().map(Vec::len).sum::<usize>() + 4096);
    let mut node_index = 0;
    let mut replacements = replacements.into_iter().peekable();
    for f in &graph_fields {
        if f.num == graph::NODE {
            if replacements.peek().is_some_and(|(i, _)| *i == node_index) {
                for n in replacements.next().unwrap().1 {
                    put_bytes(&mut new_graph, graph::NODE, &n);
                }
            } else {
                new_graph.extend_from_slice(f.raw);
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
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    use ort::session::Session;
    use ort::value::Tensor;

    /// Builds a model with a single `ConvTranspose` whose weight is a `Constant` node.
    fn conv_transpose_model(cin: i64, kernel: i64, stride: i64, weight: &[f32]) -> Vec<u8> {
        let w_bytes: Vec<u8> = weight.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut value_attr = Vec::new();
        put_bytes(&mut value_attr, attr::NAME, b"value");
        put_bytes(&mut value_attr, attr::T, &encode_tensor("w", tensor::FLOAT, &[cin, 1, kernel], &w_bytes));
        put_int(&mut value_attr, attr::TYPE, 4); // TENSOR
        let mut constant = Vec::new();
        put_bytes(&mut constant, node::OUTPUT, b"w");
        put_bytes(&mut constant, node::NAME, b"/Constant");
        put_bytes(&mut constant, node::OP_TYPE, b"Constant");
        put_bytes(&mut constant, node::ATTRIBUTE, &value_attr);

        let conv = encode_node("ConvTranspose", "/ConvTranspose", &["x", "w"], &["y"], &[("kernel_shape", &[kernel]), ("strides", &[stride])]);

        // ValueInfoProto { name = 1, type = 2 { tensor_type = 1 { elem_type = 1, shape = 2 } } }
        let value_info = |name: &str, dims: &[Option<i64>]| {
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
        };

        let mut graph = Vec::new();
        put_bytes(&mut graph, graph::NODE, &constant);
        put_bytes(&mut graph, graph::NODE, &conv);
        put_bytes(&mut graph, 2, b"test"); // name
        put_bytes(&mut graph, 11, &value_info("x", &[None, Some(cin), None]));
        put_bytes(&mut graph, 12, &value_info("y", &[None, Some(1), None]));

        let mut opset_import = Vec::new();
        put_int(&mut opset_import, opset::VERSION, 17);
        let mut model = Vec::new();
        put_int(&mut model, 1, 8); // ir_version
        put_bytes(&mut model, model::GRAPH, &graph);
        put_bytes(&mut model, model::OPSET_IMPORT, &opset_import);
        model
    }

    fn run(model: &[u8], input: &[f32], shape: [usize; 3]) -> Vec<f32> {
        let mut session = Session::builder().unwrap().commit_from_memory(model).unwrap();
        let x = Tensor::<f32>::from_array((shape, input.to_vec())).unwrap();
        let out = session.run(ort::inputs! { "x" => x }).unwrap();
        out["y"].try_extract_array::<f32>().unwrap().iter().copied().collect()
    }

    #[test]
    fn test_istft_rewrite_matches_conv_transpose() {
        // Odd Cin exercises the channel padding; R = K / S = 4 like HTDemucs.
        let (cin, kernel, stride, n, l) = (7i64, 256i64, 64i64, 2usize, 5usize);
        let pseudo = |i: usize, seed: usize| ((i * 7919 + seed * 104729) % 1000) as f32 / 500.0 - 1.0;
        let weight: Vec<f32> = (0..(cin * kernel) as usize).map(|i| pseudo(i, 1)).collect();
        let input: Vec<f32> = (0..n * cin as usize * l).map(|i| pseudo(i, 2)).collect();

        let model = conv_transpose_model(cin, kernel, stride, &weight);
        let patched = rewrite_istft_conv_transpose(&model).unwrap().expect("model should be rewritten");

        let expected = run(&model, &input, [n, cin as usize, l]);
        let actual = run(&patched, &input, [n, cin as usize, l]);
        assert_eq!(expected.len(), n * ((l - 1) * stride as usize + kernel as usize));
        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(&actual) {
            assert!((e - a).abs() < 1e-4, "{e} != {a}");
        }
    }

    #[test]
    fn test_istft_rewrite_skips_small_strides() {
        let weight = vec![0.5f32; 3 * 4];
        let model = conv_transpose_model(3, 4, 2, &weight);
        assert!(rewrite_istft_conv_transpose(&model).unwrap().is_none());
    }
}
