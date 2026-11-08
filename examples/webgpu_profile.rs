//! Profiles HTDemucs inference on the WebGPU execution provider, per operator, on the GPU.
//!
//! ```sh
//! cargo run --release --example webgpu_profile --features webgpu -- [model.onnx|URL]
//! ```
//!
//! Runs one warm-up and `RUNS` (default 3) timed inferences on a single chunk, then prints the
//! wall time per run and a breakdown of GPU time by operator type for the last run, using the
//! timestamp queries the WebGPU EP records when profiling is enabled.
//!
//! Environment: `RUNS=<n>`, `NO_PATCH=1` (skip the iSTFT rewrite), `KEEP_PROFILE=1` (keep the
//! raw ORT profile JSON), `CONFIG="key=value;..."` (extra session config entries).

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use ort::ep::WebGPU;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use stemgen::constant::DEFAULT_MODEL;

const CHUNK: usize = 343980;

fn main() {
    let code = match run() {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("Error: {err}");
            1
        }
    };
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // Skip library teardown, which crashes with the WebGPU EP (see cli/src/main.rs).
    unsafe { libc::_exit(code) };
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_MODEL.to_owned());
    let runs: usize = std::env::var("RUNS").ok().map(|v| v.parse()).transpose()?.unwrap_or(3);
    let patch = std::env::var_os("NO_PATCH").is_none();


    let path = if model.starts_with("http") { stemgen::demucs::fetch_model(&model)? } else { model.into() };
    let bytes = std::fs::read(&path)?;
    let bytes = match patch.then(|| stemgen::onnx_patch::rewrite_istft_conv_transpose(&bytes)).transpose()? {
        Some(Some(patched)) => {
            println!("iSTFT rewrite: applied");
            patched
        }
        _ => {
            println!("iSTFT rewrite: {}", if patch { "nothing to rewrite" } else { "disabled" });
            bytes
        }
    };

    let profile_prefix = std::env::temp_dir().join(format!("stemgen_webgpu_profile_{}", std::process::id()));
    let start = Instant::now();
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .with_profiling(&profile_prefix)?;
    // CONFIG="key=value;..." adds raw session config entries, e.g. WebGPU EP options as
    // "ep.webgpuexecutionprovider.<option>=<value>".
    if let Ok(config) = std::env::var("CONFIG") {
        for entry in config.split(';').filter(|e| !e.is_empty()) {
            let (key, value) = entry.split_once('=').ok_or("CONFIG entries must be key=value")?;
            builder = builder.with_config_entry(key, value)?;
        }
    }
    // Registered after the config entries, which the EP reads when it is created.
    let mut session = builder
        .with_execution_providers([WebGPU::default().with_device_id(0).build().error_on_failure()])?
        .commit_from_memory(&bytes)?;
    drop(bytes);
    println!("session creation: {:?}", start.elapsed());

    // Deterministic, non-trivial stereo signal.
    let input: Vec<f32> = (0..CHUNK * 2)
        .map(|i| {
            let t = (i % CHUNK) as f32 / 44100.0;
            0.3 * (t * 440.0 * std::f32::consts::TAU).sin() + 0.2 * (t * 97.0 * std::f32::consts::TAU + (i / CHUNK) as f32).sin()
        })
        .collect();
    for i in 0..=runs {
        let tensor = Tensor::<f32>::from_array(([1usize, 2, CHUNK], input.clone()))?;
        let start = Instant::now();
        let outputs = session.run(ort::inputs! { "input" => tensor })?;
        let _ = outputs[0].try_extract_array::<f32>()?;
        println!("{} {:?}", if i == 0 { "warm-up" } else { "run    " }, start.elapsed());
    }

    let profile_path = session.end_profiling()?;
    let profile: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&profile_path)?)?;
    if std::env::var_os("KEEP_PROFILE").is_some() {
        println!("raw profile: {profile_path}");
    } else {
        let _ = std::fs::remove_file(&profile_path);
    }

    // GPU timestamps are recorded as "Api" events named "<node>&<op>&<op>"; each run emits the
    // same number of dispatches, so the last run is the last 1/(runs+1) of them.
    let dispatches: Vec<(&str, &str, f64, &str)> = profile
        .iter()
        .filter(|e| e["cat"] == "Api")
        .filter_map(|e| {
            let mut name = e["name"].as_str()?.split('&');
            let node = name.next()?;
            let op = name.next()?;
            Some((node, op, e["dur"].as_f64()? / 1000.0, e["args"]["shapes"].as_str().unwrap_or("")))
        })
        .collect();
    if dispatches.is_empty() {
        return Err("no GPU timestamps in the profile (is the WebGPU EP in use?)".into());
    }
    let last = &dispatches[dispatches.len() - dispatches.len() / (runs + 1)..];

    let total: f64 = last.iter().map(|d| d.2).sum();
    println!("\nGPU time of the last run: {total:.1} ms over {} dispatches\n", last.len());
    let mut by_op: HashMap<&str, (f64, usize)> = HashMap::new();
    for (_, op, ms, _) in last {
        let entry = by_op.entry(op).or_default();
        entry.0 += ms;
        entry.1 += 1;
    }
    let mut by_op: Vec<_> = by_op.into_iter().collect();
    by_op.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    println!("{:<24} {:>10} {:>6} {:>6}", "op", "ms", "%", "count");
    for (op, (ms, count)) in by_op.iter().take(15) {
        println!("{op:<24} {ms:>10.1} {:>5.1}% {count:>6}", 100.0 * ms / total);
    }

    let mut slowest: Vec<_> = last.iter().collect();
    slowest.sort_by(|a, b| b.2.total_cmp(&a.2));
    println!("\nslowest dispatches:");
    for (node, op, ms, shapes) in slowest.iter().take(15) {
        println!("{ms:>8.1} ms  {op:<22} {node}\n             {shapes}");
    }
    Ok(())
}
