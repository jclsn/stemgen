use std::path::{Path, PathBuf};
use ort::value::{TensorElementType};
use ort::value::Shape;
use ort::value::ValueType;
use ndarray::{ArrayViewMut, ShapeBuilder};
use ort::{session::{builder::GraphOptimizationLevel, Session}, value::Tensor};
use std::time::Instant;

#[cfg(feature = "cuda")]
use ort::{execution_providers::CUDAExecutionProvider};
#[cfg(feature = "migraphx")]
use ort::{execution_providers::MIGraphXExecutionProvider};

use crate::constant::DEFAULT_MODEL;

#[derive(Debug)]
pub struct Demucs {
    session: Session,
    input_name: String,
    output_name: String,
    input_buffer: Vec<f32>,
}

#[derive(Debug, Clone)]
pub enum Model {
    Local(PathBuf),
    Url(String)
}

impl Default for Model {
    fn default() -> Self {
        Model::Url(DEFAULT_MODEL.to_owned())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum Device {
    #[default]
    CPU,
    #[cfg(feature = "cuda")]
    CUDA,
    #[cfg(feature = "migraphx")]
    MIGraphX
}

impl std::fmt::Display for Device {
     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "cuda")]
            Device::CUDA => write!(f, "cuda"),
            #[cfg(feature = "migraphx")]
            Device::MIGraphX => write!(f, "migraphx"),
            Device::CPU => write!(f, "cpu"),
        }
    }
}

impl TryFrom<&str> for Device {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            #[cfg(feature = "cuda")]
            "cuda" => Ok(Device::CUDA),
            #[cfg(feature = "migraphx")]
            "migraphx" => Ok(Device::MIGraphX),
            "cpu" => Ok(Device::CPU),
            _ => Err("unsupported device".to_owned()),
        }
    }
}

impl std::fmt::Display for Model {
     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match  self {
            Model::Local(path_buf) => write!(f, "{}", path_buf.to_str().unwrap()),
            Model::Url(url) => write!(f, "{url}"),
        }
    }
}

impl TryFrom<&str> for Model {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.starts_with("http") {
            Ok(Self::Url(value.to_owned()))
        } else {
            let path = Path::new(&value);
            if !path.exists() {
                Err("unable to find the model".to_owned())
            } else {
                Ok(Self::Local(path.to_path_buf()))
            }
        }
    }
}

pub struct DemusOpts {
    pub device: Device,
    pub threads: usize
}

impl Default for DemusOpts {
    fn default() -> Self {
        Self { threads: 2, device: Device::CPU }
    }
}

impl Demucs {
    pub fn new_from_file(model: &Model, ops: DemusOpts) -> Result<Self, Box<dyn std::error::Error>> {
        eprintln!("[DEMUCS] Creating new Demucs instance with device: {}", ops.device);
        let init_start = Instant::now();
        ort::init()
            .with_execution_providers(
            match ops.device {
                #[cfg(feature = "cuda")]
                Device::CUDA => {
                    eprintln!("[DEMUCS] Using CUDA execution provider");
                    vec![
                        CUDAExecutionProvider::default()
                            .with_tf32(true)
                            // TODO support specific device passing?
                            .with_device_id(0)
                            // FIXME seem to wrongly set the memory limit to 0?
                            // .with_memory_limit(1 * 1024 * 1024 * 1024)
                            .build()
                            .error_on_failure()
                    ]
                },
                #[cfg(feature = "migraphx")]
                Device::MIGraphX => {
                    eprintln!("[DEMUCS] Attempting to use MIGraphX execution provider");
                    let provider = MIGraphXExecutionProvider::default()
                        .with_device_id(0)
                        .build()
                        .error_on_failure();
                    eprintln!("[DEMUCS] MIGraphX execution provider configured");
                    vec![provider]
                },
                Device::CPU => {
                    eprintln!("[DEMUCS] Using CPU execution provider");
                    vec![]
                }
            })
            .commit();

        let init_duration = init_start.elapsed();
        eprintln!("[DEMUCS] ORT initialization took: {:?}", init_duration);

        let session_start = Instant::now();
        let mut session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(ops.threads)?;

        let session = match model {
            Model::Local(path) => {
                eprintln!("[DEMUCS] Loading model from local path: {:?}", path);
                session.commit_from_file(path)
            },
            Model::Url(url) => {
                eprintln!("[DEMUCS] Loading model from URL: {}", url);
                session.commit_from_url(url)
            },
        }?;

        let session_duration = session_start.elapsed();
        eprintln!("[DEMUCS] Model loading took: {:?}", session_duration);

        if session.inputs().len() != 1 {
            return Err("expected model to have one input".into())
        }

        if session.outputs().len() != 1 {
            return Err("expected model to have one output".into())
        }

        let input_name = {
            let input = session.inputs().first().unwrap();
            match input.dtype() {
                ValueType::Tensor {
                    ty: TensorElementType::Float32,
                    shape,
                    ..
                    // TODO support multiple buffer length and channel
                } if *shape == Shape::new([1, 2, 343980]) => {
                    eprintln!("[DEMUCS] Input tensor shape: {:?}", shape);
                    Ok(input.name().to_owned())
                }
                _ => {
                    Err(format!("unsupported input format: {}", input.dtype()))
                }
            }
        }?;

        let output_name = {
            let output = session.outputs().first().unwrap();
            match output.dtype() {
                ValueType::Tensor {
                    ty: TensorElementType::Float32,
                    shape,
                    ..
                    // TODO support multiple buffer length and channel
                } if *shape == Shape::new([1, 4, 2, 343980]) => {
                    eprintln!("[DEMUCS] Output tensor shape: {:?}", shape);
                    Ok(output.name().to_owned())
                }
                _ => {
                    Err(format!("unsupported output format: {}", output.dtype()))
                }
            }
        }?;

        eprintln!("[DEMUCS] Demucs instance created successfully");
        Ok(Self {
            session,
            input_name,
            output_name,
            input_buffer: Vec::with_capacity(2 * 343980),
        })

    }

    fn process(&mut self) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let process_start = Instant::now();
        eprintln!("[DEMUCS] Starting process() - input buffer len: {}", self.input_buffer.len());

        let tensor_start = Instant::now();
        let tensor = Tensor::<f32>::from_array(ArrayViewMut::from_shape((1, 2, 343980).strides((343980 * 2, 1, 2)), &mut self.input_buffer)?.to_owned())?;
        let tensor_duration = tensor_start.elapsed();
        eprintln!("[DEMUCS] Tensor creation took: {:?}", tensor_duration);

        let run_start = Instant::now();
        let result = self.session.run(ort::inputs! {
            &self.input_name => tensor
        })?;
        let run_duration = run_start.elapsed();
        eprintln!("[DEMUCS] Model inference took: {:?}", run_duration);

        let extract_start = Instant::now();
        let output = result[self.output_name.as_str()].try_extract_array::<f32>()?;
        let extract_duration = extract_start.elapsed();
        eprintln!("[DEMUCS] Output extraction took: {:?}", extract_duration);
        eprintln!("[DEMUCS] Output tensor shape: {:?}", output.shape());

        let mut stems = vec![Vec::new(); 4];

        // Pre-allocate stems to avoid multiple allocations
        let alloc_start = Instant::now();
        for stem in stems.iter_mut() {
            stem.reserve(2 * 343980);
        }
        let alloc_duration = alloc_start.elapsed();
        eprintln!("[DEMUCS] Stem allocation took: {:?}", alloc_duration);

        // Process all channels in a more efficient manner
        let process_loop_start = Instant::now();
        // Process all samples in a single pass with better locality
        for sample_idx in 0..343980 {
            for channel in 0..4 {
                let l_val = output[[0, channel, 0, sample_idx]];
                let r_val = output[[0, channel, 1, sample_idx]];
                stems[channel].push(l_val);
                stems[channel].push(r_val);
            }
        }
        let process_loop_duration = process_loop_start.elapsed();
        eprintln!("[DEMUCS] Processing loop took: {:?}", process_loop_duration);

        if self.input_buffer.len() == 2 * 343980 {
            self.input_buffer.clear();
        } else {
            let leftover = self.input_buffer.len() - 2 * 343980;
            let (left, right) = self.input_buffer.split_at_mut(2 * 343980);
            left[..leftover].copy_from_slice(right);
            self.input_buffer.resize(leftover, 0.0);
        }

        let process_duration = process_start.elapsed();
        eprintln!("[DEMUCS] Total process() took: {:?}", process_duration);
        Ok(stems)
    }

    pub fn send(&mut self, sample_buffer: &[f32]) -> Result<Option<Vec<Vec<f32>>>, Box<dyn std::error::Error>> {
        if sample_buffer.len() % 2 != 0 {
            return Err("uneven number of sample".into());
        }

        self.input_buffer.extend_from_slice(sample_buffer);

        if self.input_buffer.len() >= 2 * 343980 {
            Ok(Some(self.process()?))
        } else {
            Ok(None)
        }
    }

    pub fn flush(&mut self) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let buffer_size = self.input_buffer.len();
        self.input_buffer.resize(2 * 343980, 0.0);
        let mut data = self.process()?;
        for stem in data.iter_mut() {
            stem.resize(buffer_size, 0.0f32);
        }
        Ok(data)
    }

}
