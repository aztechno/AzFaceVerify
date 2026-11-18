//! Thin wrapper around the `ort` crate.
//!
//! This is the ONLY file that touches the ONNX Runtime tensor API.
//! If you upgrade `ort` to a different release-candidate and get a compile
//! error, it will almost certainly be inside `run_f32` below.

use anyhow::{Context, Result};
use ort::{
    execution_providers::{
        CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider,
        ExecutionProviderDispatch,
    },
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::Path;

/// Which hardware backend ONNX Runtime should try to use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provider {
    Cuda,
    DirectMl,
    Cpu,
}

impl Provider {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "cuda" | "gpu" | "nvidia" => Self::Cuda,
            "directml" | "dml" | "dx12" => Self::DirectMl,
            _ => Self::Cpu,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Cuda => "CUDA (NVIDIA)",
            Self::DirectMl => "DirectML (D3D12)",
            Self::Cpu => "CPU",
        }
    }
}

/// Build an ONNX Runtime session for the given model file.
pub fn build_session(path: &Path, p: Provider, threads: usize) -> Result<Session> {
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();

    match p {
        Provider::Cuda => eps.push(CUDAExecutionProvider::default().with_device_id(0).build()),
        Provider::DirectMl => eps.push(DirectMLExecutionProvider::default().with_device_id(0).build()),
        Provider::Cpu => {}
    }
    eps.push(CPUExecutionProvider::default().build());

    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?;

    if p == Provider::DirectMl {
        builder = builder.with_memory_pattern(false)?;
    }

    builder
        .with_execution_providers(eps)?
        .commit_from_file(path)
        .with_context(|| format!("failed to load model {}", path.display()))
}

/// Name of the first (and only) model input.
pub fn input_name(s: &Session) -> String {
    s.inputs[0].name.clone()
}

/// Run a session with a single NCHW float32 input and collect every output.
/// NOTE: ort 2.x requires `&mut Session` to run inference.
pub fn run_f32(
    session: &mut Session,
    name: &str,
    shape: &[i64],
    data: Vec<f32>,
    n_outputs: usize,
) -> Result<Vec<Vec<f32>>> {
    let tensor = Tensor::from_array((shape.to_vec(), data))?;

    // (A) In ort 2.x, inputs! macro returns a plain Vec, and run() requires &mut self.
    let outputs = session.run(ort::inputs![name => tensor])?;

    let mut res: Vec<Vec<f32>> = Vec::with_capacity(n_outputs);
    for i in 0..n_outputs {
        let (_shape, slice) = outputs[i].try_extract_tensor::<f32>()?;
        res.push(slice.to_vec());
    }

    if res.len() != n_outputs {
        anyhow::bail!("expected {n_outputs} model outputs, got {}", res.len());
    }
    Ok(res)
}