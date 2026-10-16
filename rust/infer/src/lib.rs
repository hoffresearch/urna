//! urna's inference runtime: the models urna ships, each written once over
//! the `Backend` trait, with the native CPU backend as the default. The
//! potion static embedder is the first model and fixes the pattern:
//! tokenizer, loader, operations and graph all live here, with no
//! inference framework in the build. Operations enter the trait with the
//! first model that uses them; today that is the potion's three (gather,
//! mean pool, l2 normalize), the rest are declared and refused.

pub mod backend;
pub mod model;
pub mod quant;
pub mod registry;
pub mod safetensors;
pub mod tokenizer;

pub use backend::{Backend, Native, Op};
pub use model::StaticModel;
pub use registry::{Graph, ModelSpec, Registry, TokenizerKind};
pub use safetensors::SafeTensors;
pub use tokenizer::Tokenizer;

/// Every failure the runtime reports, typed so a caller can branch on it.
#[derive(Debug, thiserror::Error)]
pub enum InferError {
    #[error("operation {0:?} is not implemented by this backend yet")]
    Unsupported(Op),
    #[error("shape: {0}")]
    Shape(String),
    #[error("{path}: {reason}")]
    SafeTensors { path: String, reason: String },
    #[error("{path} is a git-lfs pointer; run `git lfs pull`")]
    LfsPointer { path: String },
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, InferError>;
