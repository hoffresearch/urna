//! urna's inference runtime: the models urna ships, each written once over
//! the `Backend` trait, with the native CPU backend as the default. The
//! potion static embedder is the first model and fixes the pattern:
//! tokenizer, loader, operations and graph all live here, with no
//! inference framework in the build. Operations enter the trait with the
//! first model that uses them; today that is the potion's three (gather,
//! mean pool, l2 normalize), the rest are declared and refused.

pub mod backend;
pub mod model;
pub mod potion;
pub mod registry;
pub mod safetensors;
pub mod tokenizer;

pub use backend::{Backend, Native, Op};
pub use model::StaticModel;
pub use potion::Potion;
pub use registry::{Graph, ModelSpec, Registry, TokenizerKind};
pub use safetensors::SafeTensors;
pub use tokenizer::Tokenizer;

/// Every failure the runtime reports, typed so a caller can branch on it.
#[derive(Debug)]
pub enum InferError {
    Unsupported(Op),
    Shape(String),
    SafeTensors { path: String, reason: String },
    LfsPointer { path: String },
    Tokenizer(String),
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(v0) => {
                write!(f, "operation {v0:?} is not implemented by this backend yet")
            }
            Self::Shape(v0) => write!(f, "shape: {v0}"),
            Self::SafeTensors { path, reason } => write!(f, "{path}: {reason}"),
            Self::LfsPointer { path } => {
                write!(f, "{path} is a git-lfs pointer; run `git lfs pull`")
            }
            Self::Tokenizer(v0) => write!(f, "tokenizer: {v0}"),
            Self::Io(v0) => std::fmt::Display::fmt(v0, f),
            Self::Json(v0) => std::fmt::Display::fmt(v0, f),
        }
    }
}

impl std::error::Error for InferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => std::error::Error::source(e),
            Self::Json(e) => std::error::Error::source(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for InferError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for InferError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

pub type Result<T> = std::result::Result<T, InferError>;
