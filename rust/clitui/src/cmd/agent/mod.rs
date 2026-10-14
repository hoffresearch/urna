//! The agent-facing verbs: `ask`, `retrieve`, `build`. They are a product
//! layered OVER the engine verbs in the parent module, not part of the
//! engine: each one shells out to python (the offline query embedder or the
//! forge) and speaks in cited answers, while the engine verbs take vectors
//! and files. Kept in one submodule so the file tree says what `urna
//! --help` says.

pub mod ask;
pub mod build;
pub mod retrieve;
