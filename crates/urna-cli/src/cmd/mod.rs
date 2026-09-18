//! CLI subcommand implementations. One module per engine subcommand,
//! orchestrated by `main::Commands`; the agent verbs (`ask`, `retrieve`,
//! `build`) live under `agent/`. Shared helpers: `util` (output),
//! `embed_gate` (the ONE query-embedder spawn + model gate, used by
//! search-text, ask, retrieve and doctor), `pyenv` (interpreter lookup).

pub mod agent;
pub mod benchmark;
pub mod cite;
pub mod doctor;
pub mod embed_gate;
pub mod inspect;
pub mod media;
pub mod pyenv;
pub mod search;
pub mod search_ann;
pub mod search_graph;
pub mod search_space;
pub mod search_text;
pub mod stats;
pub mod util;
pub mod validate;
