//! CLI subcommand implementations. One module per engine subcommand,
//! orchestrated by `main::Commands`; the agent verbs (`ask`, `retrieve`,
//! `build`) live under `agent/`. Shared helpers: `util` (output),
//! `embed_gate` (the ONE query-embedder spawn + model gate, used by
//! search-text, ask, retrieve and doctor), `pyenv` (interpreter lookup),
//! `paths` (the data roots the installer writes and every reader searches),
//! `health` (the doctor checks as data, shared with the terminal ui) and
//! `tone` (palette color for plain output, off when piped).

pub mod agent;
pub mod benchmark;
pub mod buildid;
pub mod cite;
pub mod doctor;
pub mod embed_failure;
pub mod embed_gate;
pub mod health;
pub mod inspect;
pub mod media;
pub mod paths;
pub mod payload;
pub mod potion;
pub mod pyenv;
pub mod search;
pub mod search_ann;
pub mod search_graph;
pub mod search_space;
pub mod search_text;
pub mod stats;
pub mod tone;
pub mod util;
pub mod validate;
