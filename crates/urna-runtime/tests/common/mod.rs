//! Shared helpers for integration tests (each test binary compiles what it
//! uses; the `dead_code` allow keeps unused helpers quiet per binary).
#![allow(dead_code)]
pub mod mutation;
