//! Shared helpers for the integration tests of the `urna` binary crate,
//! which cannot import its own modules.

use std::path::Path;

/// The files of a complete payload, read from `cmd::payload::REQUIRED` in
/// the source, so a test that lays a payload down can never drift from the
/// list setup and the scan enforce (two hand-kept copies had drifted).
pub fn payload_required() -> Vec<String> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cmd/payload.rs");
    let text = std::fs::read_to_string(&src).expect("read src/cmd/payload.rs");
    let start = text
        .find("pub const REQUIRED")
        .expect("payload.rs declares REQUIRED");
    let body = &text[start..];
    let body = &body[body.find('[').expect("REQUIRED is an array") + 1..];
    let body = &body[body.find('[').expect("REQUIRED has a value") + 1..];
    let body = &body[..body.find("];").expect("REQUIRED ends with ];")];
    let files: Vec<String> = body
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_owned)
        .collect();
    assert!(
        files.iter().any(|f| f == "VERSION"),
        "REQUIRED parsed from payload.rs lacks VERSION: {files:?}"
    );
    files
}
