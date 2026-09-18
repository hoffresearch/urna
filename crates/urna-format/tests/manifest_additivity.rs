//! Manifest additivity guard (phase 0, task #03).
//!
//! The manifest is covered by file_hash and is JCS-canonical, so a NEW
//! required field (a bool on Capabilities, a non-optional manifest field)
//! is two breaks sold as one: old manifests fail to deserialize, and every
//! existing file's file_hash changes. This guards the additive contract:
//! every new field is `Option` with `skip_serializing_if`, so an unset
//! field serializes to NOTHING and existing files stay byte-identical; and
//! an unknown field from a newer writer survives through the flattened
//! `extra` map instead of erroring. (content_hash is over the canonical six
//! and never touches the manifest, so none of this can move a citation.)

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::manifest::{CapabilitiesExt, Manifest};

fn valid_manifest() -> Manifest {
    Manifest {
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: 1,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    }
}

/// Every additive-optional field is omitted from the canonical json when
/// unset, so adding the field never perturbs an existing file's bytes.
#[test]
fn unset_additive_fields_are_omitted() {
    let json = String::from_utf8(valid_manifest().to_canonical_json().unwrap()).unwrap();
    for key in [
        "title",
        "version",
        "created",
        "description",
        "authors",
        "license",
        "mrl_dim",
        "full_dim",
        "capabilities_ext",
    ] {
        assert!(
            !json.contains(&format!("\"{key}\"")),
            "unset additive field {key} must be omitted from canonical json, got: {json}"
        );
    }
}

/// The manifest round-trips byte-identically through canonical json: an
/// old manifest deserializes and re-serializes to the same bytes, so a
/// reader that reads then rewrites it does not change its file_hash.
#[test]
fn manifest_round_trips_byte_identical() {
    let m = valid_manifest();
    let bytes = m.to_canonical_json().unwrap();
    let back: Manifest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(m, back, "deserialize must reconstruct the manifest exactly");
    assert_eq!(
        back.to_canonical_json().unwrap(),
        bytes,
        "re-serialization must be byte-identical (file_hash stability)"
    );
}

/// A field a newer writer adds that this reader does not know must NOT
/// fail deserialization; it lands in `extra` and is preserved on rewrite.
/// This is what lets old readers open new files (additive within v1).
#[test]
fn unknown_future_field_survives_via_extra() {
    let bytes = valid_manifest().to_canonical_json().unwrap();
    let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    v.as_object_mut()
        .unwrap()
        .insert("storage_mode".into(), serde_json::json!("catalog"));
    let injected = serde_json::to_vec(&v).unwrap();

    let m: Manifest = serde_json::from_slice(&injected)
        .expect("unknown future field must not fail deserialization");
    assert_eq!(
        m.extra.get("storage_mode"),
        Some(&serde_json::json!("catalog")),
        "the unknown field must be preserved in extra so old readers round-trip it"
    );
}

/// The matryoshka disclosure fields (mrl_dim/full_dim) are additive: unset
/// they are omitted (a non-truncated file stays byte-identical to a v1
/// manifest), and a manifest WITH them set round-trips byte-identically so a
/// reader that reads then rewrites a truncated file does not move its
/// file_hash.
#[test]
fn mrl_fields_are_additive_and_round_trip() {
    // unset: omitted, byte-identical to a v1 manifest.
    let base = valid_manifest();
    assert!(base.mrl_dim.is_none() && base.full_dim.is_none());
    let base_json = String::from_utf8(base.to_canonical_json().unwrap()).unwrap();
    assert!(!base_json.contains("mrl_dim") && !base_json.contains("full_dim"));

    // set: appears, round-trips byte-identically.
    let mut m = valid_manifest();
    m.embedding_dim = 128;
    m.mrl_dim = Some(128);
    m.full_dim = Some(384);
    let bytes = m.to_canonical_json().unwrap();
    let s = String::from_utf8(bytes.clone()).unwrap();
    assert!(s.contains("\"mrl_dim\":128"));
    assert!(s.contains("\"full_dim\":384"));
    let back: Manifest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(m, back, "mrl fields must round-trip exactly");
    assert_eq!(
        back.to_canonical_json().unwrap(),
        bytes,
        "re-serialization must be byte-identical (file_hash stability)"
    );
    assert_ne!(
        bytes,
        base.to_canonical_json().unwrap(),
        "a truncated file legitimately differs from a full-dim manifest"
    );
}

/// Capabilities_ext is the additive home for future capability flags:
/// `None` (default) is omitted (byte-identical to a v1 manifest), and a set
/// flag appears, round-trips, and only ADDS bytes (file_hash moves, which is
/// expected when the file genuinely declares a new capability).
#[test]
fn capabilities_ext_is_additive() {
    let base = valid_manifest();
    assert!(base.capabilities_ext.is_none());
    let base_json = base.to_canonical_json().unwrap();
    assert!(
        !String::from_utf8(base_json.clone())
            .unwrap()
            .contains("capabilities_ext")
    );

    let mut with_ext = valid_manifest();
    with_ext.capabilities_ext = Some(CapabilitiesExt {
        graph_present: Some(true),
        ..Default::default()
    });
    let ext_json = with_ext.to_canonical_json().unwrap();
    let ext_str = String::from_utf8(ext_json.clone()).unwrap();
    assert!(ext_str.contains("\"capabilities_ext\":{\"graph_present\":true}"));
    assert_ne!(
        ext_json, base_json,
        "declaring a new capability changes file bytes (file_hash), as intended"
    );

    let back: Manifest = serde_json::from_slice(&ext_json).unwrap();
    assert_eq!(back, with_ext, "capabilities_ext must round-trip exactly");
    // an unset flag inside the ext struct is also omitted, not serialized as null.
    assert!(!ext_str.contains("null"));
}
