//! Regression for the mutation-fuzz findings in the BM25 codec: a posting
//! whose doc id is outside `doc_lengths` indexed out of bounds at query
//! time, and a hostile length field could wrap the cursor's `pos + n`
//! bounds check. Both must be typed rejections at decode, never panics.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]

use urna_runtime::bm25::Bm25Index;

fn valid_payload() -> Vec<u8> {
    let docs: Vec<String> = (0..4).map(|i| format!("alpha beta term{i}")).collect();
    Bm25Index::build(&docs, 1.2, 0.75).to_bytes()
}

#[test]
fn posting_doc_id_beyond_n_docs_is_rejected() {
    // v1 layout: version u32 | k1 f32 | b f32 | avgdl f32 | n_docs u32 |
    // n_terms u32 | doc_lengths[n_docs] u32 | terms...; shrink n_docs so an
    // existing posting points past the (now shorter) doc_lengths.
    let mut bytes = valid_payload();
    let version = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if version != 1 {
        // v2 packs postings; build the same corruption through decode of a
        // hand-written v1 payload instead.
        bytes = v1_payload_with_bad_doc();
    } else {
        bytes[16..20].copy_from_slice(&1u32.to_le_bytes());
    }
    let res = Bm25Index::from_bytes(&bytes);
    assert!(res.is_err(), "posting doc >= n_docs must be a typed error");
    if let Ok(idx) = res {
        let _ = idx.search("alpha", 3);
    }
}

fn v1_payload_with_bad_doc() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&1u32.to_le_bytes()); // version
    b.extend_from_slice(&1.2f32.to_le_bytes());
    b.extend_from_slice(&0.75f32.to_le_bytes());
    b.extend_from_slice(&3.0f32.to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes()); // n_docs = 1
    b.extend_from_slice(&1u32.to_le_bytes()); // n_terms = 1
    b.extend_from_slice(&3u32.to_le_bytes()); // doc_lengths[0]
    b.extend_from_slice(&5u32.to_le_bytes()); // term len
    b.extend_from_slice(b"alpha");
    b.extend_from_slice(&1u32.to_le_bytes()); // df
    b.extend_from_slice(&7u32.to_le_bytes()); // doc = 7 (>= n_docs)
    b.extend_from_slice(&1u32.to_le_bytes()); // tf
    b
}

#[test]
fn non_finite_parameters_are_rejected() {
    let mut bytes = valid_payload();
    bytes[12..16].copy_from_slice(&f32::NAN.to_le_bytes()); // avgdl
    assert!(Bm25Index::from_bytes(&bytes).is_err());
}

#[test]
fn huge_length_field_is_a_typed_error_not_a_wrapped_bounds_check() {
    // a term length of u32::MAX at the cursor: `pos + n` would wrap on a
    // 32-bit target and the slice would panic; the cursor compares against
    // the remaining bytes instead.
    let mut b = v1_payload_with_bad_doc();
    let term_len_at = 4 * 7;
    b[term_len_at..term_len_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Bm25Index::from_bytes(&b).is_err());
}
