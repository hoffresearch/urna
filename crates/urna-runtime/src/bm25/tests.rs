use super::*;
use index::{Bm25Index, DEFAULT_B, DEFAULT_K1};

#[test]
fn tokenizer_handles_pt_br() {
    let toks = tokenize("Olá mundo. ESTA é uma frase com acentuação.");
    // "olá", "mundo", "esta", "uma", "frase", "com", "acentuação"
    // (single-char tokens dropped: "é")
    assert!(toks.contains(&"olá".to_string()));
    assert!(toks.contains(&"acentuação".to_string()));
    assert!(!toks.contains(&"é".to_string()));
}

#[test]
fn bm25_finds_relevant_doc() {
    let docs = vec![
        "vacina contra covid no brasil".to_string(),
        "futebol jogos resultados".to_string(),
        "vacinação contra a gripe".to_string(),
        "meteorologia previsão tempo".to_string(),
    ];
    let idx = Bm25Index::build(&docs, DEFAULT_K1, DEFAULT_B);
    let hits = idx.search("vacina covid", 4);
    assert_eq!(hits[0].0, 0, "doc 0 should rank first for 'vacina covid'");
}

#[test]
fn bm25_serialize_roundtrip() {
    let docs = vec![
        "alpha beta gamma".to_string(),
        "alpha delta".to_string(),
        "gamma omega".to_string(),
    ];
    let idx = Bm25Index::build(&docs, DEFAULT_K1, DEFAULT_B);
    let bytes = idx.to_bytes();
    let back = Bm25Index::from_bytes(&bytes).unwrap();
    let a = idx.search("alpha", 3);
    let b = back.search("alpha", 3);
    assert_eq!(a, b);
}

#[test]
fn rrf_union_combines_sources() {
    let a = vec![10, 20, 30];
    let b = vec![20, 40, 50];
    let union = rrf_union(&a, &b);
    // 20 appears in both → ranks first.
    assert_eq!(union[0], 20);
    // All ids appear exactly once.
    let mut s = union.clone();
    s.sort();
    s.dedup();
    assert_eq!(s.len(), union.len());
}
