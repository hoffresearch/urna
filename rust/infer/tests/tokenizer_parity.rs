//! Our WordPiece tokenizer against the reference library on the potion's
//! own `tokenizer.json`: a fixed set that walks every rule, and every
//! card name of the corpus when `URNA_CARDS_CORPUS` names one. The
//! reference crate is a dev-dependency only; it never enters the build.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]

use std::path::PathBuf;

fn potion_tokenizer() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../bridge/python/urna/model/potionb8m/tokenizer.json")
}

fn reference() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(potion_tokenizer()).expect("reference tokenizer loads")
}

fn ours() -> urna_infer::Tokenizer {
    urna_infer::Tokenizer::from_file(&potion_tokenizer()).expect("our tokenizer loads")
}

fn same(ours: &urna_infer::Tokenizer, theirs: &tokenizers::Tokenizer, text: &str) -> bool {
    let a = ours.encode(text);
    let b = theirs
        .encode(text, false)
        .expect("reference encodes")
        .get_ids()
        .to_vec();
    if a != b {
        eprintln!("mismatch on {text:?}\n  ours   {a:?}\n  theirs {b:?}");
    }
    a == b
}

#[test]
fn the_fixed_set_matches_the_reference_id_for_id() {
    let long_word = "a".repeat(120);
    let texts = [
        "Lightning Bolt deals 3 damage to any target.",
        "Relâmpago: cause 3 pontos de dano a qualquer alvo. Ação ou não ação, é isso.",
        "",
        "   ",
        &long_word,
        "😀 smile 日本語のテキスト 한국어 中文",
        "ʘ‿ʘ ǅǈǋ ﬁ ﬀ ß",
        "hello,world!!!don't(stop)[UNK]here;[SEP]ok",
        "tab\tand\nnewline\r\nand\u{0}null\u{200B}zero",
        "Ustrój państwa – Zażółć gęślą jaźń. Привет, мир! Ωμέγα",
        "e-mail re--run foo_bar $100 50% a+b=c",
        "naïve café résumé coöperate",
    ];
    let (mine, theirs) = (ours(), reference());
    let ok = texts.iter().filter(|t| same(&mine, &theirs, t)).count();
    assert_eq!(ok, texts.len(), "{} of {} texts matched", ok, texts.len());
}

#[test]
fn every_card_name_of_the_corpus_matches_the_reference() {
    let Some(corpus) = std::env::var_os("URNA_CARDS_CORPUS") else {
        eprintln!("URNA_CARDS_CORPUS unset: skipping the corpus parity");
        return;
    };
    let file =
        urna_engine::MmapUrnaFile::open(std::path::Path::new(&corpus)).expect("corpus opens");
    let texts = file.canonical_texts().expect("texts");
    let (mine, theirs) = (ours(), reference());
    let mut total = 0usize;
    let mut matched = 0usize;
    for text in &texts {
        for line in text.lines().take(2) {
            total += 1;
            if same(&mine, &theirs, line) {
                matched += 1;
            }
        }
    }
    eprintln!("corpus parity: {matched} of {total} lines matched");
    assert_eq!(matched, total);
}
