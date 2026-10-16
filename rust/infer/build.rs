//! Generates the Unicode tables the tokenizer's normalizer needs from the
//! vendored `ucd/UnicodeData.txt` (version and hash in `ucd/UPSTREAM`):
//! the "other" categories (Cc, Cf, Co) that BERT's clean_text drops, the
//! nonspacing marks (Mn) that strip_accents removes, every punctuation
//! category (P*) the pre-tokenizer isolates, the canonical combining
//! classes, and the canonical decompositions fully expanded, for NFD.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

/// a build script stops on a malformed database with the reason named.
fn must<T, E: std::fmt::Display>(r: Result<T, E>, what: &str) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{what}: {e}"),
    }
}

fn ranges(points: &[u32]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for &p in points {
        match out.last_mut() {
            Some((_, end)) if *end + 1 == p => *end = p,
            _ => out.push((p, p)),
        }
    }
    out
}

fn expand(code: u32, raw: &BTreeMap<u32, Vec<u32>>, out: &mut Vec<u32>) {
    match raw.get(&code) {
        Some(parts) => {
            for &p in parts {
                expand(p, raw, out);
            }
        }
        None => out.push(code),
    }
}

fn main() {
    let ucd = Path::new(env!("CARGO_MANIFEST_DIR")).join("ucd/UnicodeData.txt");
    println!("cargo:rerun-if-changed={}", ucd.display());
    println!("cargo:rerun-if-changed=build.rs");
    let text = must(
        std::fs::read_to_string(&ucd),
        "ucd/UnicodeData.txt is vendored",
    );

    let mut other = Vec::new();
    let mut mn = Vec::new();
    let mut punct = Vec::new();
    let mut ccc: Vec<(u32, u8)> = Vec::new();
    let mut raw_decomp: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut range_start: Option<u32> = None;
    for line in text.lines() {
        let f: Vec<&str> = line.split(';').collect();
        if f.len() < 6 {
            continue;
        }
        let code = must(u32::from_str_radix(f[0], 16), "hex code");
        let name = f[1];
        let cat = f[2];
        if name.ends_with("First>") {
            range_start = Some(code);
            continue;
        }
        let codes: Vec<u32> = match (name.ends_with("Last>"), range_start.take()) {
            (true, Some(start)) => (start..=code).collect(),
            _ => vec![code],
        };
        for c in codes {
            match cat {
                "Cc" | "Cf" | "Co" => other.push(c),
                "Mn" => mn.push(c),
                "Pc" | "Pd" | "Pe" | "Pf" | "Pi" | "Po" | "Ps" => punct.push(c),
                _ => {}
            }
        }
        let class: u8 = must(f[3].parse(), "combining class");
        if class != 0 {
            ccc.push((code, class));
        }
        let decomposition = f[5];
        if !decomposition.is_empty() && !decomposition.starts_with('<') {
            let parts: Vec<u32> = decomposition
                .split_whitespace()
                .map(|h| must(u32::from_str_radix(h, 16), "hex decomposition"))
                .collect();
            raw_decomp.insert(code, parts);
        }
    }
    let mut decomp: Vec<(u32, Vec<u32>)> = Vec::new();
    for &code in raw_decomp.keys() {
        let mut full = Vec::new();
        expand(code, &raw_decomp, &mut full);
        decomp.push((code, full));
    }

    let mut src = String::new();
    let emit = |src: &mut String, name: &str, r: &[(u32, u32)]| {
        let _ = writeln!(src, "pub static {name}: &[(u32, u32)] = &[");
        for (a, b) in r {
            let _ = writeln!(src, "    ({a:#x}, {b:#x}),");
        }
        let _ = writeln!(src, "];");
    };
    emit(&mut src, "OTHER", &ranges(&other));
    emit(&mut src, "MARK_NONSPACING", &ranges(&mn));
    emit(&mut src, "PUNCTUATION", &ranges(&punct));
    let _ = writeln!(src, "pub static COMBINING_CLASS: &[(u32, u8)] = &[");
    for (c, k) in &ccc {
        let _ = writeln!(src, "    ({c:#x}, {k}),");
    }
    let _ = writeln!(src, "];");
    let _ = writeln!(src, "pub static DECOMPOSITION: &[(u32, &[u32])] = &[");
    for (c, parts) in &decomp {
        let list: Vec<String> = parts.iter().map(|p| format!("{p:#x}")).collect();
        let _ = writeln!(src, "    ({c:#x}, &[{}]),", list.join(", "));
    }
    let _ = writeln!(src, "];");
    let out = Path::new(&must(std::env::var("OUT_DIR"), "OUT_DIR")).join("ucd_tables.rs");
    must(std::fs::write(out, src), "write tables");
}
