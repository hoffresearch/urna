//! `eval FILE.urna FIELDS POTION_DIR [N]`: type N playable cards' names
//! (every card when N is absent; an even stride over the ordinals otherwise)
//! one letter at a time and report where the card lands and how long each
//! keystroke took; then search every name and check it comes back first.
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let a: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if a.len() < 3 {
        eprintln!("usage: eval FILE.urna FIELDS POTION_DIR [N]");
        std::process::exit(2);
    }
    let p = match pasta_core::Pasta::open(&a[0], &a[1], &a[2]) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let all: Vec<(usize, String)> = p
        .cards()
        .iter()
        .filter(|c| c.playable())
        .map(|c| (c.ordinal, c.name.clone()))
        .collect();
    let want = a
        .get(3)
        .and_then(|n| n.to_str()?.parse::<usize>().ok())
        .unwrap_or(all.len())
        .max(1);
    let stride = (all.len() / want).max(1);
    let names: Vec<(usize, String)> = all.into_iter().step_by(stride).collect();
    let (mut full_top1, mut search_top1, mut keys_to_top5) = (0usize, 0usize, Vec::new());
    let mut lat = Vec::new();
    for (ord, name) in &names {
        let chars: Vec<char> = name.chars().collect();
        let mut found = None;
        for n in 1..=chars.len() {
            let prefix: String = chars[..n].iter().collect();
            let t = Instant::now();
            let s = p.complete(&prefix, 5);
            lat.push(t.elapsed().as_secs_f64() * 1000.0);
            if s.iter().any(|x| x.name == *name) {
                found = Some(n);
                break;
            }
        }
        keys_to_top5.push(found.unwrap_or(usize::MAX));
        if p.complete(name, 1).first().is_some_and(|x| x.name == *name) {
            full_top1 += 1;
        }
        let hits = p
            .search(name, &pasta_core::Filter::default(), 1)
            .expect("search");
        if hits
            .first()
            .is_some_and(|h| p.cards()[h.ordinal].name == *name || h.ordinal == *ord)
        {
            search_top1 += 1;
        }
    }
    lat.sort_by(f64::total_cmp);
    let n = names.len();
    let reached: Vec<usize> = keys_to_top5
        .iter()
        .copied()
        .filter(|k| *k != usize::MAX)
        .collect();
    let mut sorted = reached.clone();
    sorted.sort_unstable();
    println!("playable cards evaluated: {n} (stride {stride})");
    println!(
        "full name -> 1st suggestion: {full_top1} ({:.4})",
        full_top1 as f64 / n as f64
    );
    println!(
        "name search -> 1st hit: {search_top1} ({:.4})",
        search_top1 as f64 / n as f64
    );
    println!(
        "keystrokes until the card is in the top 5: median {} p90 {} never {}",
        sorted[sorted.len() / 2],
        sorted[sorted.len() * 9 / 10],
        n - reached.len()
    );
    println!(
        "keystroke latency over {} calls: p50 {:.2} ms p99 {:.2} ms max {:.2} ms",
        lat.len(),
        lat[lat.len() / 2],
        lat[lat.len() * 99 / 100],
        lat[lat.len() - 1]
    );
}
