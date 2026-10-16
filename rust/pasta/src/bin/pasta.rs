//! `pasta FILE.urna FIELDS POTION_DIR`: opens once, then reads commands
//! from stdin, one per line, and prints how long each took.
//!
//! ```text
//! c light                                type-ahead
//! s draw three cards                     search
//! s counter spell | format=pauper id=UB  search with filters
//! s | format=commander type=dragon       browse by popularity
//! card 17823                             the card's fields
//! img 17823 out.avif                     write the card's image
//! ```
//!
//! Filters after `|`: `format=`, `id=` (color identity within, `WUBRG`),
//! `cmc<=`, `type=`, `all` (keep art-series, tokens, emblems, Alchemy).

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use pasta_core::{Filter, Pasta};

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: pasta FILE.urna FIELDS POTION_DIR");
        std::process::exit(2);
    }
    let paths: Vec<PathBuf> = args[1..].iter().map(PathBuf::from).collect();
    let pasta = match Pasta::open(&paths[0], &paths[1], &paths[2]) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let t = &pasta.timings;
    println!(
        "open {:.0} ms · fields {:.0} ms ({} cards) · index {:.0} ms · potion {:.0} ms",
        t.open_ms,
        t.fields_ms,
        pasta.cards().len(),
        t.index_ms,
        t.potion_ms
    );

    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        let (cmd, rest) = line.split_once(' ').unwrap_or((line.as_str(), ""));
        let t0 = Instant::now();
        match cmd {
            "c" => {
                let s = pasta.complete(rest, 8);
                println!("c {rest:?} · {:.2} ms", ms(t0));
                for x in s {
                    let pt = x.pt_name.map(|p| format!(" ({p})")).unwrap_or_default();
                    println!(
                        "  {:>5}  {}{}{}",
                        x.ordinal,
                        x.name,
                        pt,
                        if x.via_pt { "  ← pt" } else { "" }
                    );
                }
            }
            "s" => {
                let (q, spec) = rest.split_once('|').unwrap_or((rest, ""));
                match pasta.search(q, &Filter::parse(spec), 6) {
                    Ok(hits) => {
                        println!("s {rest:?} · {:.2} ms", ms(t0));
                        for h in hits {
                            let c = &pasta.cards()[h.ordinal];
                            println!(
                                "  {:>5}  {:.3} {:<6} {} · {} · {} · edhrec {}",
                                h.ordinal,
                                h.score,
                                h.via,
                                c.name,
                                c.mana_cost,
                                c.type_line,
                                c.edhrec_rank.map_or("-".into(), |r| r.to_string())
                            );
                        }
                    }
                    Err(e) => println!("s error: {e}"),
                }
            }
            "card" => match rest
                .trim()
                .parse::<usize>()
                .ok()
                .and_then(|o| pasta.card(o))
            {
                Some(c) => println!("{c:#?}"),
                None => println!("card: no such ordinal"),
            },
            "img" => {
                let mut it = rest.split_whitespace();
                let ord = it.next().and_then(|o| o.parse::<usize>().ok());
                let path = it.next().unwrap_or("card.avif");
                match ord.and_then(|o| pasta.image(o)) {
                    Some(bytes) => match std::fs::write(path, bytes) {
                        Ok(()) => {
                            println!("img {} bytes -> {path} · {:.2} ms", bytes.len(), ms(t0))
                        }
                        Err(e) => println!("img error: {e}"),
                    },
                    None => println!("img: no per-card image in this file"),
                }
            }
            "" => {}
            other => println!("unknown command {other:?}"),
        }
        let _ = std::io::stdout().flush();
    }
}
