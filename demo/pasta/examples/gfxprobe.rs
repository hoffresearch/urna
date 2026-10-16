//! Ask the terminal what it can draw, the way the TUIs do, and write the
//! answer to the file given: run it inside the terminal to check.
fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/gfxprobe.txt".into());
    let line = match ratatui_image::picker::Picker::from_query_stdio() {
        Ok(p) => format!(
            "protocol={:?} font={:?} term={} term_program={}",
            p.protocol_type(),
            p.font_size(),
            std::env::var("TERM").unwrap_or_default(),
            std::env::var("TERM_PROGRAM").unwrap_or_default()
        ),
        Err(e) => format!("error={e}"),
    };
    let _ = std::fs::write(&out, line);
}
