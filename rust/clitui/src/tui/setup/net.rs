//! Downloads through the system `curl`: the binary links no network stack,
//! so the only socket `urna setup` ever opens belongs to a curl child
//! process, the same tool `installer.sh` uses. bytes are hashed while they
//! stream, so the sha256 check costs no second read.
//!
//! `URNA_RELEASE_BASE` points at another artifact server (a mirror, or a
//! `file://` directory in tests), the same override the one-liners take.

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

pub const PAYLOAD: &str = "urna-embedder-payload.tar.gz";
const REPO: &str = "hoffresearch/urna";

/// Where the release artifacts for `version` live.
pub fn release_base(version: &str) -> String {
    match std::env::var("URNA_RELEASE_BASE") {
        Ok(b) if !b.is_empty() => b.trim_end_matches('/').to_string(),
        _ => {
            let v = version.trim_start_matches('v');
            format!("https://github.com/{REPO}/releases/download/v{v}")
        }
    }
}

fn curl(curl: &Path) -> Command {
    let mut c = Command::new(curl);
    // https only (file:// for local mirrors and tests), and never downgrade
    // on a redirect; --fail turns an http error into a non-zero exit.
    c.args([
        "--silent",
        "--show-error",
        "--fail",
        "--location",
        "--proto",
        "=https,file",
        "--proto-redir",
        "=https",
        "--retry",
        "2",
        "--connect-timeout",
        "20",
    ]);
    c
}

/// A small text artifact (the `.sha256` file).
pub fn fetch_text(curl_bin: &Path, url: &str) -> Result<String> {
    let out = curl(curl_bin).arg(url).output().context("spawn curl")?;
    if !out.status.success() {
        bail!("{}", curl_error(url, &String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// curl's stderr in words: a 404 on a release url means the version was
/// never published (a dev build ahead of its tag), not a network fault.
pub fn curl_error(url: &str, stderr: &str) -> String {
    let err = stderr.trim();
    if err.contains("404") && url.contains("/releases/download/") {
        let tag = url
            .split("/releases/download/")
            .nth(1)
            .and_then(|r| r.split('/').next())
            .unwrap_or("?");
        format!(
            "release {tag} has no {} on github (404): this build is ahead of its tag; pass --version with a published one",
            url.rsplit('/').next().unwrap_or(url)
        )
    } else {
        format!("{url}: {err}")
    }
}

/// The size of `url` from a HEAD request, when the server says it.
pub fn content_length(curl_bin: &Path, url: &str) -> Option<u64> {
    let out = curl(curl_bin).arg("--head").arg(url).output().ok()?;
    parse_content_length(&String::from_utf8_lossy(&out.stdout))
}

/// The last `content-length` in a (possibly redirected) header dump.
pub fn parse_content_length(headers: &str) -> Option<u64> {
    headers
        .lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse().ok())?
        })
        .next_back()
}

/// The hex digest out of a `sha256sum`-style line (`<hex> *<name>`).
pub fn parse_sha256(text: &str) -> Result<String> {
    let hex = text
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("checksum file does not start with a sha256 hex digest");
    }
    Ok(hex)
}

/// Streams `url` into `dest`, calling `progress(bytes_so_far)` per chunk;
/// returns the lowercase sha256 of what was written.
pub fn download(
    curl_bin: &Path,
    url: &str,
    dest: &Path,
    mut progress: impl FnMut(u64),
) -> Result<String> {
    let mut child = curl(curl_bin)
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn curl")?;
    let mut src = child.stdout.take().context("curl stdout")?;
    let mut file =
        std::fs::File::create(dest).with_context(|| format!("create {}", dest.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = src.read(&mut buf).context("read from curl")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n]).context("write payload")?;
        total += n as u64;
        progress(total);
    }
    file.flush()?;
    let mut err = String::new();
    if let Some(mut e) = child.stderr.take() {
        let _ = e.read_to_string(&mut err);
    }
    if !child.wait()?.success() {
        bail!("{}", curl_error(url, &err));
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_length_takes_the_final_hop() {
        let h = "HTTP/2 302\r\ncontent-length: 0\r\nlocation: x\r\n\r\nHTTP/2 200\r\nContent-Length: 31337\r\n";
        assert_eq!(parse_content_length(h), Some(31337));
        assert_eq!(parse_content_length("HTTP/2 200\r\n"), None);
    }

    #[test]
    fn a_404_on_a_release_names_the_missing_tag() {
        let url = "https://github.com/hoffresearch/urna/releases/download/v9.9.9/urna-embedder-payload.tar.gz";
        let msg = curl_error(url, "curl: (22) The requested URL returned error: 404");
        assert!(msg.contains("v9.9.9") && msg.contains("--version"), "{msg}");
        assert!(curl_error("file:///x", "curl: (37) no file").starts_with("file:///x: curl: (37)"));
    }

    #[test]
    fn sha256_line_is_parsed_or_rejected() {
        let hex = "ab".repeat(32);
        assert_eq!(parse_sha256(&format!("{hex} *{PAYLOAD}\n")).unwrap(), hex);
        assert!(parse_sha256("not-a-digest file").is_err());
        assert!(parse_sha256("").is_err());
    }

    #[test]
    fn release_base_defaults_to_the_tagged_github_release() {
        if std::env::var_os("URNA_RELEASE_BASE").is_none() {
            let b = release_base("0.5.0");
            assert_eq!(
                b,
                "https://github.com/hoffresearch/urna/releases/download/v0.5.0"
            );
            assert_eq!(release_base("v0.5.0"), b);
        }
    }

    #[test]
    fn download_streams_a_file_url_and_hashes_it() {
        let Some(curl_bin) = super::super::scan::which("curl") else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("urna_net_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.bin");
        std::fs::write(&src, b"urna payload bytes").unwrap();
        let s = src.display().to_string().replace('\\', "/");
        let url = if s.starts_with('/') {
            format!("file://{s}")
        } else {
            format!("file:///{s}")
        };
        let mut seen = 0;
        let got = download(&curl_bin, &url, &dir.join("dst.bin"), |n| seen = n).unwrap();
        assert_eq!(got, hex::encode(Sha256::digest(b"urna payload bytes")));
        assert_eq!(seen, 18);
        assert!(
            download(
                &curl_bin,
                "file:///urna/no/such/file",
                &dir.join("x"),
                |_| {}
            )
            .is_err()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
