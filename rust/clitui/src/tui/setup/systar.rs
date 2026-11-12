//! Archives through the system `tar`, the way downloads go through the
//! system `curl` (`net.rs`): the binary links no decompressor and no
//! archive reader. Every supported system has one: GNU tar or bsdtar on
//! linux, bsdtar on macOS, and `System32\tar.exe` (bsdtar) on windows 10
//! and later, which also reads the `.zip` uv ships for windows.
//!
//! An archive is listed before anything is unpacked, twice: `tar -tf` for
//! the member names and `tar -tvf` for their kinds, line for line. A
//! member whose name is absolute, carries a drive or a `..`, or that is not
//! a plain file or a folder (a symlink, a hard link, a device) refuses the
//! whole archive, so nothing a hostile archive names can land outside the
//! folder it is unpacked into. Both tars detect gzip by its magic bytes, so
//! the same calls read a `.tar.gz` and a plain `.tar`.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{Context, Result, bail};

/// What a member is, from the first column of `tar -tv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    File,
    Dir,
    /// a symlink, hard link, device or fifo: never unpacked.
    Other,
}

/// One member of an archive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub name: String,
    pub kind: Kind,
}

/// The tar setup runs. On windows the one in System32, never a GNU tar a
/// Git install put first on PATH (it reads `C:\...` as a remote host);
/// `tar` from PATH elsewhere.
pub fn program() -> PathBuf {
    #[cfg(windows)]
    {
        if let Some(root) = std::env::var_os("SystemRoot") {
            let exe = PathBuf::from(root).join("System32").join("tar.exe");
            if exe.is_file() {
                return exe;
            }
        }
    }
    PathBuf::from("tar")
}

fn run(args: &[&OsStr]) -> Result<String> {
    let out = Command::new(program())
        .args(args)
        .output()
        .context("spawn tar (setup unpacks through the system tar)")?;
    if !out.status.success() {
        bail!(
            "tar failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(|l| l.trim_end_matches('\r'))
        .filter(|l| !l.is_empty())
        .collect()
}

/// The members of `archive`, in archive order.
pub fn list(archive: &Path) -> Result<Vec<Member>> {
    let names = run(&["-tf".as_ref(), archive.as_os_str()])?;
    let long = run(&["-tvf".as_ref(), archive.as_os_str()])?;
    let (names, long) = (lines(&names), lines(&long));
    if names.len() != long.len() {
        // a name with a line break in it: no honest archive has one.
        bail!("tar lists the archive's members inconsistently; refusing it");
    }
    Ok(names
        .iter()
        .zip(&long)
        .map(|(name, row)| Member {
            name: (*name).to_string(),
            kind: match row.as_bytes().first() {
                Some(b'-') => Kind::File,
                Some(b'd') => Kind::Dir,
                _ => Kind::Other,
            },
        })
        .collect())
}

/// True when `name` stays inside the folder it is unpacked into: relative,
/// no drive, no `..` component, and not read as an option by tar.
pub fn stays_inside(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with(['/', '\\', '-'])
        && !name.contains(':')
        && name.split(['/', '\\']).all(|part| part != "..")
}

/// Refuses `members` unless every one is a plain file or folder that stays
/// inside the unpack folder; `what` names the archive in the error.
pub fn check(members: &[Member], what: &str) -> Result<()> {
    for m in members {
        if !stays_inside(&m.name) {
            bail!("{what} entry escapes the install dir: {}", m.name);
        }
        if m.kind == Kind::Other {
            bail!(
                "{what} entry {} is not a file or a folder; refusing to install it",
                m.name
            );
        }
    }
    Ok(())
}

/// Unpacks `archive` (already listed and checked) into `dir`, or only the
/// member named `only`. `-o` keeps the files owned by the user running
/// setup, not by whoever the archive names, in GNU tar and bsdtar alike.
pub fn extract(archive: &Path, dir: &Path, only: Option<&str>) -> Result<()> {
    let mut args: Vec<&OsStr> = vec![
        "-xof".as_ref(),
        archive.as_os_str(),
        "-C".as_ref(),
        dir.as_os_str(),
    ];
    if let Some(member) = only {
        args.push(member.as_ref());
    }
    run(&args).map(|_| ())
}

/// A plain ustar archive of `entries` (name, mode, bytes), for the tests:
/// the names are written raw, so a test can model a hostile archive the
/// system tar would refuse to create.
#[cfg(test)]
pub fn ustar(entries: &[(&str, u32, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, mode, data) in entries {
        let mut h = [0u8; 512];
        h[..name.len()].copy_from_slice(name.as_bytes());
        h[100..107].copy_from_slice(format!("{mode:07o}").as_bytes());
        h[108..115].copy_from_slice(b"0000000");
        h[116..123].copy_from_slice(b"0000000");
        h[124..135].copy_from_slice(format!("{:011o}", data.len()).as_bytes());
        h[136..147].copy_from_slice(b"00000000000");
        h[156] = b'0';
        h[257..263].copy_from_slice(b"ustar\0");
        h[263..265].copy_from_slice(b"00");
        // the checksum is computed with its own field read as spaces.
        h[148..156].copy_from_slice(b"        ");
        let sum: u32 = h.iter().map(|&b| b as u32).sum();
        h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(data);
        out.resize(out.len().div_ceil(512) * 512, 0);
    }
    out.resize(out.len() + 1024, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_that_leave_the_folder_are_refused() {
        for ok in [
            "urna/VERSION",
            "urna/python/urna/x.py",
            "./urna/a",
            "uv.exe",
        ] {
            assert!(stays_inside(ok), "{ok}");
        }
        for bad in [
            "",
            "/etc/passwd",
            "\\x",
            "../x",
            "urna/../../x",
            "urna\\..\\x",
            "C:x",
            "-x",
        ] {
            assert!(!stays_inside(bad), "{bad}");
        }
    }

    #[test]
    fn a_link_or_an_escape_refuses_the_whole_archive() {
        let file = |name: &str| Member {
            name: name.into(),
            kind: Kind::File,
        };
        assert!(check(&[file("urna/a"), file("urna/b")], "t").is_ok());
        let link = Member {
            name: "urna/l".into(),
            kind: Kind::Other,
        };
        let err = check(&[file("urna/a"), link], "t").unwrap_err();
        assert!(err.to_string().contains("not a file or a folder"), "{err}");
        let err = check(&[file("../x")], "t").unwrap_err();
        assert!(err.to_string().contains("escapes"), "{err}");
    }

    #[test]
    fn lists_and_extracts_through_the_system_tar() {
        let d = std::env::temp_dir().join(format!("urna_systar_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("out")).unwrap();
        let tar = d.join("a.tar");
        std::fs::write(
            &tar,
            ustar(&[("urna/VERSION", 0o644, b"1\n"), ("urna/b", 0o644, b"bb")]),
        )
        .unwrap();
        let members = list(&tar).unwrap();
        assert_eq!(
            members,
            [
                Member {
                    name: "urna/VERSION".into(),
                    kind: Kind::File
                },
                Member {
                    name: "urna/b".into(),
                    kind: Kind::File
                },
            ]
        );
        check(&members, "test").unwrap();
        extract(&tar, &d.join("out"), Some("urna/b")).unwrap();
        assert_eq!(std::fs::read(d.join("out/urna/b")).unwrap(), b"bb");
        assert!(!d.join("out/urna/VERSION").exists());
        extract(&tar, &d.join("out"), None).unwrap();
        assert_eq!(std::fs::read(d.join("out/urna/VERSION")).unwrap(), b"1\n");
        std::fs::remove_dir_all(&d).unwrap();
    }
}
