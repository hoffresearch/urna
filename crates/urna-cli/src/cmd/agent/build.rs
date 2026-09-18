//! `urna build --spec <file>` — launcher for the declarative build (RFC-0
//! N13: the build IS a python frontend; this verb resolves the interpreter
//! and the `urna_forge.py` tool, streams its output, and propagates the
//! exit code). The heavy lifting (spec validation, media, embedding, emit)
//! lives in python where torch/ffmpeg are; migration of spec validation to
//! rust is a registered future path, not this verb's job.

use anyhow::Result;
use std::path::PathBuf;
use std::process::Command as ProcCommand;

/// Resolve python/tools/urna_forge.py through the one shared script
/// ladder (repo layout, installed data dir, `<exe>/../share`).
fn forge_tool_path() -> PathBuf {
    super::super::embed_gate::installed_script_in("tools", "urna_forge.py")
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    spec: PathBuf,
    sample: Option<usize>,
    models: Option<String>,
    out_dir: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    resume: bool,
    rebuild_only: bool,
    dry_run: bool,
    allow_heavy: bool,
) -> Result<()> {
    let tool = forge_tool_path();
    if !tool.exists() {
        anyhow::bail!(
            "urna_forge.py not found ({}); run from the repo or install the forge payload",
            tool.display()
        );
    }
    let interpreter = super::super::pyenv::resolve_interpreter();
    let mut cmd = ProcCommand::new(interpreter);
    cmd.arg(&tool).arg("--spec").arg(&spec);
    if let Some(n) = sample {
        cmd.arg("--sample").arg(n.to_string());
    }
    if let Some(m) = &models {
        cmd.arg("--models").arg(m);
    }
    if let Some(d) = &out_dir {
        cmd.arg("--out-dir").arg(d);
    }
    if let Some(d) = &cache_dir {
        cmd.arg("--cache-dir").arg(d);
    }
    for (flag, on) in [
        ("--resume", resume),
        ("--rebuild-only", rebuild_only),
        ("--dry-run", dry_run),
        ("--allow-heavy", allow_heavy),
    ] {
        if on {
            cmd.arg(flag);
        }
    }
    // stream child stdout/stderr straight through; the build's progress and
    // typed errors are the python tool's contract, not re-parsed here.
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("failed to spawn {}: {}", tool.display(), e))?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}
