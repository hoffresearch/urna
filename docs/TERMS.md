---
project: urna
audience: users and contributors
status: active
last-updated: 2026-10-06
domain: naming
---

# Terms

The abbreviations used in the repository's names, one meaning each. Names follow the equal-length rule of [CONTRIBUTING](CONTRIBUTING.md#naming): siblings of the same kind have the same length, and the `LEVELS` table of `tool/tasks/namecheck.py` says which length each folder's children take.

Three rules keep the lexicon useful:

1. One abbreviation, one meaning. `cmp` never means compile in one place and compare in another.
2. No synonym invented to fill characters. Reuse the project's vocabulary first: a word already in a name, or an abbreviation already listed here.
3. A new abbreviation enters this file in the same pull request that creates the name.

`tool/tasks/namecheck.py` reads the table below in CI: an abbreviation listed twice fails, and so does an entry that cites a name the tree does not have or that does not contain the abbreviation. Whole words (`bridge`, `engine`, `orchestra`) and the names of tools and products (`pypi`, `conda`, `nixos`, `av1`, `clip`) are not abbreviations and are not listed.

## Lexicon

| Abbreviation | Meaning | Names |
| --- | --- | --- |
| `adr` | architecture decision record | `docs/adr/` |
| `ai` | artificial intelligence | `aiops/` |
| `api` | application programming interface | `test_pythonapi.py` |
| `b8m` | base-8M, the size of the potion model (`minishlab/potion-base-8M`) | `potionb8m/` |
| `bench` | benchmark | `tool/bench/`, `BENCH.md`, `benchgate.py` |
| `cbinds` | C bindings, the C ABI the native apps link | `cbinds/` |
| `chan` | distribution channel | `chanprobe.py` |
| `choco` | Chocolatey, the Windows package manager | `pkgs/choco/` |
| `cli` | command-line interface | `clitui/`, `clidriver.py`, `test_clispaces.py` |
| `cmp` | compare | `recallcmp.py`, `vectorcmp.py` |
| `crf` | constant rate factor, the video encoder's quality knob | `crfpicker.py` |
| `dbs` | databases | `vectordbs.py` |
| `dec` | decode | `av1dec/`, `decframes.py` |
| `dev` | development | `.devops/` |
| `docs` | documentation | `docs/` |
| `emb` | embedding | `visionemb.py` |
| `enc` | encode | `encstills.py`, `encstream.py` |
| `eval` | evaluation | `imageeval.py` |
| `gop` | group of pictures, the video keyframe interval | `gopprober.py` |
| `img` | image | `docs/img/`, `imgcorpus.py`, `imgsearch.py` |
| `intro` | introduction | `demo/corpora/intro/` |
| `lexi` | lexical | `lexifloor.py` |
| `medic` | medical | `demo/corpora/medic/` |
| `ops` | operations | `.devops/`, `.secops/`, `aiops/` |
| `osint` | open-source intelligence | `demo/corpora/osint/` |
| `pkgs` | packages, the distribution channels | `pkgs/` |
| `pr` | pull request | `releasepr.sh` |
| `prep` | prepare | `buildprep.yml`, `wheelprep.py` |
| `py` | Python | `flaskpy/` |
| `qry` | query | `potionqry.py`, `presetqry.py` |
| `sec` | security | `.secops/` |
| `snaps` | snapshots | `test_clipsnaps.py` |
| `spec` | specification, the declarative build spec | `specs/`, `specbuild.py`, `specparse.py`, `specpaths.py`, `specrules.py` |
| `st` | sentence-transformers | `stbackend.py`, `stprocess.py` |
| `stat` | statistics | `imagestat.py` |
| `tab` | table, the potion static embedding table | `potiontab.py` |
| `tui` | terminal user interface | `clitui/` |
| `txt` | text | `searchtxt.py` |
| `ui` | user interface | `uibackend.py` |
| `vect` | vector | `vectcache.py` |
| `wingt` | winget, the Windows Package Manager | `pkgs/wingt/` |
