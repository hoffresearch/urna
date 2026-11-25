# fullsuite.sh - the python suites of the full tier of tool/tasks/fullcheck.sh.
#
# sourced by fullcheck.sh, which defines step, ok and $PY before it and
# runs python_tests as its `python-tests` step. one `step`, one suite: the
# comment above each says what the suite holds. the push tier runs the
# suites gatecheck.yml's python job runs (python_job, in
# tool/tasks/tiersteps.sh); this is every suite, on the release build.

# shellcheck shell=bash

python_tests() {
  # the step cache of this script: a key moves with every input, never with
  # an ignored file, and a hit needs the outputs it recorded.
  step "python tool/tests/test_stepcache.py"
  "$PY" tool/tests/test_stepcache.py
  ok "stepcache (10 cases)"

  # the corpus of step 4: fetched only with consent, measured only if pinned.
  step "python tool/tests/test_benchdata.py"
  "$PY" tool/tests/test_benchdata.py
  ok "benchdata (7 cases)"

  # the atlas summary is the current state in at most three sentences; the
  # history is in docs/CHANGELOG.
  step "python tool/tests/test_atlasdocs.py"
  "$PY" tool/tests/test_atlasdocs.py
  ok "atlasdocs (4 cases)"

  # rust/ingest inherits nothing from the root workspace: its lint tables
  # and rust-version are a copy, held to the root's here.
  step "python tool/tests/test_lintmatch.py"
  "$PY" tool/tests/test_lintmatch.py
  ok "lintmatch (4 cases)"

  step "python tool/tests/test_pythonapi.py"
  "$PY" tool/tests/test_pythonapi.py
  ok "pythonapi"

  step "python tool/tests/test_ingestion.py"
  "$PY" tool/tests/test_ingestion.py
  ok "ingestion"

  step "python tool/tests/test_hashguard.py"
  "$PY" tool/tests/test_hashguard.py
  ok "hashguard: search-text model_hash gate (7 cases)"

  # builds its own dataset, so it needs no demo corpus; the compressed cases
  # skip themselves when ffmpeg/libsvtav1 is absent.
  step "python tool/tests/test_imagepipe.py"
  "$PY" tool/tests/test_imagepipe.py
  ok "imagepipe: image corpus pipeline (43 cases)"

  # declarative builds: spec validation, fake-preset e2e, triad cache, dedup,
  # output modes, L3 rebuild. no heavy ML deps; media legs skip without ffmpeg.
  step "python tool/tests/test_specrules.py"
  "$PY" tool/tests/test_specrules.py
  step "python tool/tests/test_chunkcols.py"
  "$PY" tool/tests/test_chunkcols.py
  ok "specrules: build spec + pipeline"

  # dual quality gate + jxl round-trip; skips cleanly without ssimulacra2/cjxl.
  step "python tool/tests/test_mediagate.py"
  "$PY" tool/tests/test_mediagate.py
  ok "mediagate: quality gate + jxl"

  step "python tool/tests/test_clispaces.py"
  "$PY" tool/tests/test_clispaces.py
  ok "clispaces: cli space verbs (8 cases)"

  step "python tool/tests/test_askrouter.py"
  "$PY" tool/tests/test_askrouter.py
  ok "askrouter: query embedder routing (10 cases; 7 to 10 need sentence-transformers, URNA_ST_PYTHON)"

  # the release payload, staged and run from outside the checkout: both query
  # embedders answer, the registry route names its missing deps, a half tree
  # does not stage.
  step "python tool/tests/test_embedpack.py"
  "$PY" tool/tests/test_embedpack.py
  ok "embedpack: embedder payload (3 cases)"

  # the st_multimodal model_hash of a synthetic snapshot is the value the code
  # had before the move to the urna package: the fingerprint's key names are
  # data hashed into model_hash. needs numpy only; --models none keeps the
  # model cases out, so the gate never loads a model (#488).
  step "python tool/tests/test_stbackend.py --models none"
  "$PY" tool/tests/test_stbackend.py --models none
  ok "stbackend: pinned model_hash"

  # importing the search-text embedder, the model registry and the image path
  # forces the hub offline unless URNA_ALLOW_DOWNLOAD=1.
  step "python tool/tests/test_nonetwork.py"
  "$PY" tool/tests/test_nonetwork.py
  ok "nonetwork: offline by default and the opt-in (6 cases)"

  # the model catalog setup offers is the registry's validated presets, with a
  # reason for every preset it leaves out.
  step "python tool/tests/test_catalogue.py"
  "$PY" tool/tests/test_catalogue.py
  ok "catalogue: model catalog (5 cases)"

  # the uv setup downloads and the python range it builds on are read from
  # their upstreams into upstreams.json, never written by hand (stubbed here).
  step "python tool/tests/test_upstreams.py"
  "$PY" tool/tests/test_upstreams.py
  ok "upstreams: uv, the python range and the sample (7 cases)"

  # the preset job: an unknown --only name is a usage error, a measured 0 is
  # the probe failing, and an mps that compiles no kernel (github's mac
  # runner) runs every preset on the cpu (subprocess stubbed, no urna).
  step "python tool/tests/test_presetjob.py"
  "$PY" tool/tests/test_presetjob.py
  ok "presetjob: preset job (4 cases)"

  # the model fetch: confirmed downloads only, the pinned files only, kept
  # only when the fingerprint is the catalog's (an 18 MB hub model; the
  # download cases skip by name when huggingface.co does not answer).
  step "python tool/tests/test_modelpull.py"
  "$PY" tool/tests/test_modelpull.py
  ok "modelpull: model install (5 cases)"

  # the benchmark rebuild builds beside the corpus and renames at the end, so
  # an interrupted gate never leaves target/bench without its corpora.
  step "python tool/tests/test_benchmark.py"
  "$PY" tool/tests/test_benchmark.py
  ok "benchmark: bench runner (5 cases)"

  # the version a release names agrees across the manifests, the lockfile,
  # CITATION.cff and the changelog; the tag checks need a tag and run in ci.
  step "python tool/tests/test_preflight.py"
  "$PY" tool/tests/test_preflight.py
  ok "preflight (7 cases)"

  # the pypi upload takes the release's wheels from the run its index allows,
  # and a rerun uploads only what the index does not have yet.
  step "python tool/tests/test_pypiindex.py"
  "$PY" tool/tests/test_pypiindex.py
  ok "pypiindex (5 cases)"

  # release.yml's edits (distpatch.py) land once on dist's text; the full
  # regenerate check needs the dist binary and runs in gatecheck.yml.
  step "python tool/tests/test_distpatch.py"
  "$PY" tool/tests/test_distpatch.py
  ok "distpatch (3 cases)"

  # the release rehearsal is generated from release.yml, publishes nothing,
  # and its required check fails a needed build that did not pass.
  step "python tool/tests/test_rehearsal.py"
  "$PY" tool/tests/test_rehearsal.py
  ok "rehearsal (7 cases)"

  # setuptest waits for the exact version on each registry, and the release
  # report names what every channel serves, a failed or cancelled run included.
  step "python tool/tests/test_chanprobe.py"
  "$PY" tool/tests/test_chanprobe.py
  ok "chanprobe (8 cases)"

  # each channel's manifest is rendered from the release: its URLs, its
  # hashes, the texts escaped for the file, or nothing written at all.
  step "python tool/tests/test_pkgrender.py"
  "$PY" tool/tests/test_pkgrender.py
  ok "pkgrender (4 cases)"

  # the PPA's source package: debian/ filled from the workspace texts, the
  # vendored crates' checksums, and a wrong tag refused before any write.
  step "python tool/tests/test_ppasource.py"
  "$PY" tool/tests/test_ppasource.py
  ok "ppasource (5 cases)"

  # what serves urna.dev/apt: created once, kept on a second run, and no
  # route of the Worker beyond urna.dev/apt/*.
  step "python tool/tests/test_aptserver.py"
  "$PY" tool/tests/test_aptserver.py
  ok "aptserver (3 cases)"

  # a test that lacks an external resource fails where the run requires
  # the need, and otherwise logs its skip (tool/tests/testneeds.tsv).
  step "python tool/tests/test_testneeds.py"
  "$PY" tool/tests/test_testneeds.py
  ok "testneeds (6 cases)"

  # the embedders' own self-tests: the stdlib lexical floor, the model
  # registry, the potion table through the C ABI against numpy + tokenizers
  # (the release build left the library in target/release), the offline
  # retrieve demo, and the siglip2 query path (the clip need: its packages
  # and pinned snapshot are optional here).
  step "python tool/tests/test_lexifloor.py"
  "$PY" tool/tests/test_lexifloor.py
  ok "lexifloor: the stdlib static embedder"
  step "python tool/tests/test_presetmap.py"
  "$PY" tool/tests/test_presetmap.py
  ok "presetmap: the model registry (12 cases)"
  step "python tool/tests/test_potiontab.py"
  "$PY" tool/tests/test_potiontab.py
  ok "potiontab: the potion embedder, native and reference (8 cases)"
  step "python tool/tests/test_retrieval.py"
  "$PY" tool/tests/test_retrieval.py
  ok "retrieval: the offline retrieve demo (3 cases)"
  step "python tool/tests/test_clipsnaps.py"
  "$PY" tool/tests/test_clipsnaps.py
  ok "clipsnaps: the siglip2 query path (2 cases)"

  # the bridge's media blob pair and multimodal space bands, on files built
  # through urna.build.
  step "python tool/tests/test_mediablob.py"
  "$PY" tool/tests/test_mediablob.py
  ok "mediablob: the blob pair (5 cases)"
  step "python tool/tests/test_spaceband.py"
  "$PY" tool/tests/test_spaceband.py
  ok "spaceband: the space bands (5 cases)"

  # the release pull request is prepared by cargo-release in its own worktree,
  # signed, and touches only the version, the lockfile, the changelog and
  # CITATION.cff. it needs the pinned cargo-release (the `release` need,
  # which this tier requires: without it the suite fails here).
  step "python tool/tests/test_releasepr.py"
  "$PY" tool/tests/test_releasepr.py
  ok "releasepr (4 cases)"
}
