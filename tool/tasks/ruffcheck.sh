#!/bin/sh
# tool/tasks/ruffcheck.sh -- ruff lint + format check on the python files we own.
#
# ONE list, used by tool/tasks/fullcheck.sh (best-effort, skipped when ruff
# is not importable) and by .github/workflows/gatecheck.yml (mandatory). files not
# on the list are legacy / vendored / generated and are tracked separately;
# when you touch a python module, add it here and make it clean.
#
#   URNA_PYTHON=.venv/bin/python sh tool/tasks/ruffcheck.sh
set -eu
cd "$(dirname "$0")/../.."
PY="${URNA_PYTHON:-python3}"
TARGETS="
rust/bridge/python/urna/embed/searchtxt.py
rust/bridge/python/urna/model/modelhash.py
rust/bridge/python/urna/pipes/buildfile.py
tool/bench/presetrun.py
tool/bench/benchtime.py
tool/bench/benchgate.py
rust/bridge/python/urna/embed/visionemb.py
tool/tests/test_clipsnaps.py
rust/bridge/python/urna/image/discovery.py
rust/bridge/python/urna/image/mediabase.py
rust/bridge/python/urna/image/encstream.py
rust/bridge/python/urna/image/encstills.py
rust/bridge/python/urna/image/gopprober.py
rust/bridge/python/urna/image/decframes.py
rust/bridge/python/urna/image/orchestra.py
rust/bridge/python/urna/image/av1stream.py
rust/bridge/python/urna/image/sequencer.py
rust/bridge/python/urna/image/assembler.py
rust/bridge/python/urna/entry/imgcorpus.py
demo/starter/quickstart.py
rust/bridge/python/urna/entry/imgsearch.py
tool/bench/imageeval.py
tool/bench/imagestat.py
tool/bench/imagerate.py
tool/tests/test_hashguard.py
tool/tests/test_imagepipe.py
tool/tests/test_mediablob.py
tool/tests/test_spaceband.py
rust/bridge/python/urna/model/presetmap.py
rust/bridge/python/urna/embed/stbackend.py
rust/bridge/python/urna/specs/specparse.py
rust/bridge/python/urna/specs/specpaths.py
rust/bridge/python/urna/pipes/rowloader.py
rust/bridge/python/urna/pipes/buildflow.py
rust/bridge/python/urna/pipes/vectcache.py
rust/bridge/python/urna/pipes/recipekey.py
rust/bridge/python/urna/pipes/mediastep.py
rust/bridge/python/urna/pipes/manifests.py
rust/bridge/python/urna/gates/crfpicker.py
rust/bridge/python/urna/gates/taskscore.py
rust/bridge/python/urna/image/mediaplan.py
rust/bridge/python/urna/embed/presetqry.py
rust/bridge/python/urna/model/catalogue.py
rust/bridge/python/urna/model/installer.py
rust/bridge/python/urna/entry/specbuild.py
tool/bench/modelrank.py
tool/bench/modelview.py
rust/bridge/python/urna/entry/uibackend.py
tool/tests/test_specrules.py
tool/tests/test_mediagate.py
tool/tests/test_clispaces.py
tool/tests/test_askrouter.py
tool/tests/test_embedpack.py
tool/tests/test_benchmark.py
tool/tests/test_catalogue.py
tool/tests/test_modelpull.py
tool/tasks/embedpack.py
tool/tasks/preflight.py
tool/tests/test_preflight.py
tool/tasks/pypiindex.py
tool/tests/test_pypiindex.py
tool/tasks/rehearsal.py
tool/tests/test_rehearsal.py
tool/tasks/distpatch.py
tool/tests/test_distpatch.py
tool/tasks/chanprobe.py
tool/tests/test_chanprobe.py
tool/tests/test_releasepr.py
rust/bridge/python/urna/__init__.py
rust/bridge/python/urna/embed/__init__.py
tool/tasks/wheelprep.py
tool/tasks/namecheck.py
tool/tasks/stepcache.py
tool/tasks/benchdata.py
tool/tests/test_benchdata.py
tool/tests/test_atlasdocs.py
tool/tests/test_lintmatch.py
tool/tests/test_stepcache.py
tool/tests/test_namecheck.py
"
# shellcheck disable=SC2086
"$PY" -m ruff check $TARGETS
# shellcheck disable=SC2086
"$PY" -m ruff format --check $TARGETS
