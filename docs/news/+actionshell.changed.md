`lintworks` runs shellcheck at warning level on each bash `run:` script of the local actions in `.github/actions`, which actionlint does not read, and runs on a push to `main` that changes them.
