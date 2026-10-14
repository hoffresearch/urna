# urna's CLI built from source, the way nixpkgs builds Rust: the crates
# pinned by Cargo.lock, the version and the summary read from Cargo.toml.
# The source leaves out the model under rust/bridge/python/urna/model, a
# git-lfs file a flake does not fetch, and rust/ingest, a workspace of its
# own. URNA_RELEASE_TAG, as the release workflow sets it, makes build.rs
# name the binary a release of this version, so `urna setup` fetches that
# release's payload instead of reading this source tree.
{ lib, rustPlatform }:

let
  root = ../..;
  cargo = lib.importTOML (root + "/Cargo.toml");
  inherit (cargo.workspace.package) version repository homepage;
in
rustPlatform.buildRustPackage {
  pname = "urna";
  inherit version;

  src = lib.fileset.toSource {
    inherit root;
    fileset = lib.fileset.unions [
      (root + "/Cargo.toml")
      (root + "/Cargo.lock")
      # a test checks that the CLI pins the revision this script fetches.
      (root + "/tool/tasks/getpotion.sh")
      (lib.fileset.difference (root + "/rust") (
        lib.fileset.unions [
          (lib.fileset.maybeMissing (root + "/rust/bridge/python/urna/model"))
          (lib.fileset.maybeMissing (root + "/rust/ingest"))
        ]
      ))
    ];
  };

  cargoLock.lockFile = root + "/Cargo.lock";
  cargoBuildFlags = [ "-p" "urna" ];
  cargoTestFlags = [ "-p" "urna" ];
  env.URNA_RELEASE_TAG = "v${version}";

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    test "$($out/bin/urna --version)" = "urna ${version}"
    runHook postInstallCheck
  '';

  meta = {
    description = cargo.workspace.metadata.pkgs.summary;
    inherit homepage;
    changelog = "${repository}/releases/tag/v${version}";
    license = lib.licenses.mit;
    mainProgram = "urna";
    platforms = lib.platforms.unix;
  };
}
