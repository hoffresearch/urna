# nix run github:hoffresearch/urna -- --version
{
  description = "urna, an offline-first vector database in one file, with citations you can verify";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      each = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = each (pkgs: rec {
        urna = pkgs.callPackage ./pkgs/nixos/package.nix { };
        default = urna;
      });
    };
}
