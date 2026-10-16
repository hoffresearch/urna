# urna-av1dec

The [rav1d](https://github.com/memorysafety/rav1d) AV1 decoder, the Rust
port of dav1d, carried in the urna tree as the decoder of the AVIF blobs a
`.urna` file stores. BSD-2-Clause, see `LICENSE`; the version, the crate
hash and what was copied are in `UPSTREAM`.

Why embedded and not a dependency: the crates.io build pulls `paste`
(RUSTSEC-2024-0436, unmaintained) and `to_method` (CC0-1.0), and ships
assembly for the platforms urna does not control. Here the code is ours to
change: only the Rust is kept, the `asm` features are gone, `paste` is
expanded by hand in `src/wedge.rs` (the remaining `paste!` sites in
`src/itx.rs` sit in asm-only macros that never expand), and `to_method`
is the ten-line trait in `src/to.rs`. Both bit depths exist; the default
feature is `bitdepth_8`, what urna's AVIF corpora use.

Updating: diff this tree against the upstream tag it names, apply what
matters, and record the new version and hash in `UPSTREAM`.

The files keep upstream's layout and length, out of the 639-line limit the
rest of the tree follows, until a module is rewritten here.
