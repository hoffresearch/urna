# offline by construction

urna is sovereign: the runtime never opens a socket, and every query is
answered from the memory-mapped file on the local machine. nobody outside the
operator has to be online, trusted, or even reachable for the database to
work. that is the whole privacy story, and it needs no policy engine.

to keep that promise for a brand-new user, the default embedder is a static,
offline embedder that ships with the tool. it needs no model download and no
network round-trip on first use, and it is deterministic, so a build is
byte-identical and reproducible. a power user can bring a stronger embedding
model instead, and the model's fingerprint is recorded so the corpus and the
query embedder must agree or the search fails loudly.

sovereign data is the priority: a corpus you build stays on the box you built
it on, by construction rather than by promise.
