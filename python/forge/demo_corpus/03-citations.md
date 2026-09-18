# how citations work

every search hit comes back with a stable citation of the form
`urna://content_hash/chunk_id`. the content hash is computed over the decoded
canonical sections of the file, so it identifies the content rather than a
particular copy or a particular compression choice.

because the citation points at content, two people who build the same logical
corpus on two machines get the same citation, and a stored corpus and a
compressed one cite identically. resolving a citation returns the exact
canonical text and the original byte span it came from, which is what lets an
agent quote a source it can prove.

the returned similarity score is a real cosine value, recomputed by an exact
rerank, never an approximate proxy. a result you can cite is a result you can
trust.
