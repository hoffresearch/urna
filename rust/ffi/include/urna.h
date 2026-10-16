/* urna C ABI: open a corpus, ask it, read hits, take image bytes, close.
 *
 * Strings are UTF-8 with a NUL. A function that returns NULL or 0 on
 * failure leaves the reason in urna_last_error(), per thread. No call
 * panics across the boundary. A Corpus may be shared by threads for
 * urna_ask and urna_image; urna_close must be the last call on it. */
#ifndef URNA_H
#define URNA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct UrnaCorpus UrnaCorpus;
typedef struct UrnaHits UrnaHits;

/* "MAJOR.MINOR.PATCH", static. */
const char *urna_version(void);

/* The last error of this thread; "" when none. Valid until the next
 * failing call on this thread. */
const char *urna_last_error(void);

/* Open the .urna at path. model_dir is the potion table dir (tokenizer.json
 * and model.safetensors), or NULL to read URNA_POTION_DIR on the first ask.
 * NULL on failure. */
UrnaCorpus *urna_open(const char *path, const char *model_dir);

/* Close the corpus. Free every UrnaHits of it before; read no pointer
 * urna_image returned after. NULL accepted. */
void urna_close(UrnaCorpus *corpus);

/* Chunks in the corpus; 0 for NULL. */
size_t urna_n_chunks(const UrnaCorpus *corpus);

/* Ask: free text plus the schema's prefix:value terms (t:creature c:r
 * pow>=3), top k hits. NULL on failure. Free with urna_hits_free. */
UrnaHits *urna_ask(const UrnaCorpus *corpus, const char *query, uint32_t k);

size_t urna_hits_len(const UrnaHits *hits);
/* Chunk index of hit i, or -1 out of range. */
int64_t urna_hit_chunk(const UrnaHits *hits, size_t i);
/* Exact cosine score of hit i, or NaN out of range. */
float urna_hit_score(const UrnaHits *hits, size_t i);
/* Stored canonical text of hit i; valid until urna_hits_free. */
const char *urna_hit_text(const UrnaHits *hits, size_t i);
/* urna://content_hash/chunk_id of hit i; valid until urna_hits_free. */
const char *urna_hit_citation(const UrnaHits *hits, size_t i);
void urna_hits_free(UrnaHits *hits);

/* The stored image bytes of chunk (an AVIF the platform decodes), *len
 * long, valid while the corpus is open. NULL when the chunk has none. */
const uint8_t *urna_image(const UrnaCorpus *corpus, size_t chunk, size_t *len);

/* The manifest schema as a JSON array ("[]" without one). Free with
 * urna_string_free. */
char *urna_schema_json(const UrnaCorpus *corpus);

/* "sha256:..." of the corpus content. Free with urna_string_free. */
char *urna_content_hash(const UrnaCorpus *corpus);

void urna_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif
