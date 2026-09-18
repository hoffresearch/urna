# fastapi + urna example

offline cited answers from a single-file corpus. the query is embedded with
the potion table bundled in the `urna` wheel, so nothing here touches the
network after `pip install`.

## setup

```
pip install fastapi uvicorn "urna[embed]"
```

## run

```
uvicorn main:app --port 8000
```

the demo corpus (`demo_fastapi.urna`) builds itself on first run with
`reproducible=True`, so two machines produce the same `file_hash`. point
`URNA_FILE` at a real potion-built corpus for real data.

## try

```
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"query": "vector search on the edge", "k": 2}'
```

each hit returns the stored canonical text, the exact-cosine score, the
source uri, and the `urna://content_hash/chunk_id` citation.
