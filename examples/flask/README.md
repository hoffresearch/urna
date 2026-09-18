# flask + urna example

offline cited answers from a single-file corpus, minimal flask flavor.
nothing here touches the network after `pip install`.

## setup

```
pip install flask "urna[embed]"
```

## run

```
flask --app app run --port 8000
```

## try

```
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"query": "vector search on the edge", "k": 2}'
```

see `../fastapi/README.md` for the corpus bootstrap notes; the flow is the
same (`URNA_FILE` points at your corpus, the demo builds itself once).
