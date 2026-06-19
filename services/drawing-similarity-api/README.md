# Drawing Similarity API

Minimal Cloud Run API scaffold for the kintone drawing similarity PoC.

Current endpoints are mock implementations:

- `GET /health`
- `POST /similar`
- `POST /index`

Run locally:

```sh
npm start
```

Example request:

```sh
curl -X POST http://localhost:8080/similar \
  -H "Content-Type: application/json" \
  -d "{\"recordId\":123,\"tenantId\":\"default\",\"limit\":10}"
```

Next implementation step:

1. Fetch the PDF from kintone by file key.
2. Convert the first page, or selected pages, to images.
3. Generate image embeddings with OpenCLIP.
4. Upsert and query Qdrant with `tenant_id` and `record_id` payloads.
