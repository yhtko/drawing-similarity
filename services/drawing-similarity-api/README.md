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

Required environment variables for real kintone PDF download:

```sh
KINTONE_BASE_URL=https://example.kintone.com
KINTONE_API_TOKEN=your-api-token
PDF_RENDER_DPI=160
```

Example request:

```sh
curl -X POST http://localhost:8080/similar \
  -H "Content-Type: application/json" \
  -d "{\"recordId\":123,\"tenantId\":\"default\",\"limit\":10}"
```

Next implementation step:

1. Generate image embeddings with OpenCLIP.
2. Upsert and query Qdrant with `tenant_id` and `record_id` payloads.
3. Replace the mock `/similar` response with Qdrant search results.
