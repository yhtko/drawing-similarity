# Drawing Similarity API

Minimal Cloud Run API scaffold for the kintone drawing similarity PoC.

Current endpoints:

- `GET /health`
- `POST /index`: downloads the kintone PDF, renders page 1 to PNG, creates a deterministic dummy vector, and upserts to Qdrant when configured.
- `POST /similar`: searches Qdrant with the current PDF dummy vector when configured, otherwise returns mock results.

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

Optional environment variables for Qdrant:

```sh
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=drawing_similarity
VECTOR_SIZE=384
```

The current vector is a deterministic SHA-256 based dummy vector generated from the rendered PNG. This is only for validating Cloud Run to Qdrant wiring. Replace `buildVector` with OpenCLIP embeddings before judging search quality.

Example request:

```sh
curl -X POST http://localhost:8080/similar \
  -H "Content-Type: application/json" \
  -d "{\"recordId\":123,\"tenantId\":\"default\",\"limit\":10}"
```

Next implementation step:

1. Deploy with Qdrant environment variables.
2. Register several drawings from kintone and confirm points are created in Qdrant.
3. Replace the SHA-256 dummy vector with OpenCLIP embeddings.
