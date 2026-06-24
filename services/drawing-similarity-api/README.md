# Drawing Similarity API

Minimal Cloud Run API scaffold for the kintone drawing similarity PoC.

Current endpoints:

- `GET /health`
- `POST /index`: downloads the kintone PDF, renders page 1 to PNG, creates a deterministic dummy vector, and upserts to Qdrant when configured.
- `POST /similar`: searches Qdrant with the already indexed record vector when available. If the record is not indexed yet and `fileKey` is supplied, it falls back to downloading the PDF, rendering page 1, and generating a query vector. Otherwise it returns mock results.

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

Optional environment variables for OpenCLIP:

```sh
EMBEDDING_PROVIDER=openclip
OPENCLIP_MODEL=ViT-B-32
OPENCLIP_PRETRAINED=laion2b_s34b_b79k
OPENCLIP_DEVICE=cpu
```

The default vector is a deterministic SHA-256 based dummy vector generated from the rendered PNG. This is only for validating Cloud Run to Qdrant wiring.

When `EMBEDDING_PROVIDER=openclip` is set, the Node API calls `embed_openclip.py`, which uses OpenCLIP to generate normalized image embeddings. `ViT-B-32` embeddings are 512-dimensional, so use a fresh collection such as:

```sh
QDRANT_COLLECTION=drawing_similarity_openclip
```

If an existing collection has a different vector size, the API returns a clear mismatch error instead of upserting incompatible vectors.

Example request:

```sh
curl -X POST http://localhost:8080/similar \
  -H "Content-Type: application/json" \
  -d "{\"recordId\":123,\"tenantId\":\"default\",\"limit\":10}"
```

Next implementation step:

1. Deploy with `EMBEDDING_PROVIDER=openclip`.
2. Use a new OpenCLIP Qdrant collection.
3. Register 100-300 drawings from kintone.
4. Run similarity search and judge whether the results feel visually similar.
