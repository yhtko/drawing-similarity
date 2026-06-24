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
QDRANT_COLLECTION=drawing_similarity_openclip
VECTOR_SIZE=512
```

OpenCLIP is the default embedding provider when `NODE_ENV=production`.
Use these values for the default model:

```sh
EMBEDDING_PROVIDER=openclip
OPENCLIP_MODEL=ViT-B-32
OPENCLIP_PRETRAINED=laion2b_s34b_b79k
OPENCLIP_DEVICE=cpu
EMBED_IMAGE_MODE=center_crop
VECTOR_SIZE=512
```

`EMBED_IMAGE_MODE` controls the image sent to the embedding provider:

- `full`: use the full rendered first page.
- `center_crop`: crop the rendered image to x 5%-85% and y 10%-80% before OpenCLIP embedding.

When changing `EMBED_IMAGE_MODE`, re-index the target drawings before judging similarity results. The mode is stored in Qdrant payload as `embedding_image_mode`.

For local smoke tests without Python/OpenCLIP dependencies, set:

```sh
EMBEDDING_PROVIDER=dummy
QDRANT_COLLECTION=drawing_similarity_dummy
VECTOR_SIZE=384
```

The dummy vector is a deterministic SHA-256 based vector generated from the rendered PNG. It is only for validating Cloud Run to Qdrant wiring.

When OpenCLIP is enabled, the Node API calls `embed_openclip.py`, which generates normalized image embeddings. `ViT-B-32` embeddings are 512-dimensional. If `VECTOR_SIZE` or the existing Qdrant collection size does not match the model output, the API returns a clear mismatch error instead of upserting incompatible vectors.

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
