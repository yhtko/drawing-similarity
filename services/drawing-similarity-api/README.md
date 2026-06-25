# Drawing Similarity API

Minimal Cloud Run API scaffold for the kintone drawing similarity PoC.

Current endpoints:

- `GET /health`
- `POST /index`: downloads the kintone PDF, renders page 1 to PNG, extracts OCR text, creates an embedding, and upserts to Qdrant when configured.
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
EMBED_IMAGE_MODE=auto_roi
VECTOR_SIZE=512
```


DINOv2 can be tested as an alternate embedding provider. Use a separate Qdrant collection because the vector distribution differs from OpenCLIP:

```sh
EMBEDDING_PROVIDER=dinov2
DINO_MODEL=facebook/dinov2-small
QDRANT_COLLECTION=drawing_similarity_dinov2
VECTOR_SIZE=384
EMBEDDING_ENDPOINT=http://127.0.0.1:9090
```

Start with 10-20 known drawing pairs before re-indexing a larger set. DINOv2 uses the same `EMBED_IMAGE_MODE` preprocessing path as OpenCLIP. To reduce sensitivity to portrait/landscape registration differences, index two rotations during evaluation:`n`n```sh`nEMBED_ROTATIONS=0,90`n```

OCR is enabled by default in production:

```sh
OCR_ENGINE=tesseract
OCR_LANGS=eng+jpn
OCR_TIMEOUT_MS=120000
TESSERACT_BIN=tesseract
```

The API also extracts a first-pass set of structured fields from OCR text:

- `drawingNo`
- `productName`
- `material`
- `thickness`
- `customer`
- `revision`
- `shapeCategory`

Simple shape features are enabled by default in production:

```sh
SHAPE_ENGINE=simple
SHAPE_TIMEOUT_MS=120000
```

The shape profile is computed from the rendered PNG and stored in Qdrant payload for scoring:

- `shape_profile_json`
- `shape_bbox_aspect_ratio`
- `shape_bbox_area_ratio`
- `shape_ink_ratio`
- `shape_centroid_x`
- `shape_centroid_y`
- `shape_edge_density`
- `shape_vertical_profile_json`
- `shape_horizontal_profile_json`

Similarity scores are calibrated before display. Raw OpenCLIP cosine scores for line drawings often cluster high, so `/similar` stretches the vector score between a floor and ceiling, then blends vector, OCR metadata, and shape scores:

```sh
SCORE_VECTOR_FLOOR=0.75
SCORE_VECTOR_CEILING=0.98
SCORE_VECTOR_WEIGHT=0.78
SCORE_METADATA_WEIGHT=0.12
SCORE_SHAPE_WEIGHT=0.10
```

Use `scoreBreakdown.vectorRaw` to inspect the original Qdrant score. If most `vectorRaw` values are still tightly clustered, the next step is improving the image region or shape features rather than tuning the display score.

`EMBED_IMAGE_MODE` controls the image sent to the embedding provider:

- `full`: use the full rendered first page.
- `center_crop`: crop the rendered image to x 5%-85% and y 10%-80% before OpenCLIP embedding.
- `auto_roi`: estimate the main drawing region by removing dominant ruling lines and scoring connected line components.

`SHAPE_IMAGE_MODE` controls the image used for shape features. If omitted, it follows `EMBED_IMAGE_MODE`.

```sh
SHAPE_IMAGE_MODE=auto_roi
```

When changing `EMBED_IMAGE_MODE` or `SHAPE_IMAGE_MODE`, re-index the target drawings before judging similarity results. The mode is stored in Qdrant payload as `embedding_image_mode`, and the estimated shape ROI is stored as `shape_roi_json`.

For local smoke tests without Python/OpenCLIP dependencies, set:

```sh
EMBEDDING_PROVIDER=dummy
QDRANT_COLLECTION=drawing_similarity_dummy
VECTOR_SIZE=384
```

The dummy vector is a deterministic SHA-256 based vector generated from the rendered PNG. It is only for validating Cloud Run to Qdrant wiring.


For faster registration, the Cloud Run image starts a local OpenCLIP embedding server and keeps the model loaded between requests:

```sh
EMBEDDING_ENDPOINT=http://127.0.0.1:9090
EMBEDDING_PORT=9090
```

If `EMBEDDING_ENDPOINT` is unset, the API falls back to spawning `embed_openclip.py` per request. The endpoint mode avoids reloading OpenCLIP for every `/index` call, so the first request pays the model-load cost and later requests should be faster.

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
