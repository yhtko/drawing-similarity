import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const port = Number(process.env.PORT || 8080);
const kintoneBaseUrl = String(process.env.KINTONE_BASE_URL || '').replace(/\/+$/, '');
const kintoneApiToken = process.env.KINTONE_API_TOKEN || '';
const renderDpi = Number(process.env.PDF_RENDER_DPI || 160);
const qdrantUrl = String(process.env.QDRANT_URL || '').replace(/\/+$/, '');
const qdrantApiKey = process.env.QDRANT_API_KEY || '';
const qdrantCollection = process.env.QDRANT_COLLECTION || 'drawing_similarity';
const dummyVectorSize = Number(process.env.VECTOR_SIZE || 384);
const embeddingProvider = process.env.EMBEDDING_PROVIDER || 'dummy';
const pythonBin = process.env.PYTHON_BIN || 'python';
const openClipScript = process.env.OPENCLIP_SCRIPT || join(process.cwd(), 'embed_openclip.py');
let payloadIndexesReady = false;

const readJson = async (request) => {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }
  const text = Buffer.concat(chunks).toString('utf8');
  return text ? JSON.parse(text) : {};
};

const sendJson = (response, status, payload) => {
  response.writeHead(status, {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type': 'application/json; charset=utf-8'
  });
  response.end(JSON.stringify(payload));
};

const getRuntimeInfo = () => ({
  embeddingProvider,
  qdrantConfigured: isQdrantConfigured(),
  qdrantCollection,
  dummyVectorSize,
  openclip: {
    model: process.env.OPENCLIP_MODEL || 'ViT-B-32',
    pretrained: process.env.OPENCLIP_PRETRAINED || 'laion2b_s34b_b79k',
    device: process.env.OPENCLIP_DEVICE || 'auto'
  },
  renderDpi
});

const buildMockResults = (body) => {
  const base = Number(body.recordId || 1000);
  return Array.from({ length: Math.min(Number(body.limit || 10), 10) }, (_, index) => {
    const recordId = base + index + 1;
    return {
      recordId,
      drawingNo: 'DWG-' + String(recordId).padStart(5, '0'),
      productName: index === 0 ? body.productName || 'サンプル部品' : '類似候補 ' + (index + 1),
      customer: 'PoC',
      score: Number((0.92 - index * 0.035).toFixed(3))
    };
  });
};

const isQdrantConfigured = () => Boolean(qdrantUrl);

const buildVector = (buffer) => {
  const vector = [];
  let seed = createHash('sha256').update(buffer).digest();

  while (vector.length < dummyVectorSize) {
    seed = createHash('sha256').update(seed).digest();
    for (const byte of seed) {
      vector.push((byte / 127.5) - 1);
      if (vector.length === dummyVectorSize) {
        break;
      }
    }
  }

  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => Number((value / norm).toFixed(8)));
};

const runJsonCommand = (command, args, options = {}) => new Promise((resolve, reject) => {
  const child = spawn(command, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    ...options
  });
  const stdout = [];
  const stderr = [];

  child.stdout.on('data', (chunk) => {
    stdout.push(chunk);
  });
  child.stderr.on('data', (chunk) => {
    stderr.push(chunk);
  });

  child.on('error', reject);
  child.on('close', (code) => {
    const output = Buffer.concat(stdout).toString('utf8');
    if (code !== 0) {
      reject(new Error(command + ' exited with ' + code + ': ' + Buffer.concat(stderr).toString('utf8')));
      return;
    }

    try {
      resolve(JSON.parse(output));
    } catch (error) {
      reject(new Error('Failed to parse ' + command + ' JSON output: ' + error.message + ' output=' + output.slice(0, 300)));
    }
  });
});

const buildOpenClipVector = async (buffer) => {
  const workDir = await mkdtemp(join(tmpdir(), 'drawing-embedding-'));
  const imagePath = join(workDir, 'page.png');

  try {
    await writeFile(imagePath, buffer);
    const data = await runJsonCommand(pythonBin, [openClipScript, imagePath], {
      env: process.env
    });
    if (!Array.isArray(data.vector) || !data.vector.length) {
      throw new Error('OpenCLIP returned an empty vector');
    }
    return {
      provider: data.provider || 'openclip',
      model: data.model || '',
      pretrained: data.pretrained || '',
      vector: data.vector
    };
  } finally {
    await rm(workDir, { recursive: true, force: true });
  }
};

const buildEmbedding = async (buffer) => {
  if (embeddingProvider === 'openclip') {
    return buildOpenClipVector(buffer);
  }

  return {
    provider: 'sha256-dummy',
    model: '',
    pretrained: '',
    vector: buildVector(buffer)
  };
};

const qdrantHeaders = () => {
  const headers = {
    'Content-Type': 'application/json'
  };
  if (qdrantApiKey) {
    headers['api-key'] = qdrantApiKey;
  }
  return headers;
};

const qdrantRequest = async (path, options = {}) => {
  const response = await fetch(qdrantUrl + path, {
    ...options,
    headers: {
      ...qdrantHeaders(),
      ...(options.headers || {})
    }
  });

  if (!response.ok) {
    const body = await response.text();
    const error = new Error('Qdrant request failed: ' + response.status + ' ' + body.slice(0, 300));
    error.status = response.status;
    throw error;
  }

  if (response.status === 204) {
    return {};
  }

  return response.json();
};

const getCollectionVectorSize = (data) => {
  const vectors = data?.result?.config?.params?.vectors;
  if (!vectors) {
    return null;
  }
  if (typeof vectors.size === 'number') {
    return vectors.size;
  }
  if (vectors.default && typeof vectors.default.size === 'number') {
    return vectors.default.size;
  }
  return null;
};

const ensureCollection = async (size) => {
  if (!isQdrantConfigured()) {
    return false;
  }

  const collectionPath = '/collections/' + encodeURIComponent(qdrantCollection);
  const current = await fetch(qdrantUrl + collectionPath, {
    headers: qdrantHeaders()
  });

  if (current.ok) {
    const data = await current.json();
    const currentSize = getCollectionVectorSize(data);
    if (currentSize && currentSize !== size) {
      const error = new Error(
        'Qdrant collection vector size mismatch: collection=' + qdrantCollection +
        ' current=' + currentSize +
        ' requested=' + size +
        '. Use a new QDRANT_COLLECTION or recreate the collection.'
      );
      error.status = 409;
      throw error;
    }
    await ensurePayloadIndexes();
    return true;
  }

  if (current.status !== 404) {
    const body = await current.text();
    const error = new Error('Qdrant collection check failed: ' + current.status + ' ' + body.slice(0, 300));
    error.status = current.status;
    throw error;
  }

  await qdrantRequest('/collections/' + encodeURIComponent(qdrantCollection), {
    method: 'PUT',
    body: JSON.stringify({
      vectors: {
        size,
        distance: 'Cosine'
      }
    })
  });
  await ensurePayloadIndexes();
  return true;
};

const ensurePayloadIndexes = async () => {
  if (!isQdrantConfigured()) {
    return false;
  }
  if (payloadIndexesReady) {
    return true;
  }

  try {
    await qdrantRequest('/collections/' + encodeURIComponent(qdrantCollection) + '/index', {
      method: 'PUT',
      body: JSON.stringify({
        field_name: 'tenant_id',
        field_schema: 'keyword'
      })
    });
  } catch (error) {
    if (error.status !== 409) {
      throw error;
    }
  }

  payloadIndexesReady = true;
  return true;
};

const toPointId = (recordId) => {
  const numeric = Number(recordId);
  if (Number.isSafeInteger(numeric) && numeric > 0) {
    return numeric;
  }
  return createHash('sha256').update(String(recordId)).digest('hex').slice(0, 32);
};

const getPointVector = (point) => {
  if (Array.isArray(point?.vector)) {
    return point.vector;
  }
  if (!point?.vector || typeof point.vector !== 'object') {
    return null;
  }
  return Object.values(point.vector).find((value) => Array.isArray(value)) || null;
};

const upsertDrawing = async (body, vector) => {
  if (!isQdrantConfigured()) {
    return { configured: false, upserted: false };
  }

  await ensureCollection(vector.length);
  await qdrantRequest('/collections/' + encodeURIComponent(qdrantCollection) + '/points?wait=true', {
    method: 'PUT',
    body: JSON.stringify({
      points: [
        {
          id: toPointId(body.recordId),
          vector,
          payload: {
            tenant_id: body.tenantId || 'default',
            record_id: String(body.recordId),
            app_id: body.appId ? String(body.appId) : '',
            drawing_no: body.drawingNo || '',
            product_name: body.productName || '',
            file_name: body.fileName || ''
          }
        }
      ]
    })
  });

  return {
    configured: true,
    upserted: true,
    collection: qdrantCollection,
    vectorSize: vector.length
  };
};

const getIndexedDrawingVector = async (body) => {
  if (!isQdrantConfigured() || !body.recordId) {
    return null;
  }

  try {
    const data = await qdrantRequest('/collections/' + encodeURIComponent(qdrantCollection) + '/points', {
      method: 'POST',
      body: JSON.stringify({
        ids: [toPointId(body.recordId)],
        with_payload: true,
        with_vector: true
      })
    });
    const point = Array.isArray(data.result) ? data.result[0] : null;
    if (!point) {
      return null;
    }

    const payload = point.payload || {};
    if (String(payload.tenant_id || 'default') !== String(body.tenantId || 'default')) {
      return null;
    }
    if (body.appId && payload.app_id && String(payload.app_id) !== String(body.appId)) {
      return null;
    }

    const vector = getPointVector(point);
    if (!Array.isArray(vector) || !vector.length) {
      return null;
    }

    return {
      pointId: point.id,
      payload,
      vector
    };
  } catch (error) {
    if (error.status === 404) {
      return null;
    }
    throw error;
  }
};

const searchDrawings = async (body, vector) => {
  if (!isQdrantConfigured()) {
    return null;
  }

  await ensureCollection(vector.length);
  const limit = Math.min(Number(body.limit || 10) + 1, 25);
  const data = await qdrantRequest('/collections/' + encodeURIComponent(qdrantCollection) + '/points/search', {
    method: 'POST',
    body: JSON.stringify({
      vector,
      limit,
      with_payload: true,
      filter: {
        must: [
          {
            key: 'tenant_id',
            match: {
              value: body.tenantId || 'default'
            }
          }
        ]
      }
    })
  });

  return (data.result || [])
    .filter((item) => String(item.payload?.record_id || '') !== String(body.recordId || ''))
    .slice(0, Math.min(Number(body.limit || 10), 10))
    .map((item) => ({
      recordId: item.payload?.record_id || item.id,
      drawingNo: item.payload?.drawing_no || 'record ' + item.id,
      productName: item.payload?.product_name || '',
      customer: item.payload?.file_name || '',
      score: Number(item.score || 0)
    }));
};

const assertKintoneConfig = () => {
  const missing = [];
  if (!kintoneBaseUrl) {
    missing.push('KINTONE_BASE_URL');
  }
  if (!kintoneApiToken) {
    missing.push('KINTONE_API_TOKEN');
  }
  if (missing.length) {
    const error = new Error('Missing environment variables: ' + missing.join(', '));
    error.status = 501;
    throw error;
  }
};

const fetchKintoneFile = async (fileKey) => {
  assertKintoneConfig();

  const url = new URL('/k/v1/file.json', kintoneBaseUrl);
  url.searchParams.set('fileKey', fileKey);

  const response = await fetch(url, {
    headers: {
      'X-Cybozu-API-Token': kintoneApiToken
    }
  });

  if (!response.ok) {
    const body = await response.text();
    const error = new Error('kintone file download failed: ' + response.status + ' ' + body.slice(0, 200));
    error.status = response.status;
    throw error;
  }

  return Buffer.from(await response.arrayBuffer());
};

const runCommand = (command, args) => new Promise((resolve, reject) => {
  const child = spawn(command, args, {
    stdio: ['ignore', 'ignore', 'pipe']
  });
  const stderr = [];

  child.stderr.on('data', (chunk) => {
    stderr.push(chunk);
  });

  child.on('error', reject);
  child.on('close', (code) => {
    if (code === 0) {
      resolve();
      return;
    }

    reject(new Error(command + ' exited with ' + code + ': ' + Buffer.concat(stderr).toString('utf8')));
  });
});

const convertPdfFirstPageToPng = async (pdfBuffer) => {
  const workDir = await mkdtemp(join(tmpdir(), 'drawing-similarity-'));
  const pdfPath = join(workDir, 'source.pdf');
  const outputBase = join(workDir, 'page');
  const imagePath = outputBase + '.png';

  try {
    await writeFile(pdfPath, pdfBuffer);
    await runCommand('pdftoppm', ['-f', '1', '-singlefile', '-png', '-r', String(renderDpi), pdfPath, outputBase]);
    return await readFile(imagePath);
  } finally {
    await rm(workDir, { recursive: true, force: true });
  }
};

const loadRecordImage = async (body) => {
  if (!body.recordId) {
    const error = new Error('recordId is required');
    error.status = 400;
    throw error;
  }
  if (!body.fileKey) {
    const error = new Error('fileKey is required');
    error.status = 400;
    throw error;
  }

  const pdfBuffer = await fetchKintoneFile(body.fileKey);
  const pngBuffer = await convertPdfFirstPageToPng(pdfBuffer);
  return { pdfBuffer, pngBuffer };
};

const server = createServer(async (request, response) => {
  const url = new URL(request.url || '/', 'http://localhost');

  if (request.method === 'OPTIONS') {
    sendJson(response, 204, {});
    return;
  }

  if (request.method === 'GET' && url.pathname === '/health') {
    sendJson(response, 200, {
      ok: true,
      service: 'drawing-similarity-api',
      runtime: getRuntimeInfo()
    });
    return;
  }

  if (request.method === 'POST' && url.pathname === '/similar') {
    try {
      const body = await readJson(request);
      if (isQdrantConfigured()) {
        const indexed = await getIndexedDrawingVector(body);
        let vector = indexed?.vector || null;
        let embedding = null;
        let queryVectorSource = 'indexed';

        if (!vector && body.fileKey) {
          const { pngBuffer } = await loadRecordImage(body);
          embedding = await buildEmbedding(pngBuffer);
          vector = embedding.vector;
          queryVectorSource = 'rendered-pdf';
        }

        if (vector) {
          const results = await searchDrawings(body, vector);
          sendJson(response, 200, {
            mode: indexed ? 'qdrant-indexed' : 'qdrant-' + embedding.provider,
            query: {
              tenantId: body.tenantId || 'default',
              appId: body.appId,
              recordId: body.recordId,
              drawingNo: body.drawingNo || ''
            },
            qdrant: {
              collection: qdrantCollection,
              vectorSize: vector.length,
              queryVectorSource,
              queryPointId: indexed?.pointId || null
            },
            results
          });
          return;
        }
      }

      sendJson(response, 200, {
        mode: 'mock',
        query: {
          tenantId: body.tenantId || 'default',
          appId: body.appId,
          recordId: body.recordId,
          drawingNo: body.drawingNo || ''
        },
        results: buildMockResults(body)
      });
    } catch (error) {
      sendJson(response, error.status || 500, { error: error.message });
    }
    return;
  }

  if (request.method === 'POST' && url.pathname === '/index') {
    try {
      const body = await readJson(request);
      const { pdfBuffer, pngBuffer } = await loadRecordImage(body);
      const embedding = await buildEmbedding(pngBuffer);
      const qdrant = await upsertDrawing(body, embedding.vector);

      sendJson(response, 202, {
        mode: qdrant.upserted ? 'qdrant-' + embedding.provider : 'pdf-ready',
        accepted: true,
        tenantId: body.tenantId || 'default',
        appId: body.appId || null,
        recordId: body.recordId,
        drawingNo: body.drawingNo || '',
        productName: body.productName || '',
        fileName: body.fileName || '',
        pdf: {
          bytes: pdfBuffer.length
        },
        image: {
          format: 'png',
          page: 1,
          dpi: renderDpi,
          bytes: pngBuffer.length,
          widthHint: Math.round(8.27 * renderDpi)
        },
        vector: {
          provider: embedding.provider,
          model: embedding.model,
          pretrained: embedding.pretrained,
          size: embedding.vector.length
        },
        qdrant,
        next: qdrant.upserted
          ? (
            embedding.provider === 'sha256-dummy'
              ? 'Set EMBEDDING_PROVIDER=openclip to use OpenCLIP embeddings.'
              : 'OpenCLIP embedding was stored. Register more drawings and run similarity search.'
          )
          : 'Set QDRANT_URL to enable vector upsert.'
      });
    } catch (error) {
      sendJson(response, error.status || 500, { error: error.message });
    }
    return;
  }

  sendJson(response, 404, { error: 'not found' });
});

server.listen(port, () => {
  console.log('drawing-similarity-api listening on port ' + port);
});
