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
const vectorSize = Number(process.env.VECTOR_SIZE || 384);

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

  while (vector.length < vectorSize) {
    seed = createHash('sha256').update(seed).digest();
    for (const byte of seed) {
      vector.push((byte / 127.5) - 1);
      if (vector.length === vectorSize) {
        break;
      }
    }
  }

  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => Number((value / norm).toFixed(8)));
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

const ensureCollection = async () => {
  if (!isQdrantConfigured()) {
    return false;
  }

  const collectionPath = '/collections/' + encodeURIComponent(qdrantCollection);
  const current = await fetch(qdrantUrl + collectionPath, {
    headers: qdrantHeaders()
  });

  if (current.ok) {
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
        size: vectorSize,
        distance: 'Cosine'
      }
    })
  });
  return true;
};

const toPointId = (recordId) => {
  const numeric = Number(recordId);
  if (Number.isSafeInteger(numeric) && numeric > 0) {
    return numeric;
  }
  return createHash('sha256').update(String(recordId)).digest('hex').slice(0, 32);
};

const upsertDrawing = async (body, vector) => {
  if (!isQdrantConfigured()) {
    return { configured: false, upserted: false };
  }

  await ensureCollection();
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
    vectorSize
  };
};

const searchDrawings = async (body, vector) => {
  if (!isQdrantConfigured()) {
    return null;
  }

  await ensureCollection();
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
    sendJson(response, 200, { ok: true, service: 'drawing-similarity-api' });
    return;
  }

  if (request.method === 'POST' && url.pathname === '/similar') {
    try {
      const body = await readJson(request);
      if (isQdrantConfigured() && body.fileKey) {
        const { pngBuffer } = await loadRecordImage(body);
        const results = await searchDrawings(body, buildVector(pngBuffer));
        sendJson(response, 200, {
          mode: 'qdrant-dummy-vector',
          query: {
            tenantId: body.tenantId || 'default',
            appId: body.appId,
            recordId: body.recordId,
            drawingNo: body.drawingNo || ''
          },
          qdrant: {
            collection: qdrantCollection,
            vectorSize
          },
          results
        });
        return;
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
      sendJson(response, 400, { error: error.message });
    }
    return;
  }

  if (request.method === 'POST' && url.pathname === '/index') {
    try {
      const body = await readJson(request);
      const { pdfBuffer, pngBuffer } = await loadRecordImage(body);
      const vector = buildVector(pngBuffer);
      const qdrant = await upsertDrawing(body, vector);

      sendJson(response, 202, {
        mode: qdrant.upserted ? 'qdrant-dummy-vector' : 'pdf-ready',
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
          provider: 'sha256-dummy',
          size: vector.length
        },
        qdrant,
        next: qdrant.upserted
          ? 'Replace the sha256 dummy vector with an OpenCLIP embedding.'
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
