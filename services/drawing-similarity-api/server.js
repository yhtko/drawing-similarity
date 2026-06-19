import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const port = Number(process.env.PORT || 8080);
const kintoneBaseUrl = String(process.env.KINTONE_BASE_URL || '').replace(/\/+$/, '');
const kintoneApiToken = process.env.KINTONE_API_TOKEN || '';
const renderDpi = Number(process.env.PDF_RENDER_DPI || 160);

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
      if (!body.recordId) {
        sendJson(response, 400, { error: 'recordId is required' });
        return;
      }
      if (!body.fileKey) {
        sendJson(response, 400, { error: 'fileKey is required' });
        return;
      }

      const pdfBuffer = await fetchKintoneFile(body.fileKey);
      const pngBuffer = await convertPdfFirstPageToPng(pdfBuffer);

      sendJson(response, 202, {
        mode: 'pdf-ready',
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
        next: 'OpenCLIP embedding and Qdrant upsert are not implemented yet.'
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
