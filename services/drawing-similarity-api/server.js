import { createServer } from 'node:http';

const port = Number(process.env.PORT || 8080);

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
    const body = await readJson(request);
    sendJson(response, 202, {
      mode: 'mock',
      accepted: true,
      recordId: body.recordId || null
    });
    return;
  }

  sendJson(response, 404, { error: 'not found' });
});

server.listen(port, () => {
  console.log('drawing-similarity-api listening on port ' + port);
});
