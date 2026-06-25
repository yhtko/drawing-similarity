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
const defaultEmbeddingProvider = process.env.NODE_ENV === 'production' ? 'openclip' : 'dummy';
const embeddingProvider = String(process.env.EMBEDDING_PROVIDER || defaultEmbeddingProvider).toLowerCase();
const qdrantCollection = process.env.QDRANT_COLLECTION || (
  embeddingProvider === 'dinov2' ? 'drawing_similarity_dinov2' :
    embeddingProvider === 'openclip' ? 'drawing_similarity_openclip' : 'drawing_similarity'
);
const defaultVectorSize = embeddingProvider === 'openclip' ? 512 : 384;
const expectedVectorSize = Number(process.env.VECTOR_SIZE || defaultVectorSize);
const dummyVectorSize = expectedVectorSize;
const pythonBin = process.env.PYTHON_BIN || 'python';
const openClipScript = process.env.OPENCLIP_SCRIPT || join(process.cwd(), 'embed_openclip.py');
const embeddingEndpoint = String(process.env.EMBEDDING_ENDPOINT || '').replace(/\/+$/, '');
const embeddingImageMode = String(process.env.EMBED_IMAGE_MODE || 'full').toLowerCase();
const embeddingRotations = String(process.env.EMBED_ROTATIONS || '0')
  .split(',')
  .map((value) => Number(value.trim()))
  .filter((value) => Number.isFinite(value) && value % 90 === 0)
  .map((value) => ((value % 360) + 360) % 360)
  .filter((value, index, values) => values.indexOf(value) === index);
if (!embeddingRotations.length) {
  embeddingRotations.push(0);
}
const defaultOcrEngine = process.env.NODE_ENV === 'production' ? 'tesseract' : 'none';
const ocrEngine = String(process.env.OCR_ENGINE || defaultOcrEngine).toLowerCase();
const ocrLangs = String(process.env.OCR_LANGS || 'eng+jpn').trim();
const tesseractBin = process.env.TESSERACT_BIN || 'tesseract';
const configuredOcrTimeoutMs = Number(process.env.OCR_TIMEOUT_MS || 120000);
const ocrTimeoutMs = Number.isFinite(configuredOcrTimeoutMs) && configuredOcrTimeoutMs > 0
  ? configuredOcrTimeoutMs
  : 120000;
const defaultShapeEngine = process.env.NODE_ENV === 'production' ? 'simple' : 'none';
const shapeEngine = String(process.env.SHAPE_ENGINE || defaultShapeEngine).toLowerCase();
const shapeImageMode = String(process.env.SHAPE_IMAGE_MODE || embeddingImageMode).toLowerCase();
const shapeScript = process.env.SHAPE_SCRIPT || join(process.cwd(), 'extract_shape_profile.py');
const configuredShapeTimeoutMs = Number(process.env.SHAPE_TIMEOUT_MS || 120000);
const shapeTimeoutMs = Number.isFinite(configuredShapeTimeoutMs) && configuredShapeTimeoutMs > 0
  ? configuredShapeTimeoutMs
  : 120000;
const configuredOpenClipTimeoutMs = Number(process.env.OPENCLIP_TIMEOUT_MS || 180000);
const openClipTimeoutMs = Number.isFinite(configuredOpenClipTimeoutMs) && configuredOpenClipTimeoutMs > 0
  ? configuredOpenClipTimeoutMs
  : 180000;
const scoreVectorFloor = Number(process.env.SCORE_VECTOR_FLOOR || 0.75);
const scoreVectorCeiling = Number(process.env.SCORE_VECTOR_CEILING || 0.98);
const scoreVectorWeight = Number(process.env.SCORE_VECTOR_WEIGHT || 0.78);
const scoreMetadataWeight = Number(process.env.SCORE_METADATA_WEIGHT || 0.12);
const scoreShapeWeight = Number(process.env.SCORE_SHAPE_WEIGHT || 0.10);
let payloadIndexesReady = false;

const formatLogFields = (fields = {}) => Object.entries(fields)
  .filter(([, value]) => value !== undefined && value !== null && value !== '')
  .map(([key, value]) => key + '=' + String(value).replace(/\s+/g, ' ').slice(0, 1000))
  .join(' ');

const indexLog = (message, fields) => {
  const suffix = formatLogFields(fields);
  console.log('[index] ' + message + (suffix ? ' ' + suffix : ''));
};

const indexError = (message, fields) => {
  const suffix = formatLogFields(fields);
  console.error('[index] ' + message + (suffix ? ' ' + suffix : ''));
};

const attachStep = (error, step, status, extra = {}) => {
  if (!error.step) {
    error.step = step;
  }
  if (status && !error.status) {
    error.status = status;
  }
  Object.assign(error, extra);
  return error;
};

const createStepError = (message, step, status, extra = {}) => (
  attachStep(new Error(message), step, status, extra)
);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const formatEndpointError = (error) => {
  const parts = [error.message];
  if (error.cause?.code) {
    parts.push('code=' + error.cause.code);
  }
  if (error.cause?.address) {
    parts.push('address=' + error.cause.address);
  }
  if (error.cause?.port) {
    parts.push('port=' + error.cause.port);
  }
  return parts.filter(Boolean).join(' ');
};

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
  embeddingImageMode,
  embeddingRotations,
  expectedVectorSize,
  vectorSize: expectedVectorSize,
  qdrantConfigured: isQdrantConfigured(),
  qdrantCollection,
  dummyVectorSize,
  pythonCommand: pythonBin,
  openclipScript: openClipScript,
  embeddingEndpoint: embeddingEndpoint || null,
  openclipDevice: process.env.OPENCLIP_DEVICE || 'auto',
  openclipTimeoutMs: openClipTimeoutMs,
  ocrEngine,
  ocrLangs,
  tesseractBin,
  ocrTimeoutMs,
  shapeEngine,
  shapeImageMode,
  shapeScript,
  shapeTimeoutMs,
  timeout: {
    ocrMs: ocrTimeoutMs,
    shapeMs: shapeTimeoutMs,
    openclipMs: openClipTimeoutMs
  },
  scoring: {
    vectorFloor: scoreVectorFloor,
    vectorCeiling: scoreVectorCeiling,
    vectorWeight: scoreVectorWeight,
    metadataWeight: scoreMetadataWeight,
    shapeWeight: scoreShapeWeight
  },
  nodeVersion: process.version,
  cwd: process.cwd(),
  openclip: {
    model: process.env.OPENCLIP_MODEL || 'ViT-B-32',
    pretrained: process.env.OPENCLIP_PRETRAINED || 'laion2b_s34b_b79k',
    device: process.env.OPENCLIP_DEVICE || 'auto'
  },
  dinov2: {
    model: process.env.DINO_MODEL || 'facebook/dinov2-small',
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
      productName: index === 0 ? body.productName || 'sample part' : 'similar candidate ' + (index + 1),
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
  const {
    log = null,
    errorLog = null,
    logLabel = 'command',
    step = 'command',
    timeoutMs = 0,
    timeoutMessage = command + ' timed out',
    ...spawnOptions
  } = options;
  const child = spawn(command, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    ...spawnOptions
  });
  const stdout = [];
  const stderr = [];
  let settled = false;
  let timedOut = false;
  let timeoutId = null;

  if (log) {
    log(logLabel + ' spawn start', { command, timeoutMs });
  }

  const settle = (callback, value) => {
    if (settled) {
      return;
    }
    settled = true;
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    callback(value);
  };

  if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
    timeoutId = setTimeout(() => {
      timedOut = true;
      if (errorLog) {
        errorLog(logLabel + ' timeout', { timeoutMs });
      }
      child.kill('SIGKILL');
    }, timeoutMs);
  }

  child.stdout.on('data', (chunk) => {
    stdout.push(chunk);
    if (log) {
      log(logLabel + ' stdout received', { bytes: chunk.length });
    }
  });
  child.stderr.on('data', (chunk) => {
    stderr.push(chunk);
    if (errorLog) {
      errorLog(logLabel + ' stderr received', {
        bytes: chunk.length,
        text: chunk.toString('utf8')
      });
    }
  });

  child.on('error', (error) => {
    if (errorLog) {
      errorLog(logLabel + ' spawn error', { error: error.message });
    }
    settle(reject, attachStep(error, step));
  });
  child.on('close', (code) => {
    if (log) {
      log(logLabel + ' exit code=' + code);
    }
    if (timedOut) {
      settle(reject, createStepError(timeoutMessage, step, 504, { timeoutMs }));
      return;
    }

    const output = Buffer.concat(stdout).toString('utf8');
    if (code !== 0) {
      settle(reject, createStepError(command + ' exited with ' + code + ': ' + Buffer.concat(stderr).toString('utf8'), step));
      return;
    }

    try {
      settle(resolve, JSON.parse(output));
    } catch (error) {
      settle(reject, attachStep(new Error('Failed to parse ' + command + ' JSON output: ' + error.message + ' output=' + output.slice(0, 300)), step));
    }
  });
});

const runTextCommand = (command, args, options = {}) => new Promise((resolve, reject) => {
  const {
    log = null,
    errorLog = null,
    logLabel = 'command',
    step = 'command',
    timeoutMs = 0,
    timeoutMessage = command + ' timed out',
    ...spawnOptions
  } = options;
  const child = spawn(command, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    ...spawnOptions
  });
  const stdout = [];
  const stderr = [];
  let settled = false;
  let timedOut = false;
  let timeoutId = null;

  if (log) {
    log(logLabel + ' spawn start', { command, timeoutMs });
  }

  const settle = (callback, value) => {
    if (settled) {
      return;
    }
    settled = true;
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    callback(value);
  };

  if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
    timeoutId = setTimeout(() => {
      timedOut = true;
      if (errorLog) {
        errorLog(logLabel + ' timeout', { timeoutMs });
      }
      child.kill('SIGKILL');
    }, timeoutMs);
  }

  child.stdout.on('data', (chunk) => {
    stdout.push(chunk);
    if (log) {
      log(logLabel + ' stdout received', { bytes: chunk.length });
    }
  });
  child.stderr.on('data', (chunk) => {
    stderr.push(chunk);
    if (errorLog) {
      errorLog(logLabel + ' stderr received', {
        bytes: chunk.length,
        text: chunk.toString('utf8')
      });
    }
  });

  child.on('error', (error) => {
    if (errorLog) {
      errorLog(logLabel + ' spawn error', { error: error.message });
    }
    settle(reject, attachStep(error, step));
  });
  child.on('close', (code) => {
    if (log) {
      log(logLabel + ' exit code=' + code);
    }
    if (timedOut) {
      settle(reject, createStepError(timeoutMessage, step, 504, { timeoutMs }));
      return;
    }
    if (code !== 0) {
      settle(reject, createStepError(command + ' exited with ' + code + ': ' + Buffer.concat(stderr).toString('utf8'), step));
      return;
    }
    settle(resolve, Buffer.concat(stdout).toString('utf8'));
  });
});

const normalizeEmbeddingResult = (data) => {
  if (!Array.isArray(data.vector) || !data.vector.length) {
    throw new Error('OpenCLIP returned an empty vector');
  }
  return {
    provider: data.provider || 'openclip',
    model: data.model || '',
    pretrained: data.pretrained || '',
    device: data.device || '',
    imageMode: data.image_mode || embeddingImageMode,
    image: data.image || null,
    vector: data.vector
  };
};

const buildOpenClipVectorViaEndpoint = async (buffer, context = {}) => {
  const startedAt = Date.now();
  let lastError = null;

  if (context.log) {
    context.log('openclip endpoint request start', { endpoint: embeddingEndpoint, timeoutMs: openClipTimeoutMs });
  }

  while (Date.now() - startedAt < openClipTimeoutMs) {
    const controller = new AbortController();
    const remainingMs = Math.max(1000, openClipTimeoutMs - (Date.now() - startedAt));
    const timeoutId = setTimeout(() => controller.abort(), remainingMs);
    try {
      const response = await fetch(embeddingEndpoint + '/embed', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          image_base64: buffer.toString('base64'),
          image_mode: embeddingImageMode,
          rotation: Number(context.rotation || 0)
        }),
        signal: controller.signal
      });
      const text = await response.text();
      if (context.log) {
        context.log('openclip endpoint response received', { status: response.status, bytes: text.length });
      }
      if (!response.ok) {
        throw createStepError('OpenCLIP endpoint failed with ' + response.status + ': ' + text.slice(0, 500), 'embedding', response.status === 504 ? 504 : 500);
      }
      return normalizeEmbeddingResult(JSON.parse(text));
    } catch (error) {
      lastError = error;
      if (error.name === 'AbortError') {
        throw createStepError('OpenCLIP embedding timed out', 'embedding', 504, { timeoutMs: openClipTimeoutMs });
      }
      if (context.errorLog) {
        context.errorLog('openclip endpoint retry', { endpoint: embeddingEndpoint, error: formatEndpointError(error) });
      }
      await sleep(1000);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  if (lastError) {
    throw attachStep(new Error('Embedding endpoint failed: ' + formatEndpointError(lastError)), 'embedding');
  }
  throw createStepError('OpenCLIP embedding timed out', 'embedding', 504, { timeoutMs: openClipTimeoutMs });
};

const buildOpenClipVector = async (buffer, context = {}) => {
  if (embeddingEndpoint) {
    return buildOpenClipVectorViaEndpoint(buffer, context);
  }

  const workDir = await mkdtemp(join(tmpdir(), 'drawing-embedding-'));
  const imagePath = join(workDir, 'page.png');

  try {
    await writeFile(imagePath, buffer);
    const data = await runJsonCommand(pythonBin, [openClipScript, imagePath], {
      env: {
        ...process.env,
        EMBED_IMAGE_MODE: embeddingImageMode
      },
      log: context.log,
      errorLog: context.errorLog,
      logLabel: 'openclip',
      step: 'embedding',
      timeoutMs: openClipTimeoutMs,
      timeoutMessage: 'OpenCLIP embedding timed out'
    });
    return normalizeEmbeddingResult(data);
  } finally {
    await rm(workDir, { recursive: true, force: true });
  }
};

const buildOcrText = async (pngBuffer, context = {}) => {
  if (ocrEngine === 'none') {
    return {
      engine: 'none',
      langs: '',
      text: '',
      imagePath: ''
    };
  }
  if (ocrEngine !== 'tesseract') {
    const error = new Error('Unsupported OCR_ENGINE: ' + ocrEngine);
    error.status = 500;
    throw error;
  }

  const workDir = await mkdtemp(join(tmpdir(), 'drawing-ocr-'));
  const imagePath = join(workDir, 'page.png');

  try {
    await writeFile(imagePath, pngBuffer);
    const text = await runTextCommand(tesseractBin, [imagePath, 'stdout', '--oem', '1', '--psm', '6', '-l', ocrLangs], {
      env: process.env,
      log: context.log,
      errorLog: context.errorLog,
      logLabel: 'tesseract',
      step: 'ocr',
      timeoutMs: ocrTimeoutMs,
      timeoutMessage: 'OCR timed out'
    });
    return {
      engine: 'tesseract',
      langs: ocrLangs,
      text: String(text || '').replace(/\u0000/g, '').trim(),
      imagePath
    };
  } finally {
    await rm(workDir, { recursive: true, force: true });
  }
};

const buildShapeProfile = async (pngBuffer, context = {}) => {
  if (shapeEngine === 'none') {
    return {
      engine: 'none',
      mode: 'none',
      bbox: null,
      bboxAspectRatio: 0,
      bboxAreaRatio: 0,
      inkRatio: 0,
      centroidX: 0.5,
      centroidY: 0.5,
      edgeDensity: 0,
      verticalProfile: [],
      horizontalProfile: []
    };
  }
  if (shapeEngine !== 'simple') {
    const error = new Error('Unsupported SHAPE_ENGINE: ' + shapeEngine);
    error.status = 500;
    throw error;
  }

  const workDir = await mkdtemp(join(tmpdir(), 'drawing-shape-'));
  const imagePath = join(workDir, 'page.png');

  try {
    await writeFile(imagePath, pngBuffer);
    const data = await runJsonCommand(pythonBin, [shapeScript, imagePath], {
      env: {
        ...process.env,
        SHAPE_ENGINE: shapeEngine,
        SHAPE_IMAGE_MODE: shapeImageMode,
        EMBED_IMAGE_MODE: embeddingImageMode
      },
      log: context.log,
      errorLog: context.errorLog,
      logLabel: 'shape',
      step: 'shape',
      timeoutMs: shapeTimeoutMs,
      timeoutMessage: 'Shape extraction timed out'
    });

    return {
      engine: data.engine || 'simple',
      mode: data.mode || shapeEngine,
      cropBox: data.cropBox || data.crop_box || null,
      width: Number(data.width || 0),
      height: Number(data.height || 0),
      sourceWidth: Number(data.sourceWidth || 0),
      sourceHeight: Number(data.sourceHeight || 0),
      bbox: data.bbox || null,
      bboxAspectRatio: Number(data.bboxAspectRatio || 0),
      bboxAreaRatio: Number(data.bboxAreaRatio || 0),
      inkRatio: Number(data.inkRatio || 0),
      centroidX: Number(data.centroidX || 0.5),
      centroidY: Number(data.centroidY || 0.5),
      edgeDensity: Number(data.edgeDensity || 0),
      verticalProfile: Array.isArray(data.verticalProfile) ? data.verticalProfile : [],
      horizontalProfile: Array.isArray(data.horizontalProfile) ? data.horizontalProfile : []
    };
  } finally {
    await rm(workDir, { recursive: true, force: true });
  }
};

const normalizeOcrText = (text) => String(text || '')
  .replace(/\u0000/g, '')
  .replace(/\r\n/g, '\n')
  .replace(/\r/g, '\n')
  .split('\n')
  .map((line) => line.replace(/\s+/g, ' ').trim())
  .filter(Boolean)
  .join('\n')
  .trim();

const pickMatch = (text, patterns) => {
  for (const pattern of patterns) {
    const match = String(text || '').match(pattern);
    if (match) {
      return match[1] || match[0] || '';
    }
  }
  return '';
};

const inferShapeCategory = (text, productName) => {
  const source = ((text || '') + ' ' + (productName || '')).toLowerCase();
  const rules = [
    ['bracket', /bracket|stay|lever/i],
    ['plate', /plate|sheet|panel/i],
    ['shaft', /shaft|rod|pin/i],
    ['pipe', /pipe|tube/i],
    ['cover', /cover|cap/i],
    ['frame', /frame/i],
    ['housing', /housing|case|box/i]
  ];
  for (const [label, pattern] of rules) {
    if (pattern.test(source)) {
      return label;
    }
  }
  return '';
};

const extractOcrFields = (ocrText, body = {}) => {
  const text = normalizeOcrText(ocrText);
  const lines = text.split('\n').filter(Boolean);

  const drawingNo = String(body.drawingNo || pickMatch(text, [
    /(?:DRAWING\s*NO\.?|DWG\.?\s*NO\.?|PART\s*NO\.?|NO\.?)\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-\/_\.]{2,})/i,
    /\b([A-Z][0-9A-Z]{2,}[-_][0-9A-Z][0-9A-Z\-\/_\.]{2,})\b/i
  ]) || '').trim();
  const productName = String(body.productName || pickMatch(text, [
    /(?:TITLE|NAME|DESCRIPTION)\s*[:#-]?\s*([^\n]{2,80})/i
  ]) || '').trim();
  const material = String(body.material || pickMatch(text, [
    /(?:MATERIAL|MATL\.?|MAT\.?)\s*[:#-]?\s*([^\n]{2,80})/i,
    /\b(SUS\d{3,4}|SS4?0?0|SPCC|SPHC|AL(?:UMINUM)?|A\d{4}|S45C|SCM\d{2}|SKD\d{2}|CRS|SGCC|SUJ\d{2})\b/i
  ]) || '').trim();
  const thickness = String(body.thickness || pickMatch(text, [
    /(?:THK|T)\s*[:#-]?\s*([0-9]+(?:\.[0-9]+)?\s*(?:mm)?)/i,
    /\bt\s*([0-9]+(?:\.[0-9]+)?\s*(?:mm)?)\b/i
  ]) || '').trim();
  const customer = String(body.customer || pickMatch(text, [
    /(?:CUSTOMER|CLIENT)\s*[:#-]?\s*([^\n]{2,80})/i
  ]) || '').trim();
  const revision = String(body.revision || pickMatch(text, [
    /(?:REV(?:ISION)?|REV\.)\s*[:#-]?\s*([A-Z0-9\-]+)/i
  ]) || '').trim();

  const extracted = {
    drawingNo,
    productName,
    material,
    thickness,
    customer,
    revision,
    shapeCategory: inferShapeCategory(text, productName || body.productName || ''),
    ocrText: text,
    ocrLines: lines,
    extractionConfidence: 0.25
  };

  const score = [drawingNo, productName, material, thickness, customer, revision].filter(Boolean).length;
  extracted.extractionConfidence = Number((0.25 + score * 0.12).toFixed(2));
  return extracted;
};

const normalizeArrayNumber = (value) => {
  let values = value;
  if (typeof values === 'string') {
    try {
      values = JSON.parse(values);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(values)) {
    return [];
  }
  return values
    .map((entry) => Number(entry))
    .filter((entry) => Number.isFinite(entry));
};

const normalizeShapeProfile = (value) => {
  if (!value) {
    return null;
  }

  let profile = value;
  if (typeof value === 'string') {
    try {
      profile = JSON.parse(value);
    } catch {
      return null;
    }
  }

  if (!profile || typeof profile !== 'object') {
    return null;
  }

  return {
    engine: String(profile.engine || 'simple'),
    mode: String(profile.mode || 'simple'),
    cropBox: profile.cropBox || profile.crop_box || null,
    width: Number(profile.width || 0),
    height: Number(profile.height || 0),
    sourceWidth: Number(profile.sourceWidth || profile.source_width || 0),
    sourceHeight: Number(profile.sourceHeight || profile.source_height || 0),
    bbox: typeof profile.bbox === 'string'
      ? (() => {
        try {
          const parsed = JSON.parse(profile.bbox);
          return parsed && typeof parsed === 'object' ? parsed : null;
        } catch {
          return null;
        }
      })()
      : profile.bbox && typeof profile.bbox === 'object'
        ? profile.bbox
        : null,
    bboxAspectRatio: Number(profile.bboxAspectRatio || 0),
    bboxAreaRatio: Number(profile.bboxAreaRatio || 0),
    inkRatio: Number(profile.inkRatio || 0),
    centroidX: Number(profile.centroidX || 0.5),
    centroidY: Number(profile.centroidY || 0.5),
    edgeDensity: Number(profile.edgeDensity || 0),
    verticalProfile: normalizeArrayNumber(profile.verticalProfile),
    horizontalProfile: normalizeArrayNumber(profile.horizontalProfile)
  };
};

const safeJsonStringify = (value) => {
  try {
    return JSON.stringify(value);
  } catch {
    return '';
  }
};

const ratioSimilarity = (left, right) => {
  const a = Number(left);
  const b = Number(right);
  if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 0 || b <= 0) {
    return null;
  }
  const diff = Math.abs(Math.log(a / b));
  return Math.max(0, 1 - Math.min(diff / 1.5, 1));
};

const boundedDifferenceSimilarity = (left, right) => {
  const a = Number(left);
  const b = Number(right);
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    return null;
  }
  return Math.max(0, 1 - Math.min(Math.abs(a - b), 1));
};

const profileSimilarity = (leftValues, rightValues) => {
  if (!Array.isArray(leftValues) || !Array.isArray(rightValues) || !leftValues.length || !rightValues.length) {
    return null;
  }
  const length = Math.min(leftValues.length, rightValues.length);
  let diff = 0;
  for (let index = 0; index < length; index += 1) {
    diff += Math.abs(Number(leftValues[index] || 0) - Number(rightValues[index] || 0));
  }
  return Math.max(0, 1 - Math.min(diff / 2, 1));
};

const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));

const calibrateVectorScore = (score) => {
  const raw = Number(score);
  if (!Number.isFinite(raw)) {
    return 0;
  }
  const floor = Number.isFinite(scoreVectorFloor) ? scoreVectorFloor : 0.75;
  const ceiling = Number.isFinite(scoreVectorCeiling) && scoreVectorCeiling > floor
    ? scoreVectorCeiling
    : floor + 0.01;
  return clamp01((raw - floor) / (ceiling - floor));
};

const scoreShapeCandidate = (candidatePayload = {}, queryShape = null) => {
  const candidateShape = normalizeShapeProfile(
    candidatePayload.shape_profile_json ||
    candidatePayload.shape_profile ||
    (candidatePayload.shape_bbox_aspect_ratio !== undefined ? {
      engine: candidatePayload.shape_engine || 'simple',
      mode: candidatePayload.shape_mode || 'simple',
      bbox: candidatePayload.shape_bbox_json || null,
      bboxAspectRatio: candidatePayload.shape_bbox_aspect_ratio,
      bboxAreaRatio: candidatePayload.shape_bbox_area_ratio,
      inkRatio: candidatePayload.shape_ink_ratio,
      centroidX: candidatePayload.shape_centroid_x,
      centroidY: candidatePayload.shape_centroid_y,
      edgeDensity: candidatePayload.shape_edge_density,
      verticalProfile: candidatePayload.shape_vertical_profile_json || [],
      horizontalProfile: candidatePayload.shape_horizontal_profile_json || []
    } : null)
  );

  if (!candidateShape || !queryShape) {
    return {
      score: 0,
      breakdown: {
        aspect: 0,
        area: 0,
        ink: 0,
        centroid: 0,
        edge: 0,
        projection: 0,
        total: 0
      },
      reasons: []
    };
  }

  const aspectSim = ratioSimilarity(queryShape.bboxAspectRatio, candidateShape.bboxAspectRatio);
  const areaSim = boundedDifferenceSimilarity(queryShape.bboxAreaRatio, candidateShape.bboxAreaRatio);
  const inkSim = boundedDifferenceSimilarity(queryShape.inkRatio, candidateShape.inkRatio);
  const centroidXSim = boundedDifferenceSimilarity(queryShape.centroidX, candidateShape.centroidX);
  const centroidYSim = boundedDifferenceSimilarity(queryShape.centroidY, candidateShape.centroidY);
  const edgeSim = boundedDifferenceSimilarity(queryShape.edgeDensity, candidateShape.edgeDensity);
  const verticalSim = profileSimilarity(queryShape.verticalProfile, candidateShape.verticalProfile);
  const horizontalSim = profileSimilarity(queryShape.horizontalProfile, candidateShape.horizontalProfile);

  const projectionSimValues = [verticalSim, horizontalSim].filter((value) => Number.isFinite(value));
  const projectionSim = projectionSimValues.length
    ? projectionSimValues.reduce((sum, value) => sum + value, 0) / projectionSimValues.length
    : null;

  const breakdown = {
    aspect: Number(((aspectSim || 0) * 0.03).toFixed(4)),
    area: Number(((areaSim || 0) * 0.03).toFixed(4)),
    ink: Number(((inkSim || 0) * 0.03).toFixed(4)),
    centroid: Number((((centroidXSim || 0) + (centroidYSim || 0)) / 2 * 0.03).toFixed(4)),
    edge: Number(((edgeSim || 0) * 0.02).toFixed(4)),
    projection: Number(((projectionSim || 0) * 0.09).toFixed(4)),
    total: 0
  };

  breakdown.total = Number((breakdown.aspect + breakdown.area + breakdown.ink + breakdown.centroid + breakdown.edge + breakdown.projection).toFixed(4));

  const reasons = [];
  if ((projectionSim || 0) >= 0.7) {
    reasons.push('profile similar');
  }
  if ((aspectSim || 0) >= 0.8 && (areaSim || 0) >= 0.7) {
    reasons.push('outline similar');
  }
  if ((edgeSim || 0) >= 0.7) {
    reasons.push('edge similar');
  }

  return {
    score: breakdown.total,
    breakdown,
    reasons
  };
};

const normalizeSearchText = (value) => String(value || '')
  .replace(/\u0000/g, '')
  .trim()
  .toLowerCase();

const parseThicknessValue = (value) => {
  const text = normalizeSearchText(value).replace(/,/g, '.');
  const match = text.match(/([0-9]+(?:\.[0-9]+)?)/);
  if (!match) {
    return null;
  }
  const number = Number(match[1]);
  return Number.isFinite(number) ? number : null;
};

const buildQueryProfile = (body = {}, indexedPayload = null) => ({
  drawingNo: String(indexedPayload?.drawing_no || indexedPayload?.ocr_drawing_no || body.drawingNo || '').trim(),
  productName: String(indexedPayload?.product_name || indexedPayload?.ocr_product_name || body.productName || '').trim(),
  material: String(indexedPayload?.ocr_material || body.material || '').trim(),
  thickness: String(indexedPayload?.ocr_thickness || body.thickness || '').trim(),
  customer: String(indexedPayload?.ocr_customer || body.customer || '').trim(),
  revision: String(indexedPayload?.ocr_revision || body.revision || '').trim(),
  shapeCategory: String(indexedPayload?.ocr_shape_category || body.shapeCategory || '').trim(),
  ocrText: String(indexedPayload?.ocr_text || body.ocrText || '').trim()
});

const scoreCandidate = (candidatePayload = {}, query = {}) => {
  const reasons = [];
  const breakdown = {
    vector: 0,
    drawingNo: 0,
    productName: 0,
    material: 0,
    thickness: 0,
    customer: 0,
    revision: 0,
    shapeCategory: 0,
    shape: 0,
    bonus: 0,
    total: 0
  };

  const scoreFromVector = Number(candidatePayload.__vectorScore || 0);
  const calibratedVectorScore = calibrateVectorScore(scoreFromVector);
  breakdown.vectorRaw = Number.isFinite(scoreFromVector) ? Number(scoreFromVector.toFixed(4)) : 0;
  breakdown.vector = Number(calibratedVectorScore.toFixed(4));

  const candidateDrawingNo = normalizeSearchText(candidatePayload.drawing_no || candidatePayload.ocr_drawing_no);
  const candidateProductName = normalizeSearchText(candidatePayload.product_name || candidatePayload.ocr_product_name);
  const candidateMaterial = normalizeSearchText(candidatePayload.ocr_material);
  const candidateThickness = normalizeSearchText(candidatePayload.ocr_thickness);
  const candidateCustomer = normalizeSearchText(candidatePayload.ocr_customer);
  const candidateRevision = normalizeSearchText(candidatePayload.ocr_revision);
  const candidateShapeCategory = normalizeSearchText(candidatePayload.ocr_shape_category);

  const queryDrawingNo = normalizeSearchText(query.drawingNo);
  const queryProductName = normalizeSearchText(query.productName);
  const queryMaterial = normalizeSearchText(query.material);
  const queryThickness = normalizeSearchText(query.thickness);
  const queryCustomer = normalizeSearchText(query.customer);
  const queryRevision = normalizeSearchText(query.revision);
  const queryShapeCategory = normalizeSearchText(query.shapeCategory);
  const shapeScore = scoreShapeCandidate(candidatePayload, query.shape || null);

  if (queryDrawingNo && candidateDrawingNo && queryDrawingNo === candidateDrawingNo) {
    breakdown.drawingNo = 0.15;
    reasons.push('drawingNo match');
  }
  if (queryProductName && candidateProductName && queryProductName === candidateProductName) {
    breakdown.productName = 0.1;
    reasons.push('productName match');
  }
  if (queryMaterial && candidateMaterial && queryMaterial === candidateMaterial) {
    breakdown.material = 0.08;
    reasons.push('material match');
  }
  if (queryCustomer && candidateCustomer && queryCustomer === candidateCustomer) {
    breakdown.customer = 0.06;
    reasons.push('customer match');
  }
  if (queryRevision && candidateRevision && queryRevision === candidateRevision) {
    breakdown.revision = 0.03;
    reasons.push('revision match');
  }
  if (queryShapeCategory && candidateShapeCategory && queryShapeCategory === candidateShapeCategory) {
    breakdown.shapeCategory = 0.08;
    reasons.push('shape category match');
  }
  if (shapeScore.score > 0) {
    breakdown.shape = Number(shapeScore.score.toFixed(4));
    reasons.push(...shapeScore.reasons);
  }

  const queryThicknessValue = parseThicknessValue(queryThickness);
  const candidateThicknessValue = parseThicknessValue(candidateThickness);
  if (queryThicknessValue !== null && candidateThicknessValue !== null) {
    const diff = Math.abs(queryThicknessValue - candidateThicknessValue);
    if (diff === 0) {
      breakdown.thickness = 0.08;
      reasons.push('thickness match');
    } else if (diff <= 0.2) {
      breakdown.thickness = 0.05;
      reasons.push('thickness close');
    } else if (diff <= 0.5) {
      breakdown.thickness = 0.02;
      reasons.push('thickness roughly close');
    }
  }

  const metadataBonus = breakdown.drawingNo + breakdown.productName + breakdown.material + breakdown.thickness + breakdown.customer + breakdown.revision + breakdown.shapeCategory;
  const metadataScore = clamp01(metadataBonus / 0.58);
  const normalizedShapeScore = clamp01(breakdown.shape / 0.23);
  const totalWeight = Math.max(0.01, scoreVectorWeight + scoreMetadataWeight + scoreShapeWeight);
  const weightedTotal = (
    calibratedVectorScore * scoreVectorWeight +
    metadataScore * scoreMetadataWeight +
    normalizedShapeScore * scoreShapeWeight
  ) / totalWeight;

  breakdown.metadata = Number(metadataScore.toFixed(4));
  breakdown.bonus = Number((metadataBonus + breakdown.shape).toFixed(3));
  breakdown.total = Number(clamp01(weightedTotal).toFixed(4));

  if (!reasons.length && candidatePayload.ocr_text) {
    reasons.push('ocr text available');
  }

  return {
    score: breakdown.total,
    scoreBreakdown: breakdown,
    reasons,
    shapeScoreBreakdown: shapeScore.breakdown
  };
};

const assertEmbeddingVector = (embedding) => {
  if (!Array.isArray(embedding.vector) || !embedding.vector.length) {
    throw new Error('Embedding provider returned an empty vector');
  }
  if (!Number.isFinite(expectedVectorSize) || expectedVectorSize <= 0) {
    throw new Error('VECTOR_SIZE must be a positive number');
  }
  if (embedding.vector.length !== expectedVectorSize) {
    const error = new Error(
      'Embedding vector size mismatch: provider=' + embedding.provider +
      ' actual=' + embedding.vector.length +
      ' expected=' + expectedVectorSize +
      '. Set VECTOR_SIZE=' + embedding.vector.length +
      ' and use a matching QDRANT_COLLECTION.'
    );
    error.status = 409;
    throw error;
  }
  return embedding;
};

const buildEmbedding = async (buffer, context = {}) => {
  if (embeddingImageMode !== 'full' && embeddingImageMode !== 'center_crop' && embeddingImageMode !== 'auto_roi') {
    const error = new Error('Unsupported EMBED_IMAGE_MODE: ' + embeddingImageMode);
    error.status = 500;
    throw error;
  }

  if (embeddingProvider === 'openclip' || embeddingProvider === 'dinov2') {
    if (embeddingProvider === 'dinov2' && !embeddingEndpoint) {
      const error = new Error('EMBEDDING_PROVIDER=dinov2 requires EMBEDDING_ENDPOINT');
      error.status = 500;
      throw error;
    }
    return assertEmbeddingVector(await buildOpenClipVector(buffer, context));
  }

  if (embeddingProvider !== 'dummy' && embeddingProvider !== 'sha256-dummy') {
    const error = new Error('Unsupported EMBEDDING_PROVIDER: ' + embeddingProvider);
    error.status = 500;
    throw error;
  }

  return assertEmbeddingVector({
    provider: 'sha256-dummy',
    model: '',
    pretrained: '',
    imageMode: embeddingImageMode,
    image: null,
    vector: buildVector(buffer)
  });
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

const toPointIdWithRotation = (recordId, rotation) => {
  const normalizedRotation = ((Number(rotation) || 0) % 360 + 360) % 360;
  if (normalizedRotation === 0) {
    return toPointId(recordId);
  }
  return createHash('sha256').update(String(recordId) + ':rot:' + normalizedRotation).digest('hex').slice(0, 32);
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

const upsertDrawing = async (body, embedding, context = {}) => {
  if (!isQdrantConfigured()) {
    if (context.log) {
      context.log('qdrant upsert skipped', { configured: false });
    }
    return { configured: false, upserted: false };
  }

  const embeddings = Array.isArray(embedding) ? embedding : [embedding];
  const firstVector = embeddings[0]?.vector || [];
  if (context.log) {
    context.log('qdrant ensure collection start', { collection: qdrantCollection, vectorSize: firstVector.length });
  }
  try {
    await ensureCollection(firstVector.length);
  } catch (error) {
    throw attachStep(error, 'qdrant_ensure_collection');
  }
  if (context.log) {
    context.log('qdrant ensure collection done', { collection: qdrantCollection });
    context.log('qdrant upsert start', { collection: qdrantCollection, recordId: body.recordId, points: embeddings.length });
  }

  const basePayload = {
    tenant_id: body.tenantId || 'default',
    record_id: String(body.recordId),
    app_id: body.appId ? String(body.appId) : '',
    drawing_no: context.extracted?.drawingNo || body.drawingNo || '',
    product_name: context.extracted?.productName || body.productName || '',
    part_name: context.extracted?.productName || body.productName || '',
    file_name: body.fileName || '',
    file_key: body.fileKey || '',
    indexed_at: new Date().toISOString(),
    ocr_engine: context.ocr?.engine || 'none',
    ocr_langs: context.ocr?.langs || '',
    ocr_text: context.ocr?.text || '',
    ocr_drawing_no: context.extracted?.drawingNo || '',
    ocr_product_name: context.extracted?.productName || '',
    ocr_material: context.extracted?.material || '',
    ocr_thickness: context.extracted?.thickness || '',
    ocr_customer: context.extracted?.customer || '',
    ocr_revision: context.extracted?.revision || '',
    ocr_shape_category: context.extracted?.shapeCategory || '',
    ocr_extraction_confidence: context.extracted?.extractionConfidence ?? null,
    shape_engine: context.shape?.engine || 'none',
    shape_mode: context.shape?.mode || 'none',
    shape_image_mode: shapeImageMode,
    shape_roi_json: context.shape?.cropBox ? safeJsonStringify(context.shape.cropBox) : '',
    shape_width: context.shape?.width ?? null,
    shape_height: context.shape?.height ?? null,
    shape_source_width: context.shape?.sourceWidth ?? null,
    shape_source_height: context.shape?.sourceHeight ?? null,
    shape_profile_json: context.shape ? safeJsonStringify(context.shape) : '',
    shape_bbox_json: context.shape?.bbox ? safeJsonStringify(context.shape.bbox) : '',
    shape_bbox_aspect_ratio: context.shape?.bboxAspectRatio ?? null,
    shape_bbox_area_ratio: context.shape?.bboxAreaRatio ?? null,
    shape_ink_ratio: context.shape?.inkRatio ?? null,
    shape_centroid_x: context.shape?.centroidX ?? null,
    shape_centroid_y: context.shape?.centroidY ?? null,
    shape_edge_density: context.shape?.edgeDensity ?? null,
    shape_vertical_profile_json: context.shape?.verticalProfile ? safeJsonStringify(context.shape.verticalProfile) : '',
    shape_horizontal_profile_json: context.shape?.horizontalProfile ? safeJsonStringify(context.shape.horizontalProfile) : ''
  };

  try {
    await qdrantRequest('/collections/' + encodeURIComponent(qdrantCollection) + '/points?wait=true', {
      method: 'PUT',
      body: JSON.stringify({
        points: embeddings.map((entry) => ({
          id: toPointIdWithRotation(body.recordId, entry.rotation || 0),
          vector: entry.vector,
          payload: {
            ...basePayload,
            embedding_provider: entry.provider,
            embedding_model: entry.model || '',
            embedding_pretrained: entry.pretrained || '',
            embedding_image_mode: entry.imageMode || embeddingImageMode,
            embedding_rotation: Number(entry.rotation || 0),
            embedding_image_json: entry.image ? safeJsonStringify(entry.image) : '',
            embedding_rotations: embeddingRotations.join(','),
            embedding_vector_size: entry.vector.length
          }
        }))
      })
    });
  } catch (error) {
    throw attachStep(error, 'qdrant_upsert');
  }
  if (context.log) {
    context.log('qdrant upsert done', { collection: qdrantCollection, recordId: body.recordId, points: embeddings.length });
  }

  return {
    configured: true,
    upserted: true,
    collection: qdrantCollection,
    vectorSize: firstVector.length,
    points: embeddings.length,
    rotations: embeddings.map((entry) => Number(entry.rotation || 0))
  };
};

const getIndexedDrawingVector = async (body) => {
  if (!isQdrantConfigured() || !body.recordId) {
    return null;
  }

  try {
    const ids = embeddingRotations.map((rotation) => toPointIdWithRotation(body.recordId, rotation));
    const data = await qdrantRequest('/collections/' + encodeURIComponent(qdrantCollection) + '/points', {
      method: 'POST',
      body: JSON.stringify({
        ids,
        with_payload: true,
        with_vector: true
      })
    });
    const points = Array.isArray(data.result) ? data.result : [];
    if (!points.length) {
      return null;
    }

    const vectors = [];
    let selectedPayload = null;
    let selectedPointId = null;
    for (const point of points) {
      const payload = point.payload || {};
      if (String(payload.tenant_id || 'default') !== String(body.tenantId || 'default')) {
        continue;
      }
      if (body.appId && payload.app_id && String(payload.app_id) !== String(body.appId)) {
        continue;
      }
      const pointVector = getPointVector(point);
      if (!Array.isArray(pointVector) || !pointVector.length) {
        continue;
      }
      vectors.push(pointVector);
      selectedPayload = selectedPayload || payload;
      selectedPointId = selectedPointId || point.id;
    }

    if (!vectors.length) {
      return null;
    }

    return {
      pointId: selectedPointId,
      payload: selectedPayload || {},
      vector: vectors.length === 1 ? vectors[0] : vectors,
      vectors
    };
  } catch (error) {
    if (error.status === 404) {
      return null;
    }
    throw error;
  }
};

const searchDrawings = async (body, vector, queryProfile = {}) => {
  if (!isQdrantConfigured()) {
    return null;
  }

  const queryVectors = Array.isArray(vector) && Array.isArray(vector[0]) ? vector : [vector];
  await ensureCollection(queryVectors[0].length);
  const limit = Math.min((Number(body.limit || 10) + 1) * Math.max(1, queryVectors.length) * 4, 100);
  const byRecord = new Map();

  for (let queryIndex = 0; queryIndex < queryVectors.length; queryIndex += 1) {
    const queryVector = queryVectors[queryIndex];
    const queryRotation = embeddingRotations[queryIndex] ?? queryIndex;
    const data = await qdrantRequest('/collections/' + encodeURIComponent(qdrantCollection) + '/points/search', {
      method: 'POST',
      body: JSON.stringify({
        vector: queryVector,
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

    for (const item of data.result || []) {
      const payload = item.payload || {};
      if (String(payload.record_id || '') === String(body.recordId || '')) {
        continue;
      }
      const key = String(payload.record_id || item.id);
      const score = Number(item.score || 0);
      const rotationScore = {
        queryRotation,
        candidateRotation: Number(payload.embedding_rotation ?? 0),
        vectorRaw: Number(score.toFixed(4)),
        pointId: item.id
      };
      const existing = byRecord.get(key);
      if (existing) {
        existing.rotationScores.push(rotationScore);
        if (score > Number(existing.item.score || 0)) {
          existing.item = item;
          existing.queryRotation = queryRotation;
          existing.candidateRotation = rotationScore.candidateRotation;
        }
      } else {
        byRecord.set(key, {
          item,
          queryRotation,
          candidateRotation: rotationScore.candidateRotation,
          rotationScores: [rotationScore]
        });
      }
    }
  }

  return Array.from(byRecord.values())
    .map((entry) => {
      const item = entry.item;
      const payload = item.payload || {};
      const scored = scoreCandidate({
        ...payload,
        __vectorScore: Number(item.score || 0)
      }, queryProfile);
      return {
        recordId: payload.record_id || item.id,
        drawingNo: payload.drawing_no || 'record ' + item.id,
        productName: payload.product_name || '',
        customer: payload.file_name || '',
        material: payload.ocr_material || '',
        thickness: payload.ocr_thickness || '',
        revision: payload.ocr_revision || '',
        shapeCategory: payload.ocr_shape_category || '',
        ocrText: payload.ocr_text || '',
        shape: normalizeShapeProfile(payload.shape_profile_json || payload.shape_profile || null),
        vectorRaw: scored.scoreBreakdown.vectorRaw,
        vectorScore: scored.scoreBreakdown.vector,
        embeddingRotation: entry.candidateRotation ?? payload.embedding_rotation ?? null,
        embeddingImage: (() => { try { return payload.embedding_image_json ? JSON.parse(payload.embedding_image_json) : null; } catch { return null; } })(),
        queryEmbeddingRotation: entry.queryRotation ?? null,
        rotationScores: entry.rotationScores
          .sort((a, b) => b.vectorRaw - a.vectorRaw)
          .slice(0, 20),
        score: scored.score,
        scoreBreakdown: scored.scoreBreakdown,
        shapeScoreBreakdown: scored.shapeScoreBreakdown,
        reasons: scored.reasons
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.min(Number(body.limit || 10), 10));
};

const buildMatchConfidence = (results = []) => {
  const topScore = Number(results[0]?.scoreBreakdown?.vectorRaw || 0);
  const secondScore = Number(results[1]?.scoreBreakdown?.vectorRaw || 0);
  const margin = Number(Math.max(0, topScore - secondScore).toFixed(4));
  let level = 'low';
  if (topScore >= 0.9 && margin >= 0.03) {
    level = 'high';
  } else if (topScore >= 0.87 && margin >= 0.015) {
    level = 'medium';
  }
  return {
    level,
    topScore: Number(topScore.toFixed(4)),
    secondScore: Number(secondScore.toFixed(4)),
    margin
  };
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
    return {
      pngBuffer: await readFile(imagePath),
      imagePath
    };
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
  const { pngBuffer } = await convertPdfFirstPageToPng(pdfBuffer);
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
        const queryProfile = buildQueryProfile(body, indexed?.payload || null);
        queryProfile.shape = normalizeShapeProfile(
          indexed?.payload?.shape_profile_json ||
          indexed?.payload?.shape_profile ||
          null
        );

        if (!vector && body.fileKey) {
          const { pngBuffer } = await loadRecordImage(body);
          const shape = await buildShapeProfile(pngBuffer);
          queryProfile.shape = shape;
          const queryEmbeddings = [];
          for (const rotation of embeddingRotations) {
            const queryEmbedding = await buildEmbedding(pngBuffer, { rotation });
            queryEmbedding.rotation = rotation;
            queryEmbeddings.push(queryEmbedding);
          }
          embedding = queryEmbeddings[0];
          vector = queryEmbeddings.map((entry) => entry.vector);
          queryVectorSource = 'rendered-pdf';
        }

        if (vector) {
          const results = await searchDrawings(body, vector, queryProfile);
          const matchConfidence = buildMatchConfidence(results || []);
          sendJson(response, 200, {
            mode: indexed ? 'qdrant-indexed' : 'qdrant-' + embedding.provider,
            query: {
              tenantId: body.tenantId || 'default',
              appId: body.appId,
              recordId: body.recordId,
              drawingNo: queryProfile.drawingNo || body.drawingNo || '',
              productName: queryProfile.productName || body.productName || '',
              material: queryProfile.material || body.material || '',
              thickness: queryProfile.thickness || body.thickness || '',
              customer: queryProfile.customer || body.customer || '',
              revision: queryProfile.revision || body.revision || '',
              shapeCategory: queryProfile.shapeCategory || body.shapeCategory || '',
              shape: queryProfile.shape || null
            },
            qdrant: {
              collection: qdrantCollection,
              vectorSize: Array.isArray(vector[0]) ? vector[0].length : vector.length,
              queryRotations: embeddingRotations,
              queryVectorSource,
              queryPointId: indexed?.pointId || null
            },
            matchConfidence,
            extracted: indexed?.payload ? {
              drawingNo: indexed.payload.ocr_drawing_no || indexed.payload.drawing_no || '',
              productName: indexed.payload.ocr_product_name || indexed.payload.product_name || '',
              material: indexed.payload.ocr_material || '',
              thickness: indexed.payload.ocr_thickness || '',
              customer: indexed.payload.ocr_customer || '',
              revision: indexed.payload.ocr_revision || '',
              shapeCategory: indexed.payload.ocr_shape_category || '',
              ocrTextLength: String(indexed.payload.ocr_text || '').length,
              shape: normalizeShapeProfile(indexed?.payload?.shape_profile_json || indexed?.payload?.shape_profile || null)
            } : null,
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
    let step = 'start';
    indexLog('start');
    try {
      step = 'payload';
      const body = await readJson(request);
      indexLog('payload received', {
        recordId: body.recordId,
        tenantId: body.tenantId || 'default'
      });

      if (!body.recordId) {
        throw createStepError('recordId is required', step, 400);
      }
      if (!body.fileKey) {
        throw createStepError('fileKey is required', step, 400);
      }

      step = 'fetch_kintone_file';
      indexLog('fetch kintone file start', { recordId: body.recordId });
      const pdfBuffer = await fetchKintoneFile(body.fileKey).catch((error) => {
        throw attachStep(error, step);
      });
      indexLog('fetch kintone file done', { bytes: pdfBuffer.length });

      step = 'pdf_render';
      indexLog('pdf render start', { bytes: pdfBuffer.length, dpi: renderDpi });
      const { pngBuffer, imagePath } = await convertPdfFirstPageToPng(pdfBuffer).catch((error) => {
        throw attachStep(error, step);
      });
      indexLog('pdf render done', {
        imagePath,
        bytes: pngBuffer.length
      });

      step = 'ocr';
      indexLog('ocr start', {
        engine: ocrEngine,
        langs: ocrLangs
      });
      const ocr = await buildOcrText(pngBuffer, {
        log: indexLog,
        errorLog: indexError
      }).catch((error) => {
        if (ocrEngine === 'none') {
          return {
            engine: 'none',
            langs: '',
            text: ''
          };
        }
        throw attachStep(error, step);
      });
      indexLog('ocr done', {
        engine: ocr.engine,
        textLength: ocr.text.length
      });

      step = 'shape';
      indexLog('shape start', {
        engine: shapeEngine
      });
      const shape = await buildShapeProfile(pngBuffer, {
        log: indexLog,
        errorLog: indexError
      }).catch((error) => {
        if (shapeEngine === 'none') {
          return {
            engine: 'none',
            mode: 'none',
            bbox: null,
            bboxAspectRatio: 0,
            bboxAreaRatio: 0,
            inkRatio: 0,
            centroidX: 0.5,
            centroidY: 0.5,
            edgeDensity: 0,
            verticalProfile: [],
            horizontalProfile: []
          };
        }
        throw attachStep(error, step);
      });
      indexLog('shape done', {
        engine: shape.engine,
        bboxAspectRatio: shape.bboxAspectRatio,
        edgeDensity: shape.edgeDensity
      });

      step = 'extraction';
      indexLog('extraction start');
      const extracted = extractOcrFields(ocr.text, body);
      indexLog('extraction done', {
        drawingNo: extracted.drawingNo,
        material: extracted.material,
        thickness: extracted.thickness,
        customer: extracted.customer,
        revision: extracted.revision,
        shapeCategory: extracted.shapeCategory,
        confidence: extracted.extractionConfidence
      });

      step = 'embedding';
      indexLog('embedding start', {
        provider: embeddingProvider,
        imageMode: embeddingImageMode
      });
      const embeddings = [];
      for (const rotation of embeddingRotations) {
        indexLog('embedding rotation start', { rotation });
        const entry = await buildEmbedding(pngBuffer, {
          log: indexLog,
          errorLog: indexError,
          rotation
        }).catch((error) => {
          throw attachStep(error, step);
        });
        entry.rotation = rotation;
        embeddings.push(entry);
      }
      const embedding = embeddings[0];
      indexLog('embedding done', {
        dimension: embedding.vector.length,
        provider: embedding.provider,
        imageMode: embedding.imageMode || embeddingImageMode,
        rotations: embeddings.map((entry) => entry.rotation).join(',')
      });

      step = 'qdrant_upsert';
      const qdrant = await upsertDrawing(body, embeddings, {
        log: indexLog,
        errorLog: indexError,
        ocr,
        extracted,
        shape
      });

      sendJson(response, 202, {
        ok: true,
        mode: qdrant.upserted ? 'qdrant-' + embedding.provider : 'pdf-ready',
        accepted: true,
        tenantId: body.tenantId || 'default',
        appId: body.appId || null,
        recordId: body.recordId,
        drawingNo: body.drawingNo || '',
        productName: body.productName || '',
        fileName: body.fileName || '',
        ocr: {
          engine: ocr.engine,
          langs: ocr.langs,
          textLength: ocr.text.length
        },
        extracted: {
          drawingNo: extracted.drawingNo,
          productName: extracted.productName,
          material: extracted.material,
          thickness: extracted.thickness,
          customer: extracted.customer,
          revision: extracted.revision,
          shapeCategory: extracted.shapeCategory,
          extractionConfidence: extracted.extractionConfidence
        },
        shape: {
          engine: shape.engine,
          mode: shape.mode,
          imageMode: shapeImageMode,
          cropBox: shape.cropBox || null,
          width: shape.width || null,
          height: shape.height || null,
          sourceWidth: shape.sourceWidth || null,
          sourceHeight: shape.sourceHeight || null,
          bboxAspectRatio: shape.bboxAspectRatio,
          bboxAreaRatio: shape.bboxAreaRatio,
          inkRatio: shape.inkRatio,
          centroidX: shape.centroidX,
          centroidY: shape.centroidY,
          edgeDensity: shape.edgeDensity
        },
        shapeProfiles: {
          vertical: shape.verticalProfile,
          horizontal: shape.horizontalProfile
        },
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
          device: embedding.device || '',
          imageMode: embedding.imageMode || embeddingImageMode,
          image: embedding.image || null,
          size: embedding.vector.length,
          rotations: embeddings.map((entry) => entry.rotation),
          images: embeddings.map((entry) => ({ rotation: entry.rotation, image: entry.image || null }))
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
      indexLog('response sent', { status: 202 });
    } catch (error) {
      const failedStep = error.step || step;
      const status = error.status || 500;
      indexError('failed', {
        step: failedStep,
        status,
        error: error.message
      });
      const payload = {
        ok: false,
        error: error.message,
        step: failedStep
      };
      if (error.timeoutMs) {
        payload.timeoutMs = error.timeoutMs;
      }
      sendJson(response, status, payload);
      indexLog('response sent', { status, step: failedStep });
    }
    return;
  }

  sendJson(response, 404, { error: 'not found' });
});

server.listen(port, () => {
  console.log('drawing-similarity-api listening on port ' + port);
});
