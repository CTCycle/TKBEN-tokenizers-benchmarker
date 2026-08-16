import { createReadStream, existsSync, statSync } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { createServer, request as httpRequest } from 'node:http';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const clientRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const staticRoot = resolve(clientRoot, 'dist/tkben-angular/browser');
const args = process.argv.slice(2);
const valueFor = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback;
};
const host = valueFor('--host', process.env.UI_HOST || '127.0.0.1');
const port = Number(valueFor('--port', process.env.UI_PORT || 8000));
const strictPort = args.includes('--strictPort');
const apiBase = (process.env.VITE_API_BASE_URL || '/api').startsWith('/')
  ? process.env.VITE_API_BASE_URL || '/api'
  : `/${process.env.VITE_API_BASE_URL}`;
const apiTarget = `http://${process.env.FASTAPI_HOST || '127.0.0.1'}:${process.env.FASTAPI_PORT || 5000}`;

const mimeTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
};

function proxyApi(request, response) {
  const target = new URL(request.url || '/', apiTarget);
  const outgoing = httpRequest(
    target,
    {
      method: request.method,
      headers: { ...request.headers, host: target.host },
      timeout: 600000,
    },
    (proxyResponse) => {
      response.writeHead(proxyResponse.statusCode || 502, proxyResponse.headers);
      proxyResponse.pipe(response);
    },
  );
  outgoing.on('timeout', () => outgoing.destroy(new Error('API proxy timeout')));
  outgoing.on('error', (error) => {
    if (!response.headersSent) response.writeHead(502, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ detail: error.message }));
  });
  request.pipe(outgoing);
}

async function serveStatic(request, response) {
  const requestPath = decodeURIComponent((request.url || '/').split('?')[0]);
  const safePath = normalize(requestPath).replace(/^([.][.][\\/])+/, '');
  let filePath = resolve(staticRoot, `.${safePath}`);
  if (!filePath.startsWith(staticRoot)) filePath = join(staticRoot, 'index.html');
  try {
    if (!existsSync(filePath) || !statSync(filePath).isFile()) filePath = join(staticRoot, 'index.html');
    const body = await readFile(filePath);
    response.writeHead(200, { 'content-type': mimeTypes[extname(filePath)] || 'application/octet-stream' });
    response.end(body);
  } catch {
    response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    response.end('Not found');
  }
}

if (!(await stat(staticRoot).catch(() => null))) {
  console.error(`Angular production output not found at ${staticRoot}. Run npm run build first.`);
  process.exit(1);
}

const server = createServer((request, response) => {
  if ((request.url || '/').startsWith(apiBase)) {
    void proxyApi(request, response);
    return;
  }
  void serveStatic(request, response);
});

server.on('error', (error) => {
  if (error.code === 'EADDRINUSE' && !strictPort) {
    server.listen(0, host);
    return;
  }
  throw error;
});
server.listen(port, host, () => {
  const address = server.address();
  const actualPort = typeof address === 'object' && address ? address.port : port;
  console.log(`Angular preview running at http://${host}:${actualPort}`);
});
