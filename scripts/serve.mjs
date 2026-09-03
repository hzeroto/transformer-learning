import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { projectRoot } from './learning-lib.mjs';

const port = Number(process.env.PORT || 4174);
const siteRoot = path.join(projectRoot, 'docs');
const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8'
};

const server = http.createServer((request, response) => {
  const requestPath = decodeURIComponent(new URL(request.url, `http://${request.headers.host}`).pathname);
  const relativePath = requestPath === '/' ? 'index.html' : requestPath.replace(/^\/+/, '');
  const filePath = path.resolve(siteRoot, relativePath);
  if (!filePath.startsWith(`${siteRoot}${path.sep}`) && filePath !== siteRoot) {
    response.writeHead(403).end('Forbidden');
    return;
  }
  fs.readFile(filePath, (error, content) => {
    if (error) {
      response.writeHead(error.code === 'ENOENT' ? 404 : 500).end('Not found');
      return;
    }
    response.writeHead(200, {
      'Content-Type': contentTypes[path.extname(filePath)] || 'application/octet-stream',
      'Cache-Control': 'no-store'
    });
    response.end(content);
  });
});

server.listen(port, '127.0.0.1', () => {
  console.log(`学习地图：http://127.0.0.1:${port}`);
});
