import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const siteRoot = new URL('../../docs/', import.meta.url);
const html = await readFile(new URL('index.html', siteRoot), 'utf8');
const icons = [...html.matchAll(/<link\b[^>]*\brel="icon"[^>]*>/g)].map(([tag]) => ({
  href: tag.match(/\bhref="([^"]+)"/)?.[1],
  type: tag.match(/\btype="([^"]+)"/)?.[1],
  sizes: tag.match(/\bsizes="([^"]+)"/)?.[1]
}));

test('图标资源存在，主页和知识点深链接均保留 GitHub Pages 项目路径', async () => {
  assert.equal(icons.length, 2);
  for (const icon of icons) {
    assert.match(icon.href, /^assets\//);
    assert.ok((await readFile(new URL(icon.href, siteRoot))).length > 0);
    for (const page of [
      'https://hzeroto.github.io/transformer-learning/',
      'https://hzeroto.github.io/transformer-learning/#foundation.matrix-multiplication'
    ]) {
      assert.equal(new URL(icon.href, page).pathname, '/transformer-learning/' + icon.href);
    }
  }
});

test('SVG 图标复用页头标识和品牌蓝色', async () => {
  const icon = icons.find(({ type }) => type === 'image/svg+xml');
  assert.equal(icon?.sizes, 'any');
  const svg = await readFile(new URL(icon.href, siteRoot), 'utf8');
  const styles = await readFile(new URL('assets/styles.css', siteRoot), 'utf8');
  const brand = styles.match(/--brand:\s*(#[\da-f]+);/i)[1];
  const mark = html.match(/<svg class="brand-mark"[\s\S]*?<\/svg>/)[0];
  assert.ok(svg.includes(`fill="${brand}"`));
  assert.ok(svg.includes(mark.match(/<path d="([^"]+)"/)[1]));
  assert.match(svg, /xmlns="http:\/\/www\.w3\.org\/2000\/svg"/);
  assert.match(svg, /viewBox="0 0 32 32"/);
});

test('PNG 备用图标是有效的 32×32 图片', async () => {
  const icon = icons.find(({ type }) => type === 'image/png');
  assert.equal(icon?.sizes, '32x32');
  const png = await readFile(new URL(icon.href, siteRoot));
  assert.deepEqual([...png.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  assert.equal(png.toString('ascii', 12, 16), 'IHDR');
  assert.equal(png.readUInt32BE(16), 32);
  assert.equal(png.readUInt32BE(20), 32);
});
