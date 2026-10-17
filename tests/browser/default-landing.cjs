// 使用独立临时浏览器验证首次打开的真实视口，不通过截图或点击间接滚动页面。
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { readFile, mkdir } = require('node:fs/promises');
const { chromium } = require('playwright');

async function main() {
  const docsPath = path.resolve(__dirname, '../../docs');
  const siteUrl = pathToFileURL(docsPath + path.sep).href;
  const dataSource = await readFile(path.join(docsPath, 'assets/learning-data.generated.js'), 'utf8');
  const browser = await chromium.launch({
    executablePath: process.env.BROWSER_EXECUTABLE_PATH || undefined,
    headless: true, args: ['--disable-background-networking']
  });
  try {
    for (const scenario of [
      { width: 1800, current: null, motion: 'no-preference' },
      { width: 390, current: null, motion: 'no-preference' },
      { width: 1800, current: 'attention.multi-head', motion: 'no-preference' },
      { width: 390, current: 'attention.multi-head', motion: 'no-preference' },
      { width: 1800, current: null, motion: 'reduce' }
    ]) {
      const context = await browser.newContext({
        viewport: { width: scenario.width, height: 1100 }, colorScheme: 'light', reducedMotion: scenario.motion
      });
      try {
        await context.route('**/*', (route) => {
          const url = route.request().url();
          if (!url.startsWith(siteUrl)) return route.abort();
          if (url.endsWith('/learning-data.generated.js')) {
            // 只替换浏览器内的当前任务；不改仓库进度或生成文件。
            return route.fulfill({ contentType: 'text/javascript', body: dataSource
              + '\nwindow.LEARNING_DATA.progress.currentNodeId = ' + JSON.stringify(scenario.current) + ';' });
          }
          return route.continue();
        });
        const page = await context.newPage();
        const errors = [];
        page.on('pageerror', (error) => errors.push(error.message));
        async function assertMapLanding() {
          await page.waitForFunction(() => {
            const map = document.getElementById('map-panel').getBoundingClientRect();
            const header = document.querySelector('.site-header').getBoundingClientRect();
            return Math.abs(map.top - header.bottom) <= 1;
          }, null, { timeout: 3000 });
          assert.equal(await page.locator('#topology-tab').getAttribute('aria-selected'), 'true');
          assert.equal(await page.locator('#map-panel').getAttribute('data-stage-id'), 'all');
          assert.equal(await page.locator('.topology-node.is-selected').count(), 0);
          assert.equal(await page.locator('.detail-panel').isVisible(), false);
          assert.equal(await page.evaluate(() => location.hash), '', '首页不写入知识点锚点');
          assert.equal(await page.locator('.topology-canvas').evaluate((canvas) => canvas.scrollTop), 0);
          assert.equal(await page.evaluate(() => document.querySelector('.learning-brief').getBoundingClientRect().bottom
            <= document.querySelector('.site-header').getBoundingClientRect().bottom), true, '进度卡片不占默认首屏');
        }

        await page.goto(siteUrl + 'index.html');
        await assertMapLanding();
        if (process.env.UI_SCREENSHOT_DIR && scenario.current === null && scenario.motion === 'no-preference') {
          await mkdir(process.env.UI_SCREENSHOT_DIR, { recursive: true });
          await page.screenshot({ path: path.join(process.env.UI_SCREENSHOT_DIR, 'default-landing-' + scenario.width + '.png') });
        }
        // 定位只执行一次；用户向上查看进度时不应被拉回地图。
        await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
        await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        assert.equal(await page.evaluate(() => scrollY), 0);
        assert.equal(await page.locator('.learning-brief').isVisible(), true);
        await page.reload();
        await assertMapLanding();
        // 显式知识点链接仍选中对应节点，不能被默认全局入口覆盖。
        await page.goto(siteUrl + 'index.html#attention.multi-head');
        assert.equal(await page.locator('.topology-node.is-selected').getAttribute('data-node-id'), 'attention.multi-head');
        assert.equal(await page.locator('.detail-panel').isVisible(), true);
        assert.equal(await page.evaluate(() => location.hash), '#attention.multi-head');
        assert.deepEqual(errors, []);
        console.log('通过默认落点：' + JSON.stringify(scenario));
      } finally {
        await context.close();
      }
    }
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
