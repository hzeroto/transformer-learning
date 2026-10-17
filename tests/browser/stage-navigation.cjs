// 使用独立临时浏览器验证真实 DOM；运行方式见本目录 README.md。
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { mkdir } = require('node:fs/promises');
const { chromium } = require('playwright');

async function main() {
  const siteUrl = pathToFileURL(path.resolve(__dirname, '../../docs') + path.sep).href;
  const browser = await chromium.launch({
    executablePath: process.env.BROWSER_EXECUTABLE_PATH || undefined,
    headless: true, args: ['--disable-background-networking']
  });
  try {
    const context = await browser.newContext({ viewport: { width: 1800, height: 1100 }, colorScheme: 'light', reducedMotion: 'reduce' });
    await context.route('**/*', (route) => route.request().url().startsWith(siteUrl) ? route.continue() : route.abort());
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(siteUrl + 'index.html');
    const data = await page.evaluate(() => window.LEARNING_DATA);
    const nodes = data.stages.flatMap((stage) => stage.nodes.map((node) => ({ ...node, stageId: stage.id })));
    const sidebar = (id) => page.locator('[data-stage-target="' + id + '"]');
    const graphNode = (id) => page.locator('.topology-node[data-node-id="' + id + '"]');
    const listNode = (id) => page.locator('.node-button[data-node-id="' + id + '"]');
    const currentTitle = await page.locator('#current-node').innerText();
    if (process.env.UI_SCREENSHOT_DIR) await mkdir(process.env.UI_SCREENSHOT_DIR, { recursive: true });
    async function screenshot(name) {
      if (process.env.UI_SCREENSHOT_DIR) await page.locator('#map-panel').screenshot({ path: path.join(process.env.UI_SCREENSHOT_DIR, name + '.png') });
    }
    async function assertNoSelection() {
      assert.equal(await page.locator('.topology-node.is-selected').count(), 0);
      assert.equal(await page.locator('.node-button[aria-pressed="true"]').count(), 0);
      assert.equal(await page.locator('.detail-panel').isVisible(), false);
      assert.equal(await page.evaluate(() => location.hash), '');
    }
    async function assertReadableGraph(expectedCount) {
      // 允许纵向滚动；验证的是所有文字的实际屏幕字号和可达性，而不是把整图压进一屏。
      await page.waitForFunction((count) => {
        const canvas = document.querySelector('.topology-canvas');
        const svg = document.querySelector('.topology-svg').getBoundingClientRect();
        const groups = [...document.querySelectorAll('.topology-node[aria-hidden="false"]')];
        return canvas.clientWidth > 0 && svg.width <= canvas.clientWidth + 1 && groups.length === count && groups.every((group) => {
          const card = group.querySelector('.topology-node-card').getBoundingClientRect();
          return card.left >= svg.left - 1 && card.right <= svg.right + 1 && card.top >= svg.top - 1 && card.bottom <= svg.bottom + 1;
        });
      }, expectedCount);
      const result = await page.evaluate(() => {
        const canvas = document.querySelector('.topology-canvas');
        const svg = document.querySelector('.topology-svg').getBoundingClientRect();
        const labels = [...document.querySelectorAll('.topology-node[aria-hidden="false"]')].map((group) => {
          const label = group.querySelector('.topology-node-title');
          const rect = label.getBoundingClientRect();
          const card = group.querySelector('.topology-node-card').getBoundingClientRect();
          const style = getComputedStyle(label);
          const matrix = label.getScreenCTM();
          return { id: group.dataset.nodeId, text: label.textContent, size: parseFloat(style.fontSize) * Math.hypot(matrix.a, matrix.b), opacity: style.opacity,
            fits: rect.left >= card.left && rect.right <= card.right && rect.bottom <= group.querySelector('.topology-node-meta').getBoundingClientRect().top - 2 };
        });
        return { labels, width: canvas.clientWidth, svgWidth: svg.width, scrollLeft: canvas.scrollLeft };
      });
      assert.ok(result.svgWidth <= result.width + 1, '默认排布应适配宽度，而非横向裁切');
      for (const label of result.labels) {
        assert.ok(label.size >= 12.9, label.id + ' 字号至少 13px');
        assert.notEqual(label.opacity, '0', label.id + ' 标题不能被隐藏');
        assert.equal(label.text, nodes.find((node) => node.id === label.id).title, '标题不得省略为看不清的缩略块');
        assert.ok(label.fits, label.id + ' 标题须完整位于卡片内且不挤到元信息');
      }
    }
    async function assertScope(stage, selectedId = null) {
      const id = stage?.id || 'all';
      const expected = stage?.nodes || nodes;
      assert.equal(await sidebar(id).getAttribute('aria-pressed'), 'true');
      assert.equal(await page.locator('#map-panel').getAttribute('data-stage-id'), id);
      assert.equal(await page.locator('#visible-count').innerText(), '共 ' + expected.length + ' 个知识点');
      assert.equal(await page.locator('#current-node').innerText(), currentTitle, '浏览不改变学习任务');
      if (selectedId) {
        assert.equal(await page.locator('.topology-node.is-selected').getAttribute('data-node-id'), selectedId);
        assert.equal(await page.locator('#detail-title').innerText(), nodes.find((node) => node.id === selectedId).title);
      } else await assertNoSelection();
      if (await page.locator('#topology-panel').isVisible()) {
        await assertReadableGraph(expected.length);
        const ids = await page.locator('.topology-node[aria-hidden="false"]').evaluateAll((items) => items.map((item) => item.dataset.nodeId));
        assert.deepEqual(ids.sort(), expected.map((node) => node.id).sort());
      }
    }

    assert.equal(await page.locator('#topology-tab').getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('#topology-fullscreen').getAttribute('aria-pressed'), 'false');
    await assertScope(null);
    await screenshot('global-top');
    const canvas = page.locator('.topology-canvas');
    await canvas.evaluate((element) => { element.scrollTop = element.scrollHeight; });
    await screenshot('global-bottom');
    assert.ok(await canvas.evaluate((element) => element.scrollTop > 0), '全局底部能通过正常滚动到达');
    await page.locator('#topology-fit').click();
    assert.equal(await canvas.evaluate((element) => element.scrollTop), 0);
    await page.locator('#topology-fullscreen').click();
    assert.equal(await page.locator('#topology-fullscreen').getAttribute('aria-pressed'), 'true');
    await assertScope(null);
    await page.locator('#topology-fullscreen').click();
    await assertScope(null);

    // 所有阶段入口都展示阶段全景，不自动选择首节点/当前节点/上次节点。
    for (const view of ['topology', 'stages']) {
      await page.locator('#' + view + '-tab').click();
      for (const stage of data.stages) {
        await sidebar(stage.id).click();
        await assertScope(stage);
      }
    }
    const stage = data.stages.find((item) => item.id === 'text-input');
    const node = stage.nodes[1];
    await page.locator('#topology-tab').click();
    await sidebar(stage.id).click();
    await screenshot('input-pipeline');
    const stageFits = await canvas.evaluate((element) => element.scrollHeight <= element.clientHeight + 1);
    assert.equal(stageFits, true, '宽屏输入管线阶段应能完整显示，不锚定单个节点');
    assert.ok(await canvas.evaluate((element) => element.clientHeight < 500), '短阶段图不应保留大片空白高度');
    await graphNode(node.id).click();
    assert.equal(await page.locator('#detail-title').innerText(), node.title);
    await page.locator('#topology-fit').click();
    await assertScope(stage);
    await graphNode(node.id).focus();
    await page.keyboard.press('Enter');
    assert.equal(await page.locator('#detail-title').innerText(), node.title);
    await sidebar(data.stages[0].id).click();
    await sidebar(stage.id).click();
    await assertScope(stage);
    await page.locator('#topology-zoom-in').click();
    assert.equal(await page.locator('#topology-scale').innerText(), '120%');
    await page.locator('#topology-zoom-out').click();
    assert.equal(await page.locator('#topology-scale').innerText(), '100%');
    await page.locator('#topology-fit').click();
    await assertScope(stage);
    await page.locator('#stages-tab').click();
    await sidebar(data.stages.at(-1).id).click();
    await page.locator('#topology-tab').click();
    await assertScope(data.stages.at(-1));

    // 搜索不自动选择某个结果；清除搜索保留阶段。
    await sidebar(stage.id).click();
    await page.locator('#search-input').fill(stage.nodes[0].title);
    await assertNoSelection();
    await page.locator('#search-input').fill('不存在的测试知识点');
    assert.equal(await page.locator('#topology-empty').isVisible(), true);
    await page.locator('#topology-empty button').click();
    await assertScope(stage);

    // 显式节点链接/跨阶段先修仍允许定位，但返回阶段入口必须解除定位。
    const crossSource = nodes.find((item) => item.prerequisites.some((id) => nodes.find((p) => p.id === id).stageId !== item.stageId));
    const prerequisite = crossSource.prerequisites.map((id) => nodes.find((item) => item.id === id)).find((item) => item.stageId !== crossSource.stageId);
    await page.locator('#stages-tab').click();
    await sidebar(crossSource.stageId).click();
    await listNode(crossSource.id).click();
    await page.locator('#detail-prerequisites .dependency-button').filter({ hasText: prerequisite.title }).click();
    await assertScope(data.stages.find((item) => item.id === prerequisite.stageId), prerequisite.id);
    await page.reload();
    await assertScope(data.stages.find((item) => item.id === prerequisite.stageId), prerequisite.id);
    await sidebar(prerequisite.stageId).click();
    await assertScope(data.stages.find((item) => item.id === prerequisite.stageId));

    for (const width of [320, 390, 768, 1440, 2000, 3400]) {
      await page.setViewportSize({ width, height: 1100 });
      await sidebar('all').click();
      await assertScope(null);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, width + 'px 页面不应横向溢出');
      if ([390, 1440, 3400].includes(width)) await screenshot('global-' + width);
      await sidebar(stage.id).click();
      await assertScope(stage);
      if ([390, 1440].includes(width)) await screenshot('stage-' + width);
    }
    for (const hash of ['unknown-node', '%E0%A4%A']) {
      await page.goto(siteUrl + 'index.html#' + hash);
      await assertScope(null);
    }
    assert.deepEqual(await page.evaluate(() => window.LEARNING_DATA.progress), data.progress);
    assert.deepEqual(errors, []);
    console.log('通过：全局标题至少 13px 且完整可读、正常滚动、10 个阶段 × 2 视图无自动选中、阶段全图、搜索、键盘选择、缩放、全屏、深链接及 6 种屏宽。');
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
