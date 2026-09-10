// 使用独立临时浏览器验证真实 DOM；运行方式见本目录 README.md。
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { mkdir } = require('node:fs/promises');
const { chromium } = require('playwright');

async function main() {
  const siteUrl = pathToFileURL(`${path.resolve(__dirname, '../../docs')}${path.sep}`).href;
  const browser = await chromium.launch({
    executablePath: process.env.BROWSER_EXECUTABLE_PATH || undefined,
    headless: true,
    ignoreDefaultArgs: ['--headless=old'],
    args: ['--headless=new', '--disable-background-networking']
  });
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 }, colorScheme: 'light', reducedMotion: 'reduce'
    });
    // 只加载项目静态页面，不读取已有浏览器资料，也不访问外网。
    await context.route('**/*', (route) => route.request().url().startsWith(siteUrl)
      ? route.continue() : route.abort());
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(`${siteUrl}index.html`);
    await page.locator('.stage-summary-card.is-active').waitFor();
    const data = await page.evaluate(() => window.LEARNING_DATA);
    const nodes = data.stages.flatMap((stage) => stage.nodes.map((node) => ({ ...node, stageId: stage.id })));
    const current = nodes.find((node) => node.id === data.progress.currentNodeId);
    const sidebar = (id) => page.locator(`[data-stage-target="${id}"]`);

    async function assertStage(stage) {
      assert.equal(await page.locator('.stage-summary-card.is-active').count(), 1);
      assert.equal(await page.locator('.stage-summary-card.is-active').getAttribute('data-stage-target'), stage.id);
      assert.equal(await sidebar(stage.id).getAttribute('aria-pressed'), 'true');
      assert.equal(await page.locator('#map-panel').getAttribute('data-stage-id'), stage.id);
      assert.equal(await page.locator('#map-heading').innerText(), stage.title.split('：')[0]);
      assert.equal(await page.locator('#visible-count').innerText(), `共 ${stage.nodes.length} 个知识点`);
      assert.equal(await page.locator('[data-status-filter="all"] span').innerText(), String(stage.nodes.length));
      const shownSections = page.locator('.stage:not(.is-filtered-out)');
      assert.equal(await shownSections.count(), 1, '阶段切换必须更换节点集合，而不只是滚动');
      assert.equal(await shownSections.getAttribute('data-stage-id'), stage.id);
      const graphIds = await page.locator('.topology-node:not(.is-outside-stage)').evaluateAll((items) => items.map((item) => item.dataset.nodeId));
      assert.deepEqual(graphIds.sort(), stage.nodes.map((node) => node.id).sort());
      const selectedId = await page.locator('.topology-node.is-selected').getAttribute('data-node-id');
      const selected = stage.nodes.find((node) => node.id === selectedId);
      assert.ok(selected, '右侧选中节点必须属于当前浏览阶段');
      assert.equal(await page.locator('#detail-title').innerText(), selected.title);
      assert.equal(await page.locator('#current-node').innerText(), current.title, '浏览不能改变当前学习任务');
    }

    for (const view of ['topology', 'stages']) {
      await page.locator(`#${view}-tab`).click();
      for (const stage of data.stages) {
        await sidebar(stage.id).click();
        await assertStage(stage);
        assert.equal(await page.locator(`#${view}-tab`).getAttribute('aria-selected'), 'true');
      }
    }

    // 搜索与状态计数限制在所选阶段，清除筛选仍保留浏览范围。
    const firstStage = data.stages[0];
    await sidebar(firstStage.id).click();
    await page.locator('#search-input').fill(firstStage.nodes[0].title);
    assert.equal(await page.locator('#map-panel').getAttribute('data-stage-id'), firstStage.id);
    await page.locator('#search-input').fill('不存在的测试知识点');
    assert.equal(await page.locator('#filter-empty').isVisible(), true);
    assert.equal(await page.locator('.detail-panel').isVisible(), false);
    await page.locator('#filter-empty button').click();
    await assertStage(firstStage);

    // 记住每个阶段上次浏览的节点。
    const rememberedNode = firstStage.nodes[1];
    await page.locator(`.node-button[data-node-id="${rememberedNode.id}"]`).click();
    await sidebar(data.stages[1].id).click();
    await sidebar(firstStage.id).click();
    assert.equal(await page.locator('#detail-title').innerText(), rememberedNode.title);

    // 全局入口、学习任务入口以及跨阶段先修要同步导航。
    await sidebar('all').click();
    assert.equal(await page.locator('.stage:not(.is-filtered-out)').count(), data.stages.length);
    assert.equal(await page.locator('#visible-count').innerText(), `共 ${nodes.length} 个知识点`);
    await page.locator('#current-detail-button').click();
    await assertStage(data.stages.find((stage) => stage.id === current.stageId));
    await page.locator('#topology-fit').click();
    assert.equal(await sidebar('all').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('.topology-node:not(.is-outside-stage)').count(), nodes.length);
    await page.locator('#topology-current').click();
    await assertStage(data.stages.find((stage) => stage.id === current.stageId));
    const crossStageSource = nodes.find((source) => source.prerequisites.some((id) =>
      nodes.find((node) => node.id === id).stageId !== source.stageId));
    assert.ok(crossStageSource, '课程提供跨阶段先修用例');
    await page.locator('#stages-tab').click();
    await sidebar(crossStageSource.stageId).click();
    await page.locator(`.node-button[data-node-id="${crossStageSource.id}"]`).click();
    const crossStagePrerequisite = crossStageSource.prerequisites.map((id) => nodes.find((node) => node.id === id))
      .find((node) => node.stageId !== crossStageSource.stageId);
    await page.locator('#detail-prerequisites .dependency-button').filter({ hasText: crossStagePrerequisite.title }).click();
    await assertStage(data.stages.find((stage) => stage.id === crossStagePrerequisite.stageId));
    assert.equal(await page.locator('#detail-title').innerText(), crossStagePrerequisite.title);
    await page.reload();
    await page.locator('.stage-summary-card.is-active').waitFor();
    await assertStage(data.stages.find((stage) => stage.id === crossStagePrerequisite.stageId));

    // 顶部待巩固统计是全局入口，不受阶段范围影响。
    await page.locator('[data-quick-filter="verify"]').click();
    assert.equal(await sidebar('all').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('[data-status-filter="verify"] span').innerText(), String(data.progress.statusCounts.verify));

    if (process.env.UI_SCREENSHOT_DIR) {
      await mkdir(process.env.UI_SCREENSHOT_DIR, { recursive: true });
      await page.locator('#stages-tab').click();
      for (const stage of [firstStage, data.stages.at(-1)]) {
        await sidebar(stage.id).click();
        await page.screenshot({ path: path.join(process.env.UI_SCREENSHOT_DIR, `${stage.id}.png`) });
      }
    }

    await page.setViewportSize({ width: 390, height: 844 });
    await sidebar(firstStage.id).click();
    await assertStage(firstStage);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await sidebar(data.stages.at(-1).id).click();
    await assertStage(data.stages.at(-1));
    for (const width of [320, 768, 1280, 2000]) {
      await page.setViewportSize({ width, height: 1000 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `${width}px 阶段标题和导航不应撑宽页面`);
    }
    assert.deepEqual(await page.evaluate(() => window.LEARNING_DATA.progress), data.progress);
    assert.deepEqual(errors, []);
    console.log(`通过：${data.stages.length} 个阶段 × 2 种视图，范围筛选、节点记忆、跨阶段依赖、深链接、全局入口和手机导航。`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
