import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import test from 'node:test';

const repositoryRoot = new URL('../../', import.meta.url);
const map = JSON.parse(await readFile(new URL('learning/map.json', repositoryRoot), 'utf8'));
const html = await readFile(new URL('docs/index.html', repositoryRoot), 'utf8');
const app = await readFile(new URL('docs/assets/app.js', repositoryRoot), 'utf8');
const styles = await readFile(new URL('docs/assets/styles.css', repositoryRoot), 'utf8');
const source = await readFile(new URL('docs/assets/topology.js', repositoryRoot), 'utf8');
const context = {};
vm.runInNewContext(source, context);
const { calculateRanks, collectRelationshipSets, createTopologyLayout } = context.LearningTopology;
const nodes = map.stages.flatMap((stage, stageIndex) => stage.nodes.map((node, nodeIndexInStage) => ({
  ...node, stageId: stage.id, stageTitle: stage.title, stageIndex, nodeIndexInStage, progress: { status: 'pending' }
})));

function assertGeometry(layout) {
  const positions = [...layout.positions.values()];
  for (const left of positions) {
    assert.ok(left.x >= 0 && left.x + left.width <= layout.width + 0.01, left.id + ' 不超出横向画布');
    assert.ok(left.y >= 0 && left.y + left.height <= layout.height + 0.01, left.id + ' 不超出纵向画布');
    assert.ok(left.width >= 180 && left.height >= 80, '节点保留足够空间展示文字');
    for (const right of positions) {
      if (left.id === right.id) continue;
      const overlaps = left.x < right.x + right.width && left.x + left.width > right.x
        && left.y < right.y + right.height && left.y + left.height > right.y;
      assert.equal(overlaps, false, left.id + ' 与 ' + right.id + ' 不重叠');
    }
  }
  for (const edge of layout.edges) {
    const from = layout.positions.get(edge.sourceId);
    const to = layout.positions.get(edge.targetId);
    assert.ok(from.rank < to.rank);
    assert.ok(layout.direction === 'horizontal' ? from.x + from.width < to.x : from.y + from.height < to.y);
  }
}

for (const width of [260, 600, 1100, 1700, 2900]) {
  test(width + 'px 全局图覆盖全部节点/边、按宽度排布，不依赖缩小字体', () => {
    const layout = createTopologyLayout(nodes, map.stages, { width });
    assert.equal(layout.positions.size, nodes.length);
    assert.equal(layout.edges.length, nodes.reduce((total, node) => total + node.prerequisites.length, 0));
    assert.equal(layout.direction, 'vertical');
    assertGeometry(layout);
    const maxRowWidth = Math.max(...layout.levels.map(({ rank }) => {
      const row = [...layout.positions.values()].filter((p) => p.rank === rank);
      return Math.max(...row.map((p) => p.x + p.width)) - Math.min(...row.map((p) => p.x));
    }));
    assert.ok(maxRowWidth >= (width - 40) * 0.9, '最宽依赖层应利用至少 90% 去除内边距后的可用宽度');
  });
}

test('所有阶段重新计算局部依赖层及边，不继承全局偏移', () => {
  for (const stage of map.stages) {
    for (const width of [260, 1100, 1700]) {
      const layout = createTopologyLayout(nodes, map.stages, { width, stageId: stage.id });
      assert.equal(layout.positions.size, stage.nodes.length);
      assert.equal(Math.min(...layout.ranks.values()), 0);
      assert.deepEqual([...layout.positions.keys()].sort(), stage.nodes.map((node) => node.id).sort());
      const ids = new Set(stage.nodes.map((node) => node.id));
      assert.equal(layout.edges.length, stage.nodes.reduce((total, node) => total + node.prerequisites.filter((id) => ids.has(id)).length, 0));
      assertGeometry(layout);
      assert.ok(layout.height < 2200, '阶段图不能残留其他阶段的大块空白');
    }
  }
});

test('宽屏输入管线阶段可在一屏内完整显示', () => {
  const layout = createTopologyLayout(nodes, map.stages, { width: 1700, stageId: 'text-input' });
  assert.equal(layout.direction, 'horizontal');
  assert.ok(layout.height < 500);
});

test('循环依赖和缺失先修被明确拒绝', () => {
  assert.throws(() => calculateRanks([{ id: 'a', prerequisites: ['b'] }, { id: 'b', prerequisites: ['a'] }]), /存在环/);
  assert.throws(() => calculateRanks([{ id: 'a', prerequisites: ['missing'] }]), /不存在/);
});

test('取消选择时关系集合为空，选中时仍可查看完整前驱和后继', () => {
  const empty = collectRelationshipSets(null, nodes);
  for (const set of Object.values(empty)) assert.equal(set.size, 0);
  const relationships = collectRelationshipSets('attention.single-head', nodes);
  assert.ok(relationships.ancestors.has('foundation.softmax'));
  assert.ok(relationships.directDependents.has('attention.multi-head'));
  assert.ok(relationships.descendants.has('mastery.capstone'));
});

test('默认依赖拓扑、可滚动画布及标题不隐藏', () => {
  assert.ok(html.indexOf('assets/topology.js') < html.indexOf('assets/app.js'));
  assert.match(html, /id="topology-tab"[^>]*aria-selected="true"/);
  assert.doesNotMatch(styles, /is-overview[^}]*opacity:\s*0/);
  assert.match(styles, /\.topology-canvas\s*\{[^}]*overflow:\s*auto/);
  for (const id of ['topology-panel', 'stages-panel', 'learning-topology', 'topology-fit', 'topology-current', 'topology-fullscreen']) {
    assert.equal(html.match(new RegExp('id="' + id + '"', 'g'))?.length, 1);
  }
});

test('全屏保持主动触发及沉浸式降级能力', () => {
  assert.match(app, /requestFullscreen/);
  assert.match(app, /fullscreenchange/);
  assert.match(app, /is-pseudo-fullscreen/);
});
