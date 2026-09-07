import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import test from 'node:test';

const repositoryRoot = new URL('../../', import.meta.url);
const map = JSON.parse(await readFile(new URL('learning/map.json', repositoryRoot), 'utf8'));
const html = await readFile(new URL('docs/index.html', repositoryRoot), 'utf8');
const appSource = await readFile(new URL('docs/assets/app.js', repositoryRoot), 'utf8');
const source = await readFile(new URL('docs/assets/topology.js', repositoryRoot), 'utf8');
const context = {};
vm.runInNewContext(source, context);
const { calculateRanks, collectRelationshipSets, createTopologyLayout } = context.LearningTopology;

const nodes = map.stages.flatMap((stage, stageIndex) =>
  stage.nodes.map((node, nodeIndexInStage) => ({
    ...node,
    stageId: stage.id,
    stageTitle: stage.title,
    stageIndex,
    nodeIndexInStage,
    progress: { status: 'pending' }
  }))
);

test('每条依赖边都严格从较浅层指向较深层', () => {
  const ranks = calculateRanks(nodes);
  for (const node of nodes) {
    for (const prerequisiteId of node.prerequisites) {
      assert.ok(
        ranks.get(prerequisiteId) < ranks.get(node.id),
        `${prerequisiteId} 应位于 ${node.id} 左侧`
      );
    }
  }
});

test('布局覆盖全部节点和边，且节点卡片不重叠', () => {
  const layout = createTopologyLayout(nodes, map.stages);
  assert.equal(layout.positions.size, nodes.length);
  assert.equal(layout.edges.length, nodes.reduce((total, node) => total + node.prerequisites.length, 0));
  assert.equal(layout.lanes.length, map.stages.length);

  const positions = [...layout.positions.values()];
  for (let leftIndex = 0; leftIndex < positions.length; leftIndex += 1) {
    const left = positions[leftIndex];
    for (let rightIndex = leftIndex + 1; rightIndex < positions.length; rightIndex += 1) {
      const right = positions[rightIndex];
      const overlaps = left.x < right.x + right.width
        && left.x + left.width > right.x
        && left.y < right.y + right.height
        && left.y + left.height > right.y;
      assert.equal(overlaps, false, `${left.id} 与 ${right.id} 不应重叠`);
    }
  }
});

test('选中节点时能收集完整的前驱、后继和直接关系', () => {
  const relationships = collectRelationshipSets('attention.single-head', nodes);
  assert.ok(relationships.directPrerequisites.has('attention.mask'));
  assert.ok(relationships.ancestors.has('foundation.softmax'));
  assert.ok(relationships.directDependents.has('attention.multi-head'));
  assert.ok(relationships.descendants.has('mastery.capstone'));
  assert.equal(relationships.ancestors.has('attention.single-head'), false);
});

test('循环依赖会被明确拒绝', () => {
  assert.throws(
    () => calculateRanks([
      { id: 'a', prerequisites: ['b'] },
      { id: 'b', prerequisites: ['a'] }
    ]),
    /存在环/
  );
});

test('页面在主应用之前加载拓扑模块，且交互挂载点唯一', () => {
  assert.ok(html.indexOf('assets/topology.js') < html.indexOf('assets/app.js'));
  for (const id of ['topology-panel', 'stages-panel', 'learning-topology', 'topology-fit', 'topology-fullscreen']) {
    assert.equal(html.match(new RegExp(`id="${id}"`, 'g'))?.length, 1, `${id} 应唯一`);
  }
});

test('全屏控制支持原生 API、状态同步和沉浸式降级', () => {
  assert.match(appSource, /requestFullscreen/);
  assert.match(appSource, /fullscreenchange/);
  assert.match(appSource, /is-pseudo-fullscreen/);
  assert.match(appSource, /event\.key === 'Escape' && pseudoFullscreen/);
});

test('拓扑渲染器可创建完整 SVG，并执行选中、筛选和定位', () => {
  class FakeClassList {
    constructor(element) { this.element = element; }
    values() { return new Set(this.element.className.split(/\s+/).filter(Boolean)); }
    write(values) { this.element.className = [...values].join(' '); }
    add(...names) { const values = this.values(); names.forEach((name) => values.add(name)); this.write(values); }
    remove(...names) { const values = this.values(); names.forEach((name) => values.delete(name)); this.write(values); }
    contains(name) { return this.values().has(name); }
    toggle(name, force) {
      const values = this.values();
      const shouldAdd = force === undefined ? !values.has(name) : force;
      if (shouldAdd) values.add(name);
      else values.delete(name);
      this.write(values);
      return shouldAdd;
    }
  }

  class FakeElement {
    constructor(name) {
      this.nodeName = name;
      this.attributes = new Map();
      this.children = [];
      this.parentElement = null;
      this.className = '';
      this.classList = new FakeClassList(this);
      this.listeners = new Map();
      this.textContent = '';
    }
    setAttribute(name, value) {
      this.attributes.set(name, String(value));
      if (name === 'class') this.className = String(value);
    }
    getAttribute(name) { return this.attributes.get(name); }
    append(...children) { children.forEach((child) => this.appendChild(child)); }
    appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
    replaceChildren(...children) { this.children = []; this.append(...children); }
    addEventListener(name, listener) { this.listeners.set(name, listener); }
    getBoundingClientRect() { return { left: 0, top: 0, width: 1000, height: 700 }; }
    setPointerCapture() {}
    focus() { this.isFocused = true; }
    closest(selector) {
      const className = selector.startsWith('.') ? selector.slice(1) : null;
      let current = this;
      while (current) {
        if (className && current.classList.contains(className)) return current;
        current = current.parentElement;
      }
      return null;
    }
  }

  context.document = {
    createElement: (name) => new FakeElement(name),
    createElementNS: (_namespace, name) => new FakeElement(name)
  };
  context.requestAnimationFrame = (callback) => { callback(); return 1; };
  context.cancelAnimationFrame = () => {};

  const container = new FakeElement('div');
  const testStages = map.stages.map((stage) => ({
    ...stage,
    nodes: stage.nodes.map((node) => ({ ...node, progress: { status: 'pending' } }))
  }));
  let lastScale = 0;
  const graph = context.LearningTopology.createTopologyGraph({
    container,
    nodes,
    stages: testStages,
    statusMeta: {
      mastered: { label: '已掌握', shortLabel: '掌握' },
      current: { label: '当前学习', shortLabel: '当前' },
      verify: { label: '待验证', shortLabel: '验证' },
      relearn: { label: '待重学', shortLabel: '重学' },
      pending: { label: '未开始', shortLabel: '未开始' }
    },
    onSelect: () => {},
    onTransform: ({ scale }) => { lastScale = scale; }
  });

  assert.equal(container.children.length, 1);
  assert.ok(lastScale > 0);
  assert.doesNotThrow(() => graph.selectNode('attention.single-head'));
  assert.equal(graph.setFilter({ query: 'softmax', status: 'all' }), 5);
  assert.doesNotThrow(() => graph.focusNode('foundation.softmax'));
});
