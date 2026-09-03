import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
export const projectRoot = path.resolve(scriptDirectory, '..');
export const paths = {
  map: path.join(projectRoot, 'learning', 'map.json'),
  progress: path.join(projectRoot, 'learning', 'progress.json'),
  records: path.join(projectRoot, 'learning', 'records.jsonl'),
  generated: path.join(projectRoot, 'docs', 'assets', 'learning-data.generated.js'),
  humanRecord: path.join(projectRoot, 'docs', 'learn_record.md')
};

export const allowedStatuses = new Set([
  'mastered',
  'current',
  'verify',
  'relearn',
  'pending'
]);

export function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

export function loadState() {
  const map = readJson(paths.map);
  const progress = readJson(paths.progress);
  const records = fs.readFileSync(paths.records, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`学习记录第 ${index + 1} 行不是合法JSON：${error.message}`);
      }
    });

  return { map, progress, records };
}

export function buildNodeIndex(map) {
  const index = new Map();
  for (const stage of map.stages) {
    for (const node of stage.nodes) {
      if (index.has(node.id)) {
        throw new Error(`知识点ID重复：${node.id}`);
      }
      index.set(node.id, { ...node, stageId: stage.id, stageTitle: stage.title });
    }
  }
  return index;
}

export function statusOf(progress, nodeId) {
  return progress.nodes[nodeId]?.status ?? 'pending';
}

export function validateState(map, progress, records) {
  if (map.schemaVersion !== 1 || progress.schemaVersion !== 1) {
    throw new Error('不支持的学习数据schema版本');
  }

  const index = buildNodeIndex(map);
  for (const node of index.values()) {
    for (const dependencyId of node.prerequisites) {
      if (!index.has(dependencyId)) {
        throw new Error(`${node.id} 引用了不存在的先修知识点 ${dependencyId}`);
      }
    }
  }

  for (const [nodeId, nodeProgress] of Object.entries(progress.nodes)) {
    if (!index.has(nodeId)) {
      throw new Error(`学习进度引用了不存在的知识点：${nodeId}`);
    }
    if (!allowedStatuses.has(nodeProgress.status)) {
      throw new Error(`${nodeId} 使用了非法状态：${nodeProgress.status}`);
    }
  }

  const currentNodes = [...index.keys()].filter(
    (nodeId) => statusOf(progress, nodeId) === 'current'
  );
  if (currentNodes.length > 1) {
    throw new Error(`只能有一个当前知识点，实际发现：${currentNodes.join(', ')}`);
  }
  if (progress.currentNodeId !== (currentNodes[0] ?? null)) {
    throw new Error('currentNodeId与节点状态不一致');
  }

  for (const record of records) {
    if (!index.has(record.nodeId)) {
      throw new Error(`学习记录引用了不存在的知识点：${record.nodeId}`);
    }
    if (!allowedStatuses.has(record.toStatus)) {
      throw new Error(`学习记录包含非法状态：${record.toStatus}`);
    }
    if (record.toStatus === 'mastered' && !record.evidence) {
      throw new Error(`掌握记录必须提供证据：${record.nodeId}`);
    }
  }

  const visiting = new Set();
  const visited = new Set();
  function visit(nodeId) {
    if (visiting.has(nodeId)) throw new Error(`知识依赖存在环：${nodeId}`);
    if (visited.has(nodeId)) return;
    visiting.add(nodeId);
    for (const dependencyId of index.get(nodeId).prerequisites) visit(dependencyId);
    visiting.delete(nodeId);
    visited.add(nodeId);
  }
  for (const nodeId of index.keys()) visit(nodeId);

  return index;
}

export function writeJsonAtomic(filePath, value) {
  const temporaryPath = `${filePath}.tmp`;
  fs.writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`);
  fs.renameSync(temporaryPath, filePath);
}

export function appendRecord(record) {
  fs.appendFileSync(paths.records, `${JSON.stringify(record)}\n`);
}

export function buildGeneratedData() {
  const { map, progress, records } = loadState();
  const index = validateState(map, progress, records);
  const stages = map.stages.map((stage) => ({
    ...stage,
    nodes: stage.nodes.map((node) => ({
      ...node,
      progress: progress.nodes[node.id] ?? { status: 'pending' }
    }))
  }));

  const statusCounts = {
    mastered: 0,
    current: 0,
    verify: 0,
    relearn: 0,
    pending: 0
  };
  for (const nodeId of index.keys()) statusCounts[statusOf(progress, nodeId)] += 1;

  const snapshot = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    goal: map.goal,
    stages,
    progress: {
      ...progress,
      statusCounts,
      totalNodes: index.size
    },
    records: [...records].reverse()
  };

  const javascript = `window.LEARNING_DATA = ${JSON.stringify(snapshot, null, 2)};\n`;
  const temporaryPath = `${paths.generated}.tmp`;
  fs.writeFileSync(temporaryPath, javascript);
  fs.renameSync(temporaryPath, paths.generated);

  const statusLabels = {
    mastered: '已掌握',
    current: '当前学习',
    verify: '待验证',
    relearn: '待系统重学',
    pending: '未开始'
  };
  const escapeCell = (value) => String(value ?? '')
    .replaceAll('|', '\\|')
    .replaceAll('\n', ' ');
  const currentNode = progress.currentNodeId ? index.get(progress.currentNodeId) : null;
  const markdown = [
    '# Transformer 学习记录',
    '',
    '> 本文件由学习上报脚本自动生成，请勿手工编辑。完整依赖见 `learning/map.json`，追加事件见 `learning/records.jsonl`。',
    '',
    '## 当前状态',
    '',
    `- 当前知识点：${currentNode?.title ?? '尚未指定'}`,
    `- 已掌握：${statusCounts.mastered} / ${index.size}`,
    `- 待验证：${statusCounts.verify}`,
    `- 待系统重学：${statusCounts.relearn}`,
    `- 最近更新：${progress.updatedAt ?? '尚未更新'}`,
    '',
    '## 知识点状态',
    ''
  ];

  for (const stage of stages) {
    markdown.push(`### ${stage.title}`, '', '| 知识点 | 状态 | 最近证据或备注 |', '|---|---|---|');
    for (const node of stage.nodes) {
      const latestEvidence = node.progress.evidence?.at(-1)?.text;
      markdown.push(`| ${escapeCell(node.title)} | ${statusLabels[node.progress.status]} | ${escapeCell(latestEvidence || node.progress.note || '—')} |`);
    }
    markdown.push('');
  }

  markdown.push('## 学习事件', '');
  if (records.length === 0) {
    markdown.push('暂无记录。');
  } else {
    markdown.push('| 时间 | 知识点 | 状态变化 | 证据或备注 |', '|---|---|---|---|');
    for (const record of [...records].reverse()) {
      markdown.push(`| ${escapeCell(record.at)} | ${escapeCell(record.nodeTitle)} | ${statusLabels[record.fromStatus]} → ${statusLabels[record.toStatus]} | ${escapeCell(record.evidence || record.note || '—')} |`);
    }
  }
  const humanTemporaryPath = `${paths.humanRecord}.tmp`;
  fs.writeFileSync(humanTemporaryPath, `${markdown.join('\n')}\n`);
  fs.renameSync(humanTemporaryPath, paths.humanRecord);
  return snapshot;
}
