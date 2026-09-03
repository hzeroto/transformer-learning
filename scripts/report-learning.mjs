import crypto from 'node:crypto';
import {
  appendRecord,
  buildGeneratedData,
  buildNodeIndex,
  loadState,
  paths,
  statusOf,
  validateState,
  writeJsonAtomic
} from './learning-lib.mjs';

const commandToStatus = {
  master: 'mastered',
  current: 'current',
  verify: 'verify',
  relearn: 'relearn',
  pending: 'pending'
};

function usage(message) {
  if (message) console.error(`错误：${message}\n`);
  console.error('用法：');
  console.error('  node scripts/report-learning.mjs master <知识点ID> --evidence "掌握证据" [--note "备注"]');
  console.error('  node scripts/report-learning.mjs current|verify|relearn|pending <知识点ID> [--note "备注"]');
  process.exit(1);
}

function parseFlags(args) {
  const flags = {};
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (!argument.startsWith('--')) usage(`无法识别的参数 ${argument}`);
    const equalIndex = argument.indexOf('=');
    if (equalIndex !== -1) {
      flags[argument.slice(2, equalIndex)] = argument.slice(equalIndex + 1);
      continue;
    }
    const key = argument.slice(2);
    const value = args[index + 1];
    if (!value || value.startsWith('--')) usage(`${argument} 缺少值`);
    flags[key] = value;
    index += 1;
  }
  return flags;
}

const [command, nodeId, ...rest] = process.argv.slice(2);
if (!commandToStatus[command]) usage(`未知操作 ${command ?? ''}`);
if (!nodeId) usage('缺少知识点ID');
const flags = parseFlags(rest);
const nextStatus = commandToStatus[command];
if (nextStatus === 'mastered' && !flags.evidence?.trim()) {
  usage('标记掌握时必须通过 --evidence 提供可核验的掌握证据');
}

const { map, progress, records } = loadState();
const nodeIndex = validateState(map, progress, records);
const node = nodeIndex.get(nodeId);
if (!node) usage(`知识地图中不存在 ${nodeId}`);

const now = new Date().toISOString();
if (nextStatus === 'current' && progress.currentNodeId && progress.currentNodeId !== nodeId) {
  const previousCurrentId = progress.currentNodeId;
  const previousCurrent = nodeIndex.get(previousCurrentId);
  const previousStatus = statusOf(progress, previousCurrentId);
  progress.nodes[previousCurrentId] = {
    ...progress.nodes[previousCurrentId],
    status: 'verify',
    updatedAt: now,
    note: `学习焦点切换到 ${node.title}，原知识点保留为待验证`
  };
  appendRecord({
    id: crypto.randomUUID(),
    at: now,
    nodeId: previousCurrentId,
    nodeTitle: previousCurrent.title,
    action: 'focus-shift',
    fromStatus: previousStatus,
    toStatus: 'verify',
    evidence: null,
    note: `学习焦点切换到 ${node.title}`
  });
}

const previousStatus = statusOf(progress, nodeId);
const evidenceList = progress.nodes[nodeId]?.evidence ?? [];
if (flags.evidence?.trim()) {
  evidenceList.push({ at: now, text: flags.evidence.trim() });
}
progress.nodes[nodeId] = {
  ...progress.nodes[nodeId],
  status: nextStatus,
  updatedAt: now,
  note: flags.note?.trim() || progress.nodes[nodeId]?.note || null,
  evidence: evidenceList
};
progress.currentNodeId = nextStatus === 'current'
  ? nodeId
  : progress.currentNodeId === nodeId
    ? null
    : progress.currentNodeId;
progress.updatedAt = now;

writeJsonAtomic(paths.progress, progress);
appendRecord({
  id: crypto.randomUUID(),
  at: now,
  nodeId,
  nodeTitle: node.title,
  action: command,
  fromStatus: previousStatus,
  toStatus: nextStatus,
  evidence: flags.evidence?.trim() || null,
  note: flags.note?.trim() || null
});

const snapshot = buildGeneratedData();
console.log(JSON.stringify({
  ok: true,
  nodeId,
  nodeTitle: node.title,
  fromStatus: previousStatus,
  toStatus: nextStatus,
  currentNodeId: snapshot.progress.currentNodeId,
  mastered: snapshot.progress.statusCounts.mastered,
  total: snapshot.progress.totalNodes
}, null, 2));
