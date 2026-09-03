import { buildGeneratedData, loadState, validateState } from './learning-lib.mjs';

const { map, progress, records } = loadState();
const index = validateState(map, progress, records);
const snapshot = buildGeneratedData();
console.log(JSON.stringify({
  ok: true,
  stages: map.stages.length,
  nodes: index.size,
  records: records.length,
  mastered: snapshot.progress.statusCounts.mastered,
  currentNodeId: snapshot.progress.currentNodeId
}, null, 2));
