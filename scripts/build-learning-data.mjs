import { buildGeneratedData } from './learning-lib.mjs';

const snapshot = buildGeneratedData();
console.log(`已生成前端数据：${snapshot.progress.totalNodes} 个知识点，${snapshot.records.length} 条记录`);
