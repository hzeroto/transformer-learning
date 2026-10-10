import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const root = dirname(fileURLToPath(import.meta.url));
const c = {
  ink: '#172b40',
  muted: '#526477',
  line: '#c8d3df',
  blue: '#17639b',
  pale: '#dcecf9',
  gray: '#f1f4f7',
  purple: '#8650aa',
  green: '#256846',
  orange: '#d96b27',
  cardBg: '#f8fafc',
  warmBg: '#fdfaf6',
  softGreen: '#eef8f2',
  softOrange: '#fdf3eb'
};

const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const text = (x, y, s, size = 18, options = '') => `<text x="${x}" y="${y}" font-size="${size}" ${options}>${esc(s)}</text>`;
const rect = (x, y, w, h, fill = 'white', stroke = c.line, extra = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}" ${extra}/>`;
const arrow = (d, color = c.muted) => `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" marker-end="url(#arrow)"/>`;
const lines = (x, y, a, size = 18, leading = 28, options = '') => a.map((s, i) => text(x, y + i * leading, s, size, options)).join('');
const start = (w, h, title, desc) => `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="title description"><title id="title">${esc(title)}</title><desc id="description">${esc(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${c.muted}"/></marker></defs><rect width="${w}" height="${h}" fill="white"/><g font-family="PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, sans-serif" fill="${c.ink}">`;
const finish = '</g></svg>';

// Figure 1: moe-routing.svg
let f1 = start(1060, 840, 'MoE 稀疏路由与前馈专家数据流', '展示单个 token 从输入、路由器打分、Top-k 门控权重计算、稀疏分发至选定专家、到最终加权求和的完整前向数据流。');
f1 += text(32, 45, 'MoE 稀疏路由与专家前馈：单 Token 的计算数据流', 26, 'font-weight="600"');
f1 += text(32, 80, '固定一个位置输入 x : (C,)，专家总数 E = 4，激活专家数 Top-k = 2；展示路由权重与加权组合。', 17, `fill="${c.muted}"`);

// Input Box
f1 += rect(380, 110, 300, 52, c.pale, c.blue);
f1 += text(530, 142, 'Token 输入向量 x : (C,)', 19, 'text-anchor="middle" font-weight="600" fill="' + c.blue + '"');

// Branches from x
f1 += arrow('M450 162 V210 H220 V240');
f1 += arrow('M610 162 V210 H840 V240');

// Left: Router Path
f1 += rect(50, 240, 340, 290, c.cardBg, c.line);
f1 += text(220, 272, '① 路由器门控网络 (Router)', 20, 'text-anchor="middle" font-weight="600"');
f1 += rect(70, 295, 300, 44, 'white');
f1 += text(220, 323, '原始打分 logits = x @ W_gate  (E=4)', 16, 'text-anchor="middle"');
f1 += arrow('M220 339 V365');
f1 += rect(70, 365, 300, 44, 'white');
f1 += text(220, 393, 'Top-k 筛选：取前 k=2 大的分数与索引', 16, 'text-anchor="middle"');
f1 += arrow('M220 409 V435');
f1 += rect(70, 435, 300, 68, c.softOrange, c.orange);
f1 += lines(220, 462, ['在 Top-k 候选上重新 Softmax：', 'w₁ = 0.70 (专家 0),  w₂ = 0.30 (专家 2)'], 16, 24, 'text-anchor="middle" font-weight="500"');

// Right: Experts Pool
f1 += rect(430, 240, 580, 290, c.cardBg, c.line);
f1 += text(720, 272, '② 专家网络池 (E=4 个独立的 FFN)', 20, 'text-anchor="middle" font-weight="600"');

const expData = [
  { name: 'Expert 0 (激活)', state: '计算 y₀ = FFN₀(x)', fill: c.softGreen, stroke: c.green, active: true },
  { name: 'Expert 1 (跳过)', state: '本 token 跳过此 FFN', fill: '#f8f8f8', stroke: '#d0d0d0', active: false },
  { name: 'Expert 2 (激活)', state: '计算 y₂ = FFN₂(x)', fill: c.softGreen, stroke: c.green, active: true },
  { name: 'Expert 3 (跳过)', state: '本 token 跳过此 FFN', fill: '#f8f8f8', stroke: '#d0d0d0', active: false },
];

for (let i = 0; i < 4; i++) {
  const ex = 450 + (i % 2) * 280;
  const ey = 300 + Math.floor(i / 2) * 95;
  const d = expData[i];
  f1 += rect(ex, ey, 260, 75, d.fill, d.stroke);
  f1 += text(ex + 130, ey + 30, d.name, 17, `text-anchor="middle" font-weight="600" fill="${d.active ? c.green : c.muted}"`);
  f1 += text(ex + 130, ey + 56, d.state, 15, `text-anchor="middle" fill="${d.active ? c.ink : c.muted}"`);
}

// Arrows from Router & Experts to Combine Box
f1 += arrow('M220 530 V600 H480 V635');
f1 += arrow('M710 337 H720 V565 H540 V635');
f1 += arrow('M580 470 V580 H570 V635');

// Combine Box
f1 += rect(280, 635, 500, 95, c.warmBg, c.orange);
f1 += text(530, 668, '③ 门控加权聚合 (Weighted Combination)', 19, 'text-anchor="middle" font-weight="600" fill="' + c.orange + '"');
f1 += text(530, 702, 'y = w₁ · y₀ + w₂ · y₂ = 0.70 · FFN₀(x) + 0.30 · FFN₂(x)', 17, 'text-anchor="middle" font-weight="500"');

// Output
f1 += arrow('M530 730 V775');
f1 += rect(380, 775, 300, 48, c.pale, c.blue);
f1 += text(530, 805, 'MoE 层输出 y : (C,)', 18, 'text-anchor="middle" font-weight="600" fill="' + c.blue + '"');

f1 += finish;
writeFileSync(join(root, 'moe-routing.svg'), f1);

// Figure 2: load-balancing.svg
let f2 = start(1060, 680, 'MoE 专家负载不均与辅助损失机制', '对比展示路由塌陷导致部分专家过载与辅助平衡损失促使均匀分配的机制。');
f2 += text(32, 45, '专家负载不均衡 (Routing Collapse) 与辅助平衡损失', 26, 'font-weight="600"');
f2 += text(32, 80, '这里统计 Top-1 分配占比；辅助项结合硬分配占比 f 与可导的平均概率 P，鼓励均衡。', 17, `fill="${c.muted}"`);

// Left Panel: Collapse
f2 += rect(40, 115, 470, 450, c.cardBg, c.line);
f2 += text(275, 150, 'A. 分配集中的示意（非实测）', 21, 'text-anchor="middle" font-weight="600" fill="#b32d2e"');

f2 += lines(65, 188, [
  '• 少数专家可能在训练中长期收到更多 token',
  '• 路由器将绝大多数 Token 路由到这几个专家 (如 E0, E1)',
  '• 剩余专家被“饿死”，主任务训练信号较少',
  '• 在分布式并行中，少数设备严重过载成为木桶短板'
], 15, 26);

// Bar Chart Left
const barY = 320;
f2 += text(65, barY - 15, '各专家接收的 Token 流量占比：', 16, 'font-weight="600"');
const leftBars = [
  { name: 'E0', p: '65%', w: 220, fill: '#b32d2e' },
  { name: 'E1', p: '30%', w: 100, fill: '#d96b27' },
  { name: 'E2', p: '3%', w: 15, fill: '#c8d3df' },
  { name: 'E3', p: '2%', w: 10, fill: '#c8d3df' }
];
for (let i = 0; i < 4; i++) {
  const b = leftBars[i];
  b.w = 340 * parseFloat(b.p) / 100;
  f2 += text(70, barY + 25 + i * 36, b.name, 15, 'font-weight="600"');
  f2 += `<rect x="105" y="${barY + 10 + i * 36}" width="${b.w}" height="22" rx="4" fill="${b.fill}"/>`;
  f2 += text(115 + b.w, barY + 26 + i * 36, b.p, 15, `fill="${c.ink}" font-weight="600"`);
}
f2 += text(275, 530, '⚠️ 负载不均；不能仅凭此图推断实际延迟', 16, 'text-anchor="middle" fill="#b32d2e" font-weight="600"');

// Right Panel: Balanced
f2 += rect(550, 115, 470, 450, c.cardBg, c.line);
f2 += text(785, 150, 'B. 较均衡的示意（非保证结果）', 21, 'text-anchor="middle" font-weight="600" fill="' + c.green + '"');

f2 += lines(575, 188, [
  '• 统计批次内各专家的调度频率 f_e 与平均门控权重 P_e',
  '• 构造负载均衡辅助损失：L_aux = α · E · Σ (f_e · P_e)',
  '• f_e = P_e = 1/E 时值为 α，不是一般严格下界',
  '• 梯度经过 P 调整路由参数，实际分配仍需检查'
], 15, 26);

// Bar Chart Right
f2 += text(575, barY - 15, '各专家接收的 Token 流量占比：', 16, 'font-weight="600"');
const rightBars = [
  { name: 'E0', p: '26%', w: 90, fill: c.green },
  { name: 'E1', p: '24%', w: 82, fill: c.green },
  { name: 'E2', p: '25%', w: 86, fill: c.green },
  { name: 'E3', p: '25%', w: 86, fill: c.green }
];
for (let i = 0; i < 4; i++) {
  const b = rightBars[i];
  b.w = 340 * parseFloat(b.p) / 100;
  f2 += text(580, barY + 25 + i * 36, b.name, 15, 'font-weight="600"');
  f2 += `<rect x="615" y="${barY + 10 + i * 36}" width="${b.w}" height="22" rx="4" fill="${b.fill}"/>`;
  f2 += text(625 + b.w, barY + 26 + i * 36, b.p, 15, `fill="${c.ink}" font-weight="600"`);
}
f2 += text(785, 530, '✅ 各专家接收量接近；效果和速度另行验证', 16, 'text-anchor="middle" fill="' + c.green + '" font-weight="600"');

f2 += text(32, 615, '关键总结：MoE 的核心收益是“用总参数量换容量，用激活参数量控算力”。', 18, 'font-weight="600"');
f2 += text(32, 645, '辅助项只鼓励均衡；是否有效需检查实际负载，不能单靠损失值判断训练与性能。', 17, `fill="${c.muted}"`);

f2 += finish;
writeFileSync(join(root, 'load-balancing.svg'), f2);

for (const name of ['moe-routing', 'load-balancing']) {
  execFileSync('rsvg-convert', ['--output', join(root, `${name}.png`), join(root, `${name}.svg`)]);
}
console.log('Successfully generated moe-routing.png and load-balancing.png');
