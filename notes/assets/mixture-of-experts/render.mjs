// Rebuild: node notes/assets/mixture-of-experts/render.mjs
// Requires rsvg-convert (librsvg). All diagrams and numbers are generated locally.
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';

const root = dirname(fileURLToPath(import.meta.url));
const c = { ink: '#172b40', muted: '#526477', line: '#c8d3df', blue: '#17639b', pale: '#dcecf9', gray: '#f1f4f7', purple: '#8650aa', green: '#256846', orange: '#b75519' };
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const txt = (x, y, s, size = 20, options = '') => `<text x="${x}" y="${y}" font-size="${size}" ${options}>${esc(s)}</text>`;
const mid = (x, y, s, size = 20, options = '') => txt(x, y, s, size, `text-anchor="middle" ${options}`);
const box = (x, y, w, h, fill = 'white', stroke = c.line, extra = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${fill}" stroke="${stroke}" ${extra}/>`;
const line = (d, color = c.muted, extra = '') => `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" ${extra}/>`;
const arrow = (d, extra = '') => line(d, c.muted, `marker-end="url(#arrow)" ${extra}`);
const lines = (x, y, a, size = 20, leading = 30, extra = '') => a.map((s, i) => txt(x, y + i * leading, s, size, extra)).join('');
const start = (h, title, desc) => `<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="${h}" viewBox="0 0 1050 ${h}" role="img" aria-labelledby="title description"><title id="title">${esc(title)}</title><desc id="description">${esc(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${c.muted}"/></marker></defs><rect width="1050" height="${h}" fill="white"/><g font-family="PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, sans-serif" fill="${c.ink}">`;
const end = '</g></svg>';
const heading = title => txt(32, 47, title, 28, 'font-weight="600"');
const plus = (x, y) => `<circle cx="${x}" cy="${y}" r="23" fill="white" stroke="${c.muted}" stroke-width="2"/>${mid(x, y + 8, '+', 29)}`;
const save = (name, source) => {
  writeFileSync(join(root, `${name}.svg`), source + end);
  execFileSync('rsvg-convert', ['--output', join(root, `${name}.png`), join(root, `${name}.svg`)]);
};

// Figure 1: residual paths and the exact place where MoE replaces the FFN.
let a = start(1175, 'MoE 放在哪里：替换这一层的 FFN', 'LLaMA 风格的 Pre-Norm 层：输入 X 经过 RMSNorm 和因果 Attention 后与 X 相加得到 H；H 经过 RMSNorm 得到 Z，Z 经 MoE 后与 H 相加。MoE 内部为每个 token 选择专家、运行完整专家 FFN 并加权相加。');
a += heading('MoE 放在哪里：替换这一层的 FFN');
a += txt(32, 85, '单层 LLaMA 风格 Pre-Norm；B = batch，T = token 数，C = 隐藏宽度。', 20);
a += box(32, 106, 986, 520, '#f8fafc');
a += mid(515, 140, '输入 X：(B, T, C)', 22, 'font-weight="600"');
a += arrow('M515 150 V166');
a += arrow('M515 150 H175 V317 H491');
a += txt(90, 242, '保留 X', 20, `fill="${c.muted}"`);
a += box(360, 168, 310, 43); a += mid(515, 197, 'RMSNorm', 21);
a += arrow('M515 211 V232');
a += box(360, 234, 310, 51, c.pale); a += mid(515, 267, '因果 Attention', 22);
a += arrow('M515 285 V293');
a += plus(515, 317);
a += txt(567, 323, 'H = X + Attention(RMSNorm(X))', 20);
a += arrow('M515 340 V369');
a += arrow('M515 350 H883 V537 H539');
a += txt(900, 444, '保留 H', 20, `fill="${c.muted}"`);
a += box(360, 371, 310, 43); a += mid(515, 400, 'RMSNorm', 21);
a += arrow('M515 414 V450');
a += txt(551, 440, 'Z：(B, T, C)', 20, `fill="${c.blue}"`);
a += box(360, 452, 310, 51, '#f0e7f6', c.purple); a += mid(515, 485, 'MoE：替换原 SwiGLU FFN', 21);
a += arrow('M515 503 V513');
a += plus(515, 537);
a += arrow('M515 560 V581');
a += mid(515, 609, '本层输出 = H + MoE(Z)；shape 仍为 (B, T, C)', 22, 'font-weight="600"');
a += txt(62, 589, '⊕ 表示逐元素相加', 18, `fill="${c.muted}"`);
a += txt(32, 664, '放大 MoE：下面只看 Z 中的一行向量 z，shape 为 (C,)。', 23, 'font-weight="600"');
a += txt(32, 698, '示意：3 个独立专家，当前 token 只选择专家 0 和 2；每个专家都是完整 FFN。', 20);
a += box(32, 725, 986, 340, '#fbf9fd');
a += mid(92, 900, 'z', 27, 'font-weight="600"');
a += arrow('M113 890 H177');
a += box(180, 812, 220, 152, 'white', c.purple);
a += mid(290, 850, 'Router（路由器）', 22, 'font-weight="600"');
a += mid(290, 885, '计算专家分数', 20);
a += mid(290, 918, '选编号 0、2', 20);
a += mid(290, 949, '得到权重 w₀、w₂', 19);
a += arrow('M290 812 V766 H900 V810', 'stroke-dasharray="6 5"');
a += mid(615, 754, '所选专家的权重 w₀、w₂', 19, `fill="${c.purple}"`);
a += arrow('M400 850 H459 V834 H503');
a += arrow('M459 850 V973 H503');
a += txt(411, 1027, '复制整条 z', 19, `fill="${c.muted}"`);
a += box(506, 805, 226, 58, c.pale); a += mid(619, 842, '专家 0：F₀(z)', 21);
a += box(506, 876, 226, 52, c.gray); a += mid(619, 909, '专家 1：此 token 不运行', 18, `fill="${c.muted}"`);
a += box(506, 944, 226, 58, '#e7f2eb'); a += mid(619, 981, '专家 2：F₂(z)', 21);
a += arrow('M732 834 H763 V858 H795');
a += arrow('M732 973 H763 V922 H795');
a += box(798, 812, 203, 154, 'white', c.purple);
a += mid(899, 850, '加权相加', 23, 'font-weight="600"');
a += mid(899, 887, 'w₀F₀(z)', 23);
a += mid(899, 919, '+ w₂F₂(z)', 23);
a += mid(899, 950, '输出：(C,)', 19);
a += txt(32, 1110, '稀疏的是“每个 token 只运行少数专家”；Attention 的可见范围由自己的 mask 决定。', 20);
a += txt(32, 1146, '所有专家参数仍属于模型；上图展示的是这个 token 本次实际经过的计算路径。', 20, `fill="${c.muted}"`);
save('block-location', a);

// Figure 2: use computed top-k choices, normalized weights and dispatch groups.
const probabilities = [[.6, .3, .1], [.2, .5, .3], [.1, .2, .7], [.55, .1, .35]];
const selected = probabilities.map(row => {
  assert.ok(Math.abs(row.reduce((s, p) => s + p, 0) - 1) < 1e-12);
  const ids = [0, 1, 2].sort((i, j) => row[j] - row[i]).slice(0, 2);
  const denom = ids.reduce((sum, id) => sum + row[id], 0);
  const result = ids.map(id => ({ id, weight: row[id] / denom }));
  assert.ok(Math.abs(result.reduce((s, entry) => s + entry.weight, 0) - 1) < 1e-12);
  return result;
});
const groups = [0, 1, 2].map(id => selected.flatMap((entries, token) => entries.some(entry => entry.id === id) ? [token] : []));
assert.deepEqual(groups, [[0, 3], [0, 1, 2], [1, 2, 3]]);
assert.equal(groups.reduce((n, tokens) => n + tokens.length, 0), 8);
const sub = ['₀', '₁', '₂', '₃'];
const num = n => n.toFixed(3);

let b = start(1370, '4 个 token 如何分给专家，再恢复原顺序', 'E=3，k=2，4 个 token 各选 2 个专家。先给出完整专家概率和所选专家归一化权重，再按专家分组，最后按原 token 下标加权累加。总计 8 次 token-专家分配，整条 C 维向量参与每个被选专家。');
b += heading('4 个 token 如何分给专家，再恢复原顺序');
b += txt(32, 85, 'MoE 输入展平为 Z_flat：(N, C)，N = B × T = 4；一行 zᵢ 是完整的 C 维向量。', 20);
b += txt(32, 118, '3 个专家（E = 3）；每个 token 选 2 个（k = 2）；本图不设容量限制。', 20);
b += txt(32, 164, '① Router 对每个 token 打分；Softmax 得到所有专家的概率 p。', 23, 'font-weight="600"');
const cx = [32, 158, 275, 392, 509, 711, 1018];
b += box(32, 187, 986, 253, 'white');
b += `<rect x="33" y="188" width="984" height="48" fill="${c.gray}"/>`;
const headers = ['token', 'p₀', 'p₁', 'p₂', '选中专家', '所选权重（和为 1）'];
headers.forEach((t, i) => b += mid((cx[i] + cx[i + 1]) / 2, 220, t, 20, 'font-weight="600"'));
for (let i = 0; i < 4; i++) {
  const y = 273 + i * 50;
  if (i > 0) b += line(`M32 ${y - 33} H1018`, c.line);
  b += mid(95, y, `z${sub[i]}`, 22, 'font-weight="600"');
  probabilities[i].forEach((p, j) => b += mid((cx[j + 1] + cx[j + 2]) / 2, y, p.toFixed(2), 21));
  b += mid(610, y, selected[i].map(entry => entry.id).join('、'), 21, `fill="${c.blue}"`);
  b += mid(864, y, selected[i].map(entry => `w${sub[entry.id]}=${num(entry.weight)}`).join('；'), 19);
}
b += txt(32, 477, '例如 z₀：保留 p₀ = 0.6、p₁ = 0.3，再除以 0.9 → 权重 2/3、1/3。', 21);
b += txt(32, 511, '归一化只在选中的两个专家之间进行；表中小数显示到 3 位，计算保留完整精度。', 19, `fill="${c.muted}"`);
b += txt(32, 563, '② 按专家分组，让同一个专家一次处理分给自己的 token。', 23, 'font-weight="600"');
const xs = [32, 370, 708];
groups.forEach((tokens, expert) => {
  b += box(xs[expert], 588, 310, 237, expert === 0 ? c.pale : expert === 1 ? '#f0e7f6' : '#e7f2eb');
  b += txt(xs[expert] + 20, 625, `专家 ${expert}：F${sub[expert]}`, 23, 'font-weight="600"');
  b += txt(xs[expert] + 20, 663, `输入行：${tokens.map(t => `z${sub[t]}`).join('、')}`, 21);
  b += txt(xs[expert] + 20, 699, `输入 shape：(${tokens.length}, C)`, 20);
  b += txt(xs[expert] + 20, 737, '运行自己的完整 FFN', 21);
  b += txt(xs[expert] + 20, 777, `输出 shape：(${tokens.length}, C)`, 20);
});
b += txt(32, 863, '4 个 token × 每个选 2 个专家 = 8 次分配；同一 zᵢ 会出现在两个专家的输入里。', 20);
b += txt(32, 897, '复制的是整条 C 维向量；没有把前 C/2 维、后 C/2 维分给不同专家。', 20, `fill="${c.blue}"`);
b += txt(32, 949, '③ 用原 token 编号找回位置，把专家输出乘权重后相加。', 23, 'font-weight="600"');
b += box(32, 973, 986, 249, '#f8fafc');
selected.forEach((entries, token) => {
  const sorted = [...entries].sort((a, b) => a.id - b.id);
  const equation = `y${sub[token]} ≈ ` + sorted.map(entry => `${num(entry.weight)} × F${sub[entry.id]}(z${sub[token]})`).join(' + ');
  b += txt(59, 1015 + token * 54, equation, 24);
});
b += txt(32, 1264, '输出 Y：(4, C)，行顺序仍是 y₀、y₁、y₂、y₃；随后恢复成 (B, T, C)。', 22);
b += txt(32, 1301, '若两位专家都处理 z₀，两份结果必须累加到 y₀；不能让后写入的结果覆盖前一份。', 20);
b += txt(32, 1341, '每位专家的 FFN 参数各自独立；Router 决定本次计算选谁、各占多少权重。', 20, `fill="${c.muted}"`);
save('route-and-combine', b);

// Figure 3: same number of assignments, different loads; capacity is per expert.
const balanced = [2, 2, 2, 2], congested = [8, 0, 0, 0], capacity = 3;
assert.equal(balanced.reduce((a, b) => a + b, 0), 8);
assert.equal(congested.reduce((a, b) => a + b, 0), 8);
assert.deepEqual(congested.map(n => Math.max(0, n - capacity)), [5, 0, 0, 0]);
function chart(x, y, counts, title) {
  let s = box(x, y, 472, 507, '#fafbfd');
  s += txt(x + 22, y + 37, title, 23, 'font-weight="600"');
  s += txt(x + 22, y + 70, '纵轴：分给各专家的 token 数', 18, `fill="${c.muted}"`);
  const base = y + 395, top = y + 107, step = 36;
  for (let t = 0; t <= 8; t++) {
    const yy = base - t * step;
    s += line(`M${x + 55} ${yy} H${x + 447}`, t === 0 ? c.muted : '#e6ebf1');
    s += txt(x + 31, yy + 7, t, 17, `fill="${c.muted}"`);
  }
  counts.forEach((n, j) => {
    const bx = x + 80 + j * 93;
    const accepted = Math.min(n, capacity), overflow = Math.max(0, n - capacity);
    if (accepted) s += `<rect x="${bx}" y="${base - accepted * step}" width="49" height="${accepted * step}" fill="${c.blue}"/>`;
    if (overflow) s += `<rect x="${bx}" y="${base - n * step}" width="49" height="${overflow * step}" fill="#edb081" stroke="${c.orange}"/>`;
    s += mid(bx + 24.5, base - n * step - 12, n, 22, 'font-weight="600"');
    s += mid(bx + 24.5, base + 36, `E${sub[j]}`, 22);
  });
  const capY = base - capacity * step;
  s += line(`M${x + 55} ${capY} H${x + 447}`, c.purple, 'stroke-dasharray="8 5"');
  s += txt(x + 175, capY - 12, '每位专家容量 = 3', 18, `fill="${c.purple}"`);
  s += txt(x + 22, y + 474, `分配数：[${counts.join(', ')}]；总计 8`, 20);
  return s;
}
let d = start(1060, '为什么需要关心负载：同样 8 次分配，可以很不均匀', '独立的 N=8、E=4、k=1 例子。均衡时每个专家各收到2个token；拥堵时8个token都选择专家0，其余专家空闲。每个专家容量3，拥堵时仅专家0超出5次分配；总空位不能自动承接，是否丢弃、改派或等待由具体策略决定。');
d += heading('为什么需要关心负载：同样 8 次分配，可以很不均匀');
d += txt(32, 86, '独立例子：N = 8 个 token，E = 4 个专家，每个 token 只选 1 个专家（k = 1）。', 20);
d += txt(32, 121, '先看 Router 想把 token 分给谁；虚线是假设的每个专家容量 Ccap = 3。', 20);
d += chart(32, 151, balanced, 'A. 均衡：每位专家收到 2 个');
d += chart(546, 151, congested, 'B. 拥堵：全部选择专家 0');
d += box(32, 687, 986, 116, '#fff5ec', '#dfb993');
d += txt(55, 727, '橙色 5 格 = 专家 0 超出的 5 次分配。', 23, 'font-weight="600"');
d += txt(55, 764, '其余 3 位专家虽然空闲，也不会自动接走它们；处理方式需要另行定义。', 21);
d += txt(32, 852, '如果设置硬容量：每位专家最多接收 3 次分配；它不是“整个 batch 只留 3 个 token”。', 20);
d += txt(32, 891, '本图只标出溢出位置；丢弃该分支、改派或等待，是不同的实现策略。', 21);
d += txt(32, 944, '不设置硬容量：专家 0 可以继续处理全部 8 个，但计算任务集中在它身上。', 21);
d += txt(32, 983, '这说明了负载均衡的目的：减少专家长期拥堵或闲置；不直接证明运行会更快。', 21);
d += txt(32, 1025, '图中柱高是容量限制前的分配需求；蓝色部分不代表“整个模型只计算这些 token”。', 19, `fill="${c.muted}"`);
save('load-and-capacity', d);

console.log('Generated 3 SVG/PNG pairs; verified top-2 weights, dispatch groups, 8 assignments and 5 overflow assignments.');
