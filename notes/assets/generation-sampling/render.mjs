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
  softOrange: '#fdf3eb',
  danger: '#b32d2e'
};

const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const text = (x, y, s, size = 18, options = '') => `<text x="${x}" y="${y}" font-size="${size}" ${options}>${esc(s)}</text>`;
const rect = (x, y, w, h, fill = 'white', stroke = c.line, extra = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}" ${extra}/>`;
const arrow = (d, color = c.muted) => `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" marker-end="url(#arrow)"/>`;
const lines = (x, y, a, size = 18, leading = 28, options = '') => a.map((s, i) => text(x, y + i * leading, s, size, options)).join('');
const start = (w, h, title, desc) => `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="title description"><title id="title">${esc(title)}</title><desc id="description">${esc(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${c.muted}"/></marker></defs><rect width="${w}" height="${h}" fill="white"/><g font-family="PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, sans-serif" fill="${c.ink}">`;
const finish = '</g></svg>';

// Figure 1: temperature-distribution.svg
let f1 = start(1060, 680, '温度调节对概率分布的影响', '展示相同 logits 在 T<1 尖锐化、T=1 原始分布与 T>1 平坦化三种情况下的概率变化对照。');
f1 += text(32, 45, '温度参数（Temperature）对 Softmax 概率分布的调节', 26, 'font-weight="600"');
f1 += text(32, 80, '原始分数 logits = [4.0, 2.0, 1.0, 0.0]；对比不同 T 下除以 T 后 Softmax 的概率演变。', 17, `fill="${c.muted}"`);

const tempCards = [
  {
    title: 'A. 低温 T = 0.5（尖锐化）',
    desc: '放大分数差异，高分概率被极度放大',
    probs: [
      { name: 'token 0 (4.0)', p: '87.4%', w: 220, fill: c.blue },
      { name: 'token 1 (2.0)', p: '10.5%', w: 30, fill: c.pale },
      { name: 'token 2 (1.0)', p: '1.8%', w: 8, fill: c.gray },
      { name: 'token 3 (0.0)', p: '0.3%', w: 2, fill: c.gray }
    ],
    foot: '更确定、保守，趋近贪心选择',
    color: c.blue
  },
  {
    title: 'B. 原始 T = 1.0（标准分布）',
    desc: '忠实反映模型训练出的原始预测概率',
    probs: [
      { name: 'token 0 (4.0)', p: '62.9%', w: 160, fill: c.green },
      { name: 'token 1 (2.0)', p: '23.1%', w: 60, fill: c.softGreen },
      { name: 'token 2 (1.0)', p: '9.8%', w: 26, fill: c.gray },
      { name: 'token 3 (0.0)', p: '4.2%', w: 12, fill: c.gray }
    ],
    foot: '保留模型原始 Softmax 分布',
    color: c.green
  },
  {
    title: 'C. 高温 T = 2.0（平坦化）',
    desc: '缩小相对差距，低分候选获得更多机会',
    probs: [
      { name: 'token 0 (4.0)', p: '40.6%', w: 105, fill: c.orange },
      { name: 'token 1 (2.0)', p: '27.4%', w: 72, fill: c.softOrange },
      { name: 'token 2 (1.0)', p: '19.0%', w: 50, fill: c.gray },
      { name: 'token 3 (0.0)', p: '13.0%', w: 35, fill: c.gray }
    ],
    foot: '随机性强、多样，效果须按任务验证',
    color: c.orange
  }
];

for (let i = 0; i < 3; i++) {
  const tc = tempCards[i];
  const temperature = [.5, 1, 2][i];
  const logits = [4, 2, 1, 0];
  const exps = logits.map(v => Math.exp((v - 4) / temperature));
  const total = exps.reduce((a, b) => a + b, 0);
  tc.probs.forEach((entry, j) => {
    const probability = exps[j] / total;
    entry.p = `${(100 * probability).toFixed(2)}%`;
    entry.w = 220 * probability;
  });
  const cx = 35 + i * 340;
  f1 += rect(cx, 115, 320, 470, c.cardBg, c.line);
  f1 += text(cx + 160, 150, tc.title, 18, `text-anchor="middle" font-weight="600" fill="${tc.color}"`);
  f1 += text(cx + 160, 178, tc.desc, 14, `text-anchor="middle" fill="${c.muted}"`);

  const by = 220;
  for (let j = 0; j < 4; j++) {
    const pr = tc.probs[j];
    f1 += text(cx + 20, by + j * 55, pr.name, 14, 'font-weight="500"');
    f1 += `<rect x="${cx + 20}" y="${by + 10 + j * 55}" width="${pr.w}" height="18" rx="4" fill="${pr.fill}"/>`;
    f1 += text(cx + 30 + pr.w, by + 24 + j * 55, pr.p, 14, `font-weight="600" fill="${c.ink}"`);
  }

  f1 += rect(cx + 20, 510, 280, 50, 'white', c.line);
  f1 += text(cx + 160, 541, tc.foot, 14, `text-anchor="middle" font-weight="600" fill="${tc.color}"`);
}

f1 += text(32, 620, '温度公式：P(x_i) = softmax(z_i / T)。唯一最大项时，T → 0 集中到该项；有限 logits 下 T → ∞ 趋向均匀。', 17, 'font-weight="600"');
f1 += text(32, 650, '温度改变了分布的平缓程度，但不会改变候选词之间的相对排序（保持单调性）。', 16, `fill="${c.muted}"`);

f1 += finish;
writeFileSync(join(root, 'temperature-distribution.svg'), f1);

// Figure 2: topk-topp-filtering.svg
let f2 = start(1060, 720, 'Top-k 与 Top-p 动态截断机制对照', '对比 Top-k 的固定数量截断与 Top-p（核采样）根据累积概率动态自适应截断尾部的机制。');
f2 += text(32, 45, '截断采样：Top-k 与 Top-p（Nucleus Sampling）机制对比', 26, 'font-weight="600"');
f2 += text(32, 80, '截断将部分候选 logits 置为 -inf；低概率不等于错误，保留下来的高概率项也不保证正确。', 17, `fill="${c.muted}"`);

// Left Panel: Top-k
f2 += rect(40, 115, 470, 510, c.cardBg, c.line);
f2 += text(275, 150, 'A. Top-k 采样（固定数量截断）', 21, 'text-anchor="middle" font-weight="600" fill="' + c.blue + '"');
f2 += text(275, 178, '仅保留概率最高的前 k 个候选词，其余设为 -inf', 15, `text-anchor="middle" fill="${c.muted}"`);

f2 += lines(65, 215, [
  '• 规则：固定保留 k=3 个候选，丢弃所有后续候选',
  '• 优点：严格限制采样空间，实现简单高效',
  '• 缺点（刚性）：无法适应不同上下文的确定性差异'
], 15, 26);

// Case 1 in Top-k
f2 += rect(65, 305, 420, 140, 'white', c.line);
f2 += text(80, 330, '情景 1：分布平坦（多个合理词）', 15, 'font-weight="600"');
f2 += text(80, 355, '候选词概率：[0.30, 0.28, 0.25, 0.15, 0.02]', 14, `fill="${c.muted}"`);
f2 += `<rect x="80" y="375" width="260" height="20" rx="4" fill="${c.pale}"/>`;
f2 += text(210, 390, '保留前 3 个 (共占 83%)', 13, `text-anchor="middle" font-weight="600" fill="${c.blue}"`);
f2 += `<rect x="350" y="375" width="110" height="20" rx="4" fill="#fed7d7"/>`;
f2 += text(405, 390, '截断 (17%)', 13, `text-anchor="middle" font-weight="600" fill="${c.danger}"`);
f2 += text(80, 425, '⚠️ 第 4 个词概率 0.15 可能合理，也会被截断。', 13, `font-weight="600" fill="${c.danger}"`);

// Case 2 in Top-k
f2 += rect(65, 460, 420, 140, 'white', c.line);
f2 += text(80, 485, '情景 2：分布极其集中（高确定性）', 15, 'font-weight="600"');
f2 += text(80, 510, '候选词概率：[0.96, 0.02, 0.015, 0.005]', 14, `fill="${c.muted}"`);
f2 += `<rect x="80" y="530" width="310" height="20" rx="4" fill="${c.pale}"/>`;
f2 += text(235, 545, '强行保留 3 个候选 (共 99.5%)', 13, `text-anchor="middle" font-weight="600" fill="${c.blue}"`);
f2 += `<rect x="400" y="530" width="60" height="20" rx="4" fill="#fed7d7"/>`;
f2 += text(430, 545, '截断', 13, `text-anchor="middle" font-weight="600" fill="${c.danger}"`);
f2 += text(80, 580, '⚠️ 0.02、0.015 的低概率候选仍会参与采样。', 13, `font-weight="600" fill="${c.danger}"`);

// Right Panel: Top-p
f2 += rect(550, 115, 470, 510, c.cardBg, c.line);
f2 += text(785, 150, 'B. Top-p 核采样（动态累积概率截断）', 21, 'text-anchor="middle" font-weight="600" fill="' + c.green + '"');
f2 += text(785, 178, '按概率降序累加，保留累积和刚好达到 p 的最小集合', 15, `text-anchor="middle" fill="${c.muted}"`);

f2 += lines(575, 215, [
  '• 规则：设 p=0.90，动态自适应决定保留多少个候选词',
  '• 优点：上下文确定时缩小候选池，发散时自动扩大候选池',
  '• 边界：筛选依据是概率，不是候选答案的正确性'
], 15, 26);

// Case 1 in Top-p
f2 += rect(575, 305, 420, 140, 'white', c.line);
f2 += text(590, 330, '情景 1：分布平坦（动态扩大）', 15, 'font-weight="600"');
f2 += text(590, 355, '累积概率：0.30 → 0.58 → 0.83 → 0.98 (>0.90 停止)', 14, `fill="${c.muted}"`);
f2 += `<rect x="590" y="375" width="330" height="20" rx="4" fill="${c.softGreen}"/>`;
f2 += text(755, 390, '自适应保留前 4 个词 (累积 98%)', 13, `text-anchor="middle" font-weight="600" fill="${c.green}"`);
f2 += `<rect x="930" y="375" width="45" height="20" rx="4" fill="#fed7d7"/>`;
f2 += text(952, 390, '截断', 13, `text-anchor="middle" font-weight="600" fill="${c.danger}"`);
f2 += text(590, 425, '✅ 自动扩展到 4 个词，保留更多候选，不保证它们正确', 13, `font-weight="600" fill="${c.green}"`);

// Case 2 in Top-p
f2 += rect(575, 460, 420, 140, 'white', c.line);
f2 += text(590, 485, '情景 2：分布集中（动态收缩）', 15, 'font-weight="600"');
f2 += text(590, 510, '累积概率：0.96 (首个词已 ≥ 0.90，立即截断)', 14, `fill="${c.muted}"`);
f2 += `<rect x="590" y="530" width="280" height="20" rx="4" fill="${c.softGreen}"/>`;
f2 += text(730, 545, '自适应仅保留第 1 个词 (96%)', 13, `text-anchor="middle" font-weight="600" fill="${c.green}"`);
f2 += `<rect x="880" y="530" width="95" height="20" rx="4" fill="#fed7d7"/>`;
f2 += text(927, 545, '截断尾部', 13, `text-anchor="middle" font-weight="600" fill="${c.danger}"`);
f2 += text(590, 580, '✅ 自动收缩为只保留 1 个词，仅此候选参与后续采样', 13, `font-weight="600" fill="${c.green}"`);

f2 += text(32, 655, '本课组合顺序：先除以 Temperature → 再应用 Top-k 粗筛 → 再应用 Top-p 细筛 → 最后 multinomial 采样。', 17, 'font-weight="600"');
f2 += text(32, 685, '色带仅示意保留与截断分组，不按概率比例绘制；输出概率在保留集合内重新归一化。', 16, `fill="${c.muted}"`);

f2 += finish;
writeFileSync(join(root, 'topk-topp-filtering.svg'), f2);

for (const name of ['temperature-distribution', 'topk-topp-filtering']) {
  execFileSync('rsvg-convert', ['--output', join(root, `${name}.png`), join(root, `${name}.svg`)]);
}
console.log('Successfully generated temperature-distribution.png and topk-topp-filtering.png');
