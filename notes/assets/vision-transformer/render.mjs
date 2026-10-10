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
  danger: '#b32d2e',
  patchFill: '#e0effa',
  patchStroke: '#3182ce'
};

const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const text = (x, y, s, size = 18, options = '') => `<text x="${x}" y="${y}" font-size="${size}" ${options}>${esc(s)}</text>`;
const rect = (x, y, w, h, fill = 'white', stroke = c.line, extra = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}" ${extra}/>`;
const arrow = (d, color = c.muted) => `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" marker-end="url(#arrow)"/>`;
const lines = (x, y, a, size = 18, leading = 28, options = '') => a.map((s, i) => text(x, y + i * leading, s, size, options)).join('');
const start = (w, h, title, desc) => `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="title description"><title id="title">${esc(title)}</title><desc id="description">${esc(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${c.muted}"/></marker></defs><rect width="${w}" height="${h}" fill="white"/><g font-family="PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, sans-serif" fill="${c.ink}">`;
const finish = '</g></svg>';

// Figure 1: vit-architecture.svg
let f1 = start(1060, 880, 'Vision Transformer (ViT) 端到端架构与数据流', '展示从二维图像分块、展平线性投影为 Patch Tokens、拼接入 CLS Token、叠加一维可学习位置编码，送入标准 Transformer Encoder，最后通过 MLP Head 进行分类的完整流水线。');
f1 += text(32, 45, 'Vision Transformer (ViT)：图像到序列的端到端架构', 26, 'font-weight="600"');
f1 += text(32, 80, '输入图像 (H, W, C) → 切分成 N 个 (P, P, C) 图块 → 线性映射为向量 → 拼 CLS 与位置编码 → 标准 Encoder → 分类头', 17, `fill="${c.muted}"`);

// 1. Image & Patch Extraction
f1 += rect(40, 115, 300, 310, c.cardBg, c.line);
f1 += text(190, 145, '① 图像分块 (Patch Extraction)', 18, 'text-anchor="middle" font-weight="600" fill="' + c.blue + '"');
f1 += text(190, 172, '原图: H×W×C（下方网格仅示意）', 14, `text-anchor="middle" fill="${c.muted}"`);

// Draw 3x3 patches representing image
const patchSize = 50;
const startX = 115, startY = 190;
for (let r = 0; r < 3; r++) {
  for (let col = 0; col < 3; col++) {
    const idx = r * 3 + col + 1;
    f1 += `<rect x="${startX + col * (patchSize + 4)}" y="${startY + r * (patchSize + 4)}" width="${patchSize}" height="${patchSize}" rx="4" fill="${c.patchFill}" stroke="${c.patchStroke}" stroke-width="1.5"/>`;
    f1 += text(startX + col * (patchSize + 4) + 25, startY + r * (patchSize + 4) + 30, `P${idx}`, 14, 'text-anchor="middle" font-weight="600" fill="' + c.blue + '"');
  }
}
f1 += text(190, 375, '切成 N = (H/P) × (W/P) 个图块', 14, 'text-anchor="middle" font-weight="500"');
f1 += text(190, 398, '每个图块展平维度: P² · C (如 16×16×3 = 768)', 13, `text-anchor="middle" fill="${c.muted}"`);

// Arrow from Step 1 to Step 2
f1 += arrow('M340 270 H390');

// 2. Linear Projection & Token Prep
f1 += rect(390, 115, 630, 310, c.cardBg, c.line);
f1 += text(705, 145, '② 线性投影与序列构造 (Patch Embedding + CLS + Pos)', 18, 'text-anchor="middle" font-weight="600" fill="' + c.purple + '"');

// Draw Sequence of Tokens
const tokens = ['[CLS]', 'P1', 'P2', 'P3', '...', 'PN'];
const tokColors = [c.orange, c.blue, c.blue, c.blue, c.muted, c.blue];
const tokFills = [c.softOrange, c.pale, c.pale, c.pale, c.gray, c.pale];

for (let i = 0; i < 6; i++) {
  const tx = 420 + i * 95;
  const ty = 180;
  f1 += rect(tx, ty, 80, 45, tokFills[i], tokColors[i], 'stroke-width="1.5"');
  f1 += text(tx + 40, ty + 28, tokens[i], 15, `text-anchor="middle" font-weight="600" fill="${tokColors[i]}"`);

  // Pos Embedding below
  f1 += text(tx + 40, ty + 68, '+', 18, 'text-anchor="middle" font-weight="600" fill="' + c.muted + '"');
  f1 += rect(tx, ty + 80, 80, 40, c.warmBg, c.orange, 'stroke-dasharray="3,3"');
  f1 += text(tx + 40, ty + 105, (['Pos 0', 'Pos 1', 'Pos 2', 'Pos 3', '...', 'Pos N'][i]), 13, 'text-anchor="middle" font-weight="500" fill="' + c.orange + '"');
}

f1 += arrow('M705 315 V355');
f1 += rect(420, 355, 570, 50, 'white', c.purple);
f1 += text(705, 386, '最终进入 Encoder 的序列: (B, N+1, D)    (D = 模型隐藏维度)', 16, 'text-anchor="middle" font-weight="600" fill="' + c.purple + '"');

// Arrow down to Step 3
f1 += arrow('M705 425 V475');

// 3. Standard Transformer Encoder
f1 += rect(180, 475, 700, 175, '#f4fbf7', c.green);
f1 += text(530, 505, '③ 标准 Transformer Encoder（复用 Attention 与 FFN 积木）', 20, 'text-anchor="middle" font-weight="600" fill="' + c.green + '"');

f1 += rect(210, 525, 640, 105, 'white', c.line);
f1 += lines(530, 555, [
  '• 堆叠 L 层标准 Pre-LN Transformer Block（双向 Multi-Head Self-Attention + FFN）',
  '• 全序列双向可见（无因果 Mask），所有 Patch Token 与 CLS Token 之间自由全局交互',
  '• 输出保持同维度特征: (B, N+1, D)'
], 15, 26, 'text-anchor="middle"');

// Arrow down to Step 4
f1 += arrow('M530 650 V695');

// 4. MLP Classification Head
f1 += rect(280, 695, 500, 140, c.warmBg, c.orange);
f1 += text(530, 725, '④ 分类头 (MLP Head)', 19, 'text-anchor="middle" font-weight="600" fill="' + c.orange + '"');

f1 += rect(310, 745, 440, 70, 'white', c.line);
f1 += text(530, 772, '提取第 0 位特征 z_cls = Output[:, 0, :] : (B, D)', 15, 'text-anchor="middle" font-weight="600"');
f1 += text(530, 798, 'logits = LayerNorm(z_cls) @ W_head + b_head', 15, 'text-anchor="middle" fill="' + c.ink + '"');

f1 += finish;
writeFileSync(join(root, 'vit-architecture.svg'), f1);

// Figure 2: vit-patch-conv.svg
let f2 = start(1060, 660, 'Patch 展平映射与 2D 卷积实现的等价性', '说明将图像切块展平后乘以权重矩阵 W 与使用核大小为 P、步长为 P 的 Conv2d 在数学与计算上的完全等价。');
f2 += text(32, 45, 'Patch Embedding 的两种等价实现：矩阵乘法与跨步卷积', 26, 'font-weight="600"');
f2 += text(32, 80, '相同权重排列、偏置与分块约定下，两种写法计算相同；速度取决于设备和实现。', 17, `fill="${c.muted}"`);

// Left Panel: Flatten + Linear
f2 += rect(40, 115, 470, 480, c.cardBg, c.line);
f2 += text(275, 150, 'A. 显式切块 + 展平 + 线性映射 (Linear)', 20, 'text-anchor="middle" font-weight="600" fill="' + c.blue + '"');

f2 += lines(65, 190, [
  '① 固定一个 batch：将 (C,H,W) 切为 N 个 (C,P,P) 图块',
  '② 展平每个图块为一维向量，长度为 (P² · C)',
  '③ 组合为矩阵 X_patch : (N, P² · C)',
  '④ 乘上投影参数 W_proj : (P² · C, D)',
  '⑤ 得到 Patch Tokens : (N, D)'
], 15, 27);

f2 += rect(65, 350, 420, 215, 'white', c.line);
f2 += text(275, 380, 'PyTorch 显式切块代码：', 15, 'text-anchor="middle" font-weight="600"');
f2 += lines(85, 412, [
  '# x: (B, C, H, W)',
  '# unfold 切块并展平：',
  'patches = x.unfold(2, P, P).unfold(3, P, P)',
  '# patches: (B, C, N_h, N_w, P, P)',
  'patches = patches.permute(0,2,3,1,4,5)',
  'patches = patches.reshape(B, -1, C*P*P)',
  '# 线性映射: (B, N, P*P*C) @ W_proj -> (B, N, D)',
  'tokens = linear_proj(patches)'
], 13, 20, 'font-family="monospace" fill="' + c.blue + '"');

// Right Panel: Conv2d
f2 += rect(550, 115, 470, 480, c.cardBg, c.line);
f2 += text(785, 150, 'B. 跨步卷积一步到位 (Conv2d 习惯实现)', 20, 'text-anchor="middle" font-weight="600" fill="' + c.green + '"');

f2 += lines(575, 190, [
  '① 采用一个 2D 卷积层：',
  '   - in_channels = C (如 3)',
  '   - out_channels = D (如 768)',
  '   - kernel_size = (P, P)',
  '   - stride = (P, P) (不重叠滑动)',
  '② 卷积输出 shape: (B, D, H/P, W/P) = (B, D, N_h, N_w)',
  '③ 展平后两轴并转置: (B, D, N) → (B, N, D)'
], 15, 27);

f2 += rect(575, 380, 420, 185, 'white', c.line);
f2 += text(785, 410, 'PyTorch Conv2d 标准实现：', 15, 'text-anchor="middle" font-weight="600"');
f2 += lines(595, 442, [
  '# conv: in=C, out=D, kernel=P, stride=P',
  'feat = conv2d(x) # (B, D, H/P, W/P)',
  '# 展平空间维并交换轴:',
  'tokens = feat.flatten(2).transpose(1, 2)',
  '# 得到 (B, N, D)；CPU/GPU 均可验证'
], 13, 22, 'font-family="monospace" fill="' + c.green + '"');

f2 += text(32, 625, '权重对应：W_proj = conv.weight.flatten(1).T；共享相同 bias，padding=0、dilation=1。', 17, 'font-weight="600"');

f2 += finish;
writeFileSync(join(root, 'vit-patch-conv.svg'), f2);

for (const name of ['vit-architecture', 'vit-patch-conv']) {
  execFileSync('rsvg-convert', ['--output', join(root, `${name}.png`), join(root, `${name}.svg`)]);
}
console.log('Successfully generated vit-architecture.png and vit-patch-conv.png');
