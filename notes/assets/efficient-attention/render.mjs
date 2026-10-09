// Rebuild the two exact lesson figures: node notes/assets/efficient-attention/render.mjs
// Requires librsvg's rsvg-convert; output stays beside this source.
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const root = dirname(fileURLToPath(import.meta.url));
const c = { ink: '#172b40', muted: '#526477', line: '#c8d3df', blue: '#17639b', pale: '#dcecf9', gray: '#f1f4f7', purple: '#8650aa', green: '#256846' };
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const text = (x, y, s, size = 18, options = '') => `<text x="${x}" y="${y}" font-size="${size}" ${options}>${esc(s)}</text>`;
const rect = (x, y, w, h, fill = 'white', stroke = c.line, extra = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}" ${extra}/>`;
const arrow = (d, color = c.muted) => `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" marker-end="url(#arrow)"/>`;
const lines = (x, y, a, size = 18, leading = 28, options = '') => a.map((s, i) => text(x, y + i * leading, s, size, options)).join('');
const start = (w, h, title, desc) => `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="title description"><title id="title">${esc(title)}</title><desc id="description">${esc(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${c.muted}"/></marker></defs><rect width="${w}" height="${h}" fill="white"/><g font-family="PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, sans-serif" fill="${c.ink}">`;
const finish = '</g></svg>';

function grid(x, y, window, tiled) {
  const n = 8, cell = 30;
  let out = text(x + 116, y - 42, '列：key 位置 j', 17, 'text-anchor="middle"');
  out += text(x - 39, y - 10, 'i', 17);
  for (let j = 0; j < n; j++) out += text(x + j * cell + 15, y - 12, j, 17, 'text-anchor="middle"');
  let allowedCount = 0;
  for (let i = 0; i < n; i++) {
    out += text(x - 17, y + i * cell + 21, i, 17, 'text-anchor="middle"');
    for (let j = 0; j < n; j++) {
      const allowed = j <= i && (!window || j >= i - window + 1);
      allowedCount += Number(allowed);
      out += `<rect x="${x + j * cell}" y="${y + i * cell}" width="${cell}" height="${cell}" fill="${allowed ? c.pale : c.gray}" stroke="white"/>`;
      out += text(x + j * cell + 15, y + i * cell + 21, allowed ? '1' : '·', 18, `text-anchor="middle" fill="${allowed ? c.blue : c.muted}"`);
    }
  }
  if (allowedCount !== (window ? 21 : 36)) throw new Error('Grid count mismatch');
  if (tiled) {
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      out += `<rect x="${x + j * 60}" y="${y + i * 60}" width="60" height="60" fill="none" stroke="${c.purple}" stroke-width="1.5"/>`;
    }
  }
  out += `<rect x="${x}" y="${y + 7 * cell}" width="240" height="30" fill="none" stroke="${c.green}" stroke-width="2"/>`;
  return out;
}

let a = start(1050, 727, '先区分：改变可见范围，还是改变执行方式？', '固定 batch 和 head，T=8。三张 query-key 网格分别表示全因果、W=3 的因果滑动窗口和仍然全因果的分块处理。1 为允许，点为禁止。窗口有效位置21，全因果36；分块没有删去远处历史。');
a += text(32, 45, '先区分：改变可见范围，还是改变执行方式？', 27, 'font-weight="600"');
a += text(32, 81, '固定一个 batch b、一个 head h；T = 8，无 PAD；行是 query i，列是 key j。', 19);
a += text(32, 114, '1 = 允许读取    · = 禁止读取；绿框突出最后一个 query（i = 7）。', 18, `fill="${c.muted}"`);
const px = [30, 370, 710];
const titles = ['① 全因果 · 稠密计算', '② 滑动窗口 · W = 3', '③ 全因果 · 分块计算'];
for (let i = 0; i < 3; i++) {
  a += rect(px[i], 145, 310, 463, '#ffffff');
  a += text(px[i] + 18, 179, titles[i], 21, 'font-weight="600"');
  a += grid(px[i] + 45, 259, i === 1 ? 3 : 0, i === 2);
}
a += lines(48, 532, ['最后一行读取：0 … 7', '36 个位置允许读取', '先算稠密分数：8 × 8 = 64'], 18, 28);
a += lines(388, 532, ['最后一行读取：5、6、7', '21 个位置允许读取', '若先算稠密分数：仍是 64'], 18, 28);
a += lines(728, 532, ['最后一行仍读取：0 … 7', '仍是 36 个位置允许读取', '紫框：每次处理 2 × 2 小块'], 18, 28);
a += text(32, 644, '滑窗改变“哪些 key 参与 Softmax”；真正局部读取，才跳过窗口外的计算。', 20);
a += text(32, 678, '分块保留相同权限，逐块合并结果；上图是逻辑权限图，不是实际存下的完整矩阵。', 20);
a += text(32, 710, '两种选择可以组合：窗口内的计算也可以分块。', 18, `fill="${c.muted}"`);
a += finish;
writeFileSync(join(root, 'attention-choices.svg'), a);

let b = start(1040, 1117, 'FlashAttention 改变的是中间结果的存放与复用', 'GPU 显存 HBM 与片上存储的对照。普通非融合实现将完整分数和权重矩阵写回 HBM；分块实现载入小块，在片上维护每个 query 的最大值、分母和加权和，最终写回输出，不将完整分数或权重矩阵存入 HBM。');
b += text(32, 45, 'FlashAttention 改变的是中间结果的存放与复用', 27, 'font-weight="600"');
b += text(32, 82, '固定 b、h：Q、K 的 shape 为 (T, Dk)，V、O 为 (T, Dv)；省略 batch 和 head 轴。', 18);
b += text(32, 119, 'A. 普通非融合实现：完整 S、A 在算子之间进出 HBM', 23, 'font-weight="600"');
b += rect(32, 142, 451, 337, '#f7f9fc');
b += rect(535, 142, 473, 337, '#f1f8fb');
b += text(54, 174, 'GPU 显存 / HBM：容量大', 20, 'font-weight="600"');
b += text(557, 174, 'GPU 片上：计算时暂存数据', 20, 'font-weight="600"');
const ys = [198, 266, 334, 402];
const memories = ['Q、K（V 也已在 HBM）', '分数 S：(T, T)', '权重 A：(T, T)；以及 V', '最终输出 O：(T, Dv)'];
for (let i = 0; i < ys.length; i++) {
  b += rect(55, ys[i], 404, 47, i === 1 || i === 2 ? '#fff0df' : 'white');
  b += text(76, ys[i] + 30, memories[i], 19);
}
const ops = ['S = Q @ Kᵀ / sqrt(Dk)', 'A = softmax(S + mask)', 'O = A @ V'];
for (let i = 0; i < ops.length; i++) {
  b += rect(560, ys[i], 424, 47, 'white');
  b += text(585, ys[i] + 30, ops[i], 19);
  b += arrow(`M459 ${ys[i]+23} H560`);
  b += arrow(`M772 ${ys[i]+47} V${ys[i]+56} H504 V${ys[i+1]+23} H459`);
}
b += text(488, 210, '读', 16);
b += text(512, 263, '写', 16);
b += text(488, 278, '读', 16);
b += text(512, 331, '写', 16);
b += text(488, 346, '读', 16);
b += text(512, 399, '写', 16);
b += text(558, 443, 'mask 相同，才是在比较同一个 Attention。', 17, `fill="${c.muted}"`);

b += text(32, 533, 'B. 分块与融合：小块计算、在线合并，中间大矩阵不落 HBM', 23, 'font-weight="600"');
b += rect(32, 555, 451, 453, '#f7f9fc');
b += rect(535, 555, 473, 453, '#f1f8fb');
b += text(54, 588, 'GPU 显存 / HBM', 20, 'font-weight="600"');
b += text(557, 588, 'GPU 片上存储：小而快', 20, 'font-weight="600"');
b += rect(55, 615, 404, 66, 'white');
b += lines(76, 642, ['Q、K、V 的完整输入', '按需要读入当前 query 块 / key-value 块'], 18, 25);
b += arrow('M459 648 H560');
b += rect(560, 615, 424, 66, 'white');
b += lines(578, 641, ['Qᵢ：(Bq, Dk)    Kⱼ：(Bk, Dk)', 'Vⱼ：(Bk, Dv)；Bq / Bk 是块长度'], 18, 25);
b += arrow('M772 681 V708');
b += rect(560, 710, 424, 51, 'white');
b += text(578, 742, '当前分数块 Sᵢⱼ：(Bq, Bk) + mask', 18);
b += arrow('M772 761 V790');
b += rect(560, 792, 424, 95, c.pale);
b += lines(578, 818, ['计算指数权重，在线更新每个 query：', 'm：最大分数；l：指数权重之和', 'u：(Dv,) 向量，未归一化的加权和'], 18, 26);
b += arrow('M772 887 V920');
b += rect(560, 922, 424, 58, 'white');
b += text(578, 946, '全部合法 key 块处理完后', 18);
b += text(578, 970, '每个 query 输出 Oᵢ = u / l', 18);
b += rect(55, 922, 404, 58, 'white');
b += text(76, 957, '写回 O 的当前块：(Bq, Dv)', 18);
b += arrow('M560 951 H459');
b += lines(76, 740, ['不存完整 S：(T, T)', '不存完整 A：(T, T)', '保留输入、输出及必要统计量。'], 20, 34, `fill="${c.blue}"`);
b += lines(76, 859, ['片上状态可跨 key 块继续累积；', '具体循环顺序决定输入块的复用。'], 18, 28);
b += text(32, 1042, '减少的是大中间张量的显存读写；完整精确 Attention 的二次算术量没有因此消失。', 19);
b += text(32, 1074, 'HBM 是 GPU 显存，不是 CPU 内存；图不承诺每个输入块只读一次。', 18, `fill="${c.muted}"`);
b += text(32, 1104, 'CPU 分块代码可验证算法；只有实际融合内核与硬件测量，才能验证 GPU 性能。', 18, `fill="${c.muted}"`);
b += finish;
writeFileSync(join(root, 'attention-io.svg'), b);

for (const name of ['attention-choices', 'attention-io']) {
  execFileSync('rsvg-convert', ['--output', join(root, `${name}.png`), join(root, `${name}.svg`)]);
}
console.log('Generated attention-choices.svg/.png and attention-io.svg/.png; verified grid counts: 36, 21, 36.');
