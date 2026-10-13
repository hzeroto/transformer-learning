// 静态课程图：用真实接口与 checkpoint 形状绘制；不执行或代写模型。
// 重绘：node notes/assets/qwen3-inference/render.mjs（需要 rsvg-convert）。
// 仅生成 SVG：追加 --svg-only。
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { renderOverview, overviewSize } from './overview.mjs';

const svgOnly = process.argv.includes('--svg-only');

const dir = dirname(fileURLToPath(import.meta.url));
const colors = {
  ink: '#182738', line: '#566676', muted: '#566676',
  given: '#edf4fc', core: '#f2effb', state: '#eaf6f0',
  neutral: '#f6f7f9', blue: '#285b91', purple: '#614295', green: '#267153',
};
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
function drawing(width, height, title, subtitle) {
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">`,
    `<title id="title">${esc(title)}</title><desc id="desc">${esc(subtitle)}</desc>`,
    `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="${colors.line}"/></marker></defs>`,
    '<rect width="100%" height="100%" fill="white"/>',
    `<style>text {font-family:'PingFang SC','Noto Sans CJK SC','Arial',sans-serif; fill:${colors.ink}} .mono {font-family:'Menlo','Consolas',monospace}</style>`,
  ];
  const text = (x,y,s,size=22,anchor='middle',fill=colors.ink,weight=400) => parts.push(
    `<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" font-weight="${weight}" style="fill:${fill}">${esc(s)}</text>`);
  const lines = (x,y,rows,size=20,step=28,anchor='middle') => rows.forEach((s,i)=>text(x,y+i*step,s,size,anchor));
  const box = (x,y,w,h,title,rows=[],kind='core',titleSize=23) => {
    parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${colors[kind]}" stroke="#cbd3dc" stroke-width="1.4"/>`);
    text(x+w/2,y+33,title,titleSize,'middle',colors.ink,600);
    lines(x+w/2,y+65,rows,20,28);
  };
  const path = (d,dashed=false,arrow=true) => parts.push(`<path d="${d}" fill="none" stroke="${colors.line}" stroke-width="2" ${dashed?'stroke-dasharray="7 5"':''} ${arrow?'marker-end="url(#arrow)"':''}/>`);
  const dot = (x,y,s) => {
    parts.push(`<circle cx="${x}" cy="${y}" r="25" fill="${colors.core}" stroke="${colors.purple}" stroke-width="1.6"/>`);
    text(x,y+8,s,29);
  };
  text(44,51,title,31,'start',colors.ink,600);
  text(44,88,subtitle,20,'start',colors.muted);
  return { text, lines, box, path, dot, save(name) {
    parts.push('</svg>');
    const svg = join(dir,`${name}.svg`);
    writeFileSync(svg,parts.join('\n')+'\n');
    if (!svgOnly) execFileSync('rsvg-convert',[svg,'-o',join(dir,`${name}.png`)]);
    console.log(`${name}: ${width} x ${height}`);
  }};
}

// 图一：完整前向的运算图，以及独立的输入、请求状态与生成控制。
{
  const svg = join(dir, 'overview.svg');
  writeFileSync(svg, renderOverview());
  if (!svgOnly) execFileSync('rsvg-convert', [svg, '-o', join(dir, 'overview.png')]);
  console.log(`overview: ${overviewSize.width} x ${overviewSize.height}`);
}

// 图二：本图一条残差旁路对应一次逐元素加法，不是拼接。
{
  const {box,text,lines,path,dot,save}=drawing(1120,2100,
    '放大：第 i 个 Qwen3 Block 如何计算',
    '固定 batch=1 · n 为本次 token 数 · t 为本层旧缓存长度 · 每层参数独立');
  box(350,135,430,95,'本层输入 X', ['(1,n,1024)：来自查表或上一层'], 'state');
  path('M565 230 V280');
  path('M350 183 H50 V1315 H635');
  text(91,1259,'保留 X',21,'start');
  box(350,280,430,95,'input_layernorm：RMSNorm', ['只沿最后 1024 维归一化'], 'core',23);

  path('M565 375 V405 H245 V440');
  path('M565 405 V440');
  path('M565 405 H885 V440');
  box(120,440,250,100,'Q 投影 + 拆头', ['(1,16,n,128)'], 'core');
  box(440,440,250,100,'K 投影 + 拆头', ['(1,8,n,128)'], 'core');
  box(760,440,250,100,'V 投影 + 拆头', ['(1,8,n,128)'], 'core');
  path('M245 540 V585');
  path('M565 540 V585');
  box(120,585,250,90,'Q 的 RMSNorm', ['gamma: (128,)'], 'core');
  box(440,585,250,90,'K 的 RMSNorm', ['gamma: (128,)'], 'core');
  text(908,615,'不做 Q/K Norm',20,'start');
  text(908,645,'不做 RoPE',20,'start');
  path('M245 675 V725');
  path('M565 675 V725');
  box(120,725,250,90,'Q 的半分 RoPE', ['位置 t..t+n-1'], 'core');
  box(440,725,250,90,'新 K 的半分 RoPE', ['位置 t..t+n-1'], 'core',22);
  path('M565 815 V860');
  path('M885 540 V860');
  box(440,860,570,115,'与本层旧 cache[i] 沿 token 轴追加',
    ['新 cache[i].k / .v：各 (1,8,t+n,128)',
     '旧 K 不再旋转；返回新缓存；只保留 8 个 KV 头'], 'state',23);
  path('M245 815 V1050 H350');
  path('M725 975 V1000');
  box(350,1000,660,160,'GQA：Q heads 0/1 → KV 0，2/3 → KV 1 …',
    ['分数 = Q @ Kᵀ / sqrt(128)',
     '屏蔽 key 位置 > query 位置，再沿 key 轴 Softmax',
     '注意力权重 @ V → (1,16,n,128)'], 'core',22);
  path('M660 1160 V1200');
  box(350,1200,660,85,'合头 (1,n,2048) → o_proj → (1,n,1024)',
    [], 'core',23);
  text(680,1264,'Q 总宽 2048；输出投影返回残差主干的 1024 维',19);
  path('M660 1285 V1290');
  dot(660,1315,'+');
  text(705,1323,'U = X + Attention 更新',22,'start');

  path('M660 1340 V1400');
  // U 绕过整个 FFN 分支，而不是只绕过归一化。
  path('M660 1368 H1065 V1870 H685');
  text(967,1815,'保留 U',21);
  box(350,1400,660,95,'post_attention_layernorm：RMSNorm',
    ['输入是 U；归一化后才进入 FFN'], 'core',23);
  path('M660 1495 V1525 H475 V1565');
  path('M660 1525 H855 V1565');
  box(300,1565,350,100,'gate_proj → SiLU', ['(1,n,3072)'], 'core');
  box(720,1565,270,100,'up_proj', ['(1,n,3072)'], 'core');
  path('M475 1665 V1720 H635');
  path('M855 1665 V1720 H685');
  dot(660,1720,'×');
  text(699,1756,'逐元素乘法：SwiGLU 的门控',20,'start');
  path('M660 1745 V1765');
  box(440,1765,440,65,'down_proj → (1,n,1024)', [], 'core',23);
  path('M660 1830 V1845');
  dot(660,1870,'+');
  text(705,1911,'本层输出 = U + FFN 更新',21,'start');
  path('M660 1895 V1950');
  box(440,1950,440,85,'传给下一层： (1,n,1024)', ['第 28 层后进入 Final RMSNorm'], 'state',23);
  text(44,2074,'“+”是逐元素残差相加；“×”是逐元素门控。各层的 K/V 来自本层输入，不来自同一张全局特征表。',19,'start');
  save('block');
}
