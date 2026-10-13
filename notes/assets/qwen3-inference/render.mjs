// 静态课程图：用真实接口与 checkpoint 形状绘制；不执行或代写模型。
// 重绘：node notes/assets/qwen3-inference/render.mjs（需要 rsvg-convert）。
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

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
    execFileSync('rsvg-convert',[svg,'-o',join(dir,`${name}.png`)]);
    console.log(`${name}: ${width} x ${height}`);
  }};
}

// 图一：持久参数、请求状态、token 回送是三种不同关系。
{
  const {box,text,lines,path,save}=drawing(1120,1400,
    'Qwen3-0.6B：从问题到回答',
    '目标链路 · CPU float32 · batch=1 · 正输出预算 · 紫色核心待实现');
  text(44,125,'蓝色：已提供的辅助入口',18,'start',colors.blue);
  text(350,125,'紫色：手写模型与生成逻辑',18,'start',colors.purple);
  text(680,125,'绿色：数据 / 请求状态',18,'start',colors.green);

  box(45,160,420,125,'① 用户问题 → 对话模板 → tokenizer',
    ['encode_prompt：官方非思考模板','只生成 ID，不调用 HF 模型前向'], 'given',22);
  box(720,160,355,125,'启动时加载本地文件',
    ['config.json + model.safetensors','read_config / load_weights'], 'given',23);
  path('M255 285 V325');
  box(70,325,370,95,'完整 prompt 的 ID', ['input_ids: (1,T)，T 为 prompt 长度'], 'state',22);
  path('M897 285 V325');
  box(720,325,355,95,'固定参数 weights + config', ['整个请求期间不更新参数'], 'state',23);

  path('M255 420 V451 H560 V475');
  box(360,475,400,95,'② 本次送入模型的 ID',
    ['prefill: (1,T)；decode: (1,1)'], 'core');
  text(47,608,'prefill / decode 共用 forward',20,'start');
  path('M560 570 V632');

  // 与参数供给分开绘制的显式请求缓存。
  box(45,650,255,230,'本请求的 28 层 KV',
    ['初次 cache=None','每层旧 K/V: (1,8,t,128)'], 'state',21);
  text(282,767,'传入旧 cache',19,'end');
  text(282,828,'返回 new_cache',19,'end');
  text(173,866,'下轮用新缓存；旧缓存不改',17);
  path('M300 760 H360');
  path('M360 821 H300');
  text(47,921,'新 K/V: (1,8,t+n,128)',19,'start');
  text(47,951,'K 已归一化、旋转；V 仅投影',17,'start');
  text(47,981,'每层各一对，不在请求间混用',17,'start');

  box(360,632,400,280,'③ forward：本次 n 个位置',[], 'core');
  lines(560,704,['Embedding 查表 → (1,n,1024)',
    '28 个 Qwen3 Block（见放大图）',
    'Final RMSNorm → (1,n,1024)',
    '词表投影：归一化输出 @ E.T',
    'logits: (1,n,151936)'],21,42);
  path('M897 420 V704 H760',true);
  text(918,607,'提供本次前向所需的',18);
  text(918,635,'各层参数与配置',18);
  text(911,750,'E 同时用于输入查表',18);
  text(911,777,'和最终词表投影',18);

  path('M560 912 V955');
  box(360,955,400,95,'④ 取末位 logits → argmax',
    ['next_id: (1,1)，先追加到输出 ID'], 'core',23);
  path('M560 1050 V1090');
  box(360,1090,400,105,'⑤ 命中 EOS 或用完输出预算？',
    ['EOS 来自 generation_config.json', '[151645,151643]'], 'core',22);
  // 未结束：送回新 ID；参数不重载，旧 prompt 不重送。
  path('M760 1142 H1090 V522 H760');
  text(905,993,'否：继续 decode',22,'middle',colors.purple,600);
  lines(913,1030,['只送刚预测的 next_id','并传回本轮 new_cache','不重复送 prompt 末尾'],19,30);
  path('M560 1195 V1244');
  text(589,1228,'是',19,'start');
  box(360,1244,400,95,'⑥ 整段 decode → 显示回答',
    ['仅解码新增 ID；结束后不再跑 forward'], 'given',22);
  text(44,1373,'t = 已缓存长度；n = 本次输入长度。第一次 prefill 的末位 logits 就产生第一个新 token。',20,'start');
  save('overview');
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
