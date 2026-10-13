// 总览图源：一个节点对应一个运算，模型计算与请求控制分栏展示。
export const overviewSize = { width: 2240, height: 3000 };

export function renderOverview() {
  const c = {
    ink: '#192e46', muted: '#61758c', line: '#8396ac', border: '#dce4ee',
    blue: '#3472bd', blueFill: '#edf4ff', teal: '#267e78', tealFill: '#eaf6f3',
    purple: '#7654b9', purpleFill: '#f2edfc', amber: '#a77325', amberFill: '#fff6e8',
    green: '#338268', greenFill: '#ecf7f0',
  };
  const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  const p = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="2240" height="3000" viewBox="0 0 2240 3000" role="img" aria-labelledby="title desc">',
    '<title id="title">Qwen3-0.6B：完整前向与生成架构</title>',
    '<desc id="desc">展开 embedding、每层 Q/K/V 投影、Q/K RMSNorm、RoPE、KV 缓存追加、因果 GQA、两条残差、SwiGLU、final RMSNorm 与词表投影；两侧为输入处理、请求状态与生成循环。</desc>',
    `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="${c.line}"/></marker></defs>`,
    '<style>text {font-family:"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif} path {stroke-linecap:round;stroke-linejoin:round}</style>',
    '<rect width="2240" height="3000" fill="#f7f9fc"/>',
  ];
  const text = (x,y,s,size=24,fill=c.ink,weight=400,anchor='middle') => p.push(
    `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`);
  const rect = (x,y,w,h,fill='white',stroke=c.border,r=18,dashed=false) => p.push(
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="2" ${dashed ? 'stroke-dasharray="8 8"' : ''}/>`);
  const edge = (d, dashed=false, arrow=true, color=c.line) => p.push(
    `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.5" ${dashed ? 'stroke-dasharray="7 7"' : ''} ${arrow ? 'marker-end="url(#arrow)"' : ''}/>`);
  const node = (x,y,w,h,title,subtitle='',kind='blue') => {
    rect(x,y,w,h,c[`${kind}Fill`],c.border,14);
    text(x+w/2,y+(subtitle ? 36 : h/2+9),title,27,c[kind],600);
    if (subtitle) text(x+w/2,y+68,subtitle,21,c.muted);
  };
  const junction = (x,y) => p.push(`<circle cx="${x}" cy="${y}" r="5" fill="${c.line}"/>`);
  const operation = (x,y,label,kind='blue') => {
    p.push(`<circle cx="${x}" cy="${y}" r="28" fill="${c[`${kind}Fill`]}" stroke="${c[kind]}" stroke-width="2"/>`);
    text(x,y+10,label,32,c[kind],500);
  };
  const heading = (x,y,number,title,subtitle) => {
    text(x,y,number,22,c.blue,600,'start');
    text(x+48,y,title,29,c.ink,600,'start');
    text(x,y+34,subtitle,20,c.muted,400,'start');
  };

  // Header and reading order.
  text(70,77,'Qwen3-0.6B',48,c.ink,700,'start');
  text(437,77,'完整前向与生成架构',42,c.ink,600,'start');
  text(73,120,'CPU float32 · B = 1 · n = 本次 token 数 · t = 历史缓存长度 · 图示为待实现的目标链路',23,c.muted,400,'start');
  heading(70,220,'01','输入','prompt 只在第一轮完整送入');
  heading(560,220,'02','forward(input_ids, weights, config, cache)','每个运算独立展开；所有层共享结构，各自使用参数');
  heading(1775,220,'03','生成循环','greedy_generate · 正输出预算');

  // The forward container holds a real graph, not paragraphs of operations.
  rect(535,260,1115,2600,'white');
  rect(565,425,1055,1975,'#fcfdff','#cbd7e6',20,true);
  text(600,470,'Qwen3 Block × 28',28,c.ink,600,'start');
  text(600,502,'以下展开第 i 层；输出依次送入下一层',21,c.muted,400,'start');

  // All edges are placed before their node faces.
  edge('M265 365 V425');
  edge('M265 525 V590');
  edge('M265 680 V765');
  edge('M460 812 H500 V317 H860');
  text(277,727,'首次 prefill',21,c.muted);
  edge('M1090 365 V550');
  junction(1090,525);
  edge('M1090 525 H595 V1740 H1062');
  text(630,1691,'保留 X',22,c.blue,500,'start');
  edge('M1090 640 V675 H775 V710');
  edge('M1090 675 V710');
  edge('M1090 675 H1405 V710');
  junction(1090,675);
  edge('M775 800 V845');
  edge('M1090 800 V845');
  edge('M775 925 V970');
  edge('M1090 925 V970');
  edge('M1090 1060 V1130');
  edge('M1405 800 V1130');
  text(1432,962,'V 不做',21,c.muted,400,'start');
  text(1432,992,'Q/K Norm',21,c.muted,400,'start');
  text(1432,1022,'或 RoPE',21,c.muted,400,'start');
  edge('M775 1060 V1330 H845');
  text(795,1308,'Q',24,c.purple,600,'start');
  edge('M1247 1235 V1270 H1090 V1295');
  text(1220,1260,'K',22,c.green,500);
  edge('M1405 1235 V1520 H1335');
  text(1435,1440,'V',22,c.green,500,'start');
  edge('M1090 1365 V1390');
  edge('M1090 1460 V1485');
  edge('M1090 1555 V1580');
  edge('M1090 1655 V1712');
  edge('M1090 1768 V1810');
  junction(1090,1790);
  edge('M1090 1790 H1580 V2350 H1118');
  text(1538,2283,'保留 U',21,c.amber,500,'end');
  edge('M1090 1895 V1930 H865 V1960');
  edge('M1090 1930 H1320 V1960');
  junction(1090,1930);
  edge('M865 2040 V2080');
  edge('M865 2150 V2190 H1062');
  edge('M1320 2040 V2190 H1118');
  edge('M1090 2218 V2245');
  edge('M1090 2315 V2322');
  edge('M1090 2378 V2460');
  text(1120,2430,'第 28 层输出',21,c.muted,400,'start');
  edge('M1090 2545 V2600');
  edge('M1090 2685 V2740');

  // Shared E is a parameter connection, not another activation path.
  node(70,1510,390,115,'固定参数与配置','启动时从本地文件加载','blue');
  text(265,1603,'weights + config',22,c.muted);
  text(75,1678,'各层使用自己的投影与归一化权重',21,c.muted,400,'start');
  text(75,1720,'E 同时用于 embedding 与词表投影',21,c.muted,400,'start');
  edge('M460 1560 H565',true);
  text(75,1805,'实线：张量流向',21,c.muted,400,'start');
  text(75,1840,'虚线：参数 / 位置 / 缓存状态',21,c.muted,400,'start');

  // Input lane.
  node(70,275,390,90,'用户问题','文字','blue');
  node(70,425,390,100,'对话模板 + tokenizer','encode_prompt','blue');
  node(70,590,390,90,'完整 prompt ID','(1, T)','blue');
  node(70,765,390,130,'本次 input_ids','prefill: (1,T) · decode: (1,1)','blue');
  text(265,873,'下一轮只送新预测的一个 ID',21,c.muted);
  node(70,970,390,120,'本次绝对位置','positions = t … t+n−1','purple');
  text(265,1063,'t 来自本请求的旧缓存长度',21,c.muted);
  edge('M460 1015 H645',true);
  text(73,1145,'Q 与 K 使用相同的位置',21,c.purple,500,'start');
  text(73,1181,'每层只旋转本次新增 Q/K',21,c.muted,400,'start');

  // Embedding and attention branch.
  node(860,275,460,90,'Embedding 查表','E[input_ids] → (1,n,1024)','blue');
  node(855,550,470,90,'输入 RMSNorm','input_layernorm · X: (1,n,1024)','teal');
  node(645,710,260,90,'Q 投影 + 拆头','(1,16,n,128)','blue');
  node(960,710,260,90,'K 投影 + 拆头','(1,8,n,128)','blue');
  node(1275,710,260,90,'V 投影 + 拆头','(1,8,n,128)','blue');
  node(645,845,260,80,'Q 的 RMSNorm','q_norm.weight','teal');
  node(960,845,260,80,'K 的 RMSNorm','k_norm.weight','teal');
  node(645,970,260,90,'Q 的 RoPE','位置 t … t+n−1','purple');
  node(960,970,260,90,'新 K 的 RoPE','位置 t … t+n−1','purple');
  node(960,1130,575,105,'沿 token 轴拼接历史 K / V','K = [旧 K, 新 K]；V = [旧 V, 新 V]','green');
  text(1247,1225,'(1,8,t+n,128) · 旧 K 已旋转，不重转',20,c.green);

  // GQA split into scoring, normalization and value reading.
  node(845,1295,490,70,'GQA 分组 · QKᵀ / √128','','blue');
  node(845,1390,490,70,'因果 mask → Softmax','','blue');
  node(845,1485,490,70,'注意力权重 @ V','','blue');
  node(845,1580,490,75,'合头 → o_proj','2048 → 1024','blue');

  // FFN / SwiGLU is its own explicit branch pair.
  operation(1090,1740,'+');
  text(1140,1750,'U = X + Attention 更新',23,c.ink,500,'start');
  node(855,1810,470,85,'FFN 前 RMSNorm','post_attention_layernorm','teal');
  node(720,1960,290,80,'gate_proj','(1,n,3072)','amber');
  node(1175,1960,290,80,'up_proj','(1,n,3072)','amber');
  node(720,2080,290,70,'SiLU','','amber');
  operation(1090,2190,'×','amber');
  text(1135,2220,'逐元素门控',22,c.amber,500,'start');
  node(855,2245,470,70,'down_proj · 3072 → 1024','','amber');
  operation(1090,2350,'+','amber');
  text(1140,2389,'U + FFN 更新',23,c.ink,500,'start');

  // Tail of forward stays visibly outside the repeated block.
  node(855,2460,470,85,'Final RMSNorm','model.norm.weight · (1,n,1024)','teal');
  node(855,2600,470,85,'词表投影','归一化输出 @ E.T','blue');
  node(855,2740,470,85,'logits','(1,n,151936) · 本次全部位置','blue');

  // Generation is a separate control flow, fed by the model output.
  edge('M1325 2782 H1695 V320 H1775');
  node(1775,275,390,90,'取最后位置 logits','logits[:, −1, :]','blue');
  edge('M1970 365 V425');
  node(1775,425,390,90,'argmax','next_id','purple');
  edge('M1970 515 V575');
  node(1775,575,390,90,'追加到输出 ID','包含预测出的 EOS','purple');
  edge('M1970 665 V725');
  node(1775,725,390,120,'EOS / 预算用尽？','是：结束；否：继续生成','purple');
  edge('M1970 845 V925');
  text(1990,895,'是',23,c.muted,500,'start');
  node(1775,925,390,100,'整体解码 → 文本回答','只解码新增 ID','blue');
  edge('M2165 785 H2200 V165 H485 V840 H460');
  text(1115,157,'未结束：只回送 next_id，并复用本轮 new_cache',22,c.purple,500);

  // Cache has paired edges: read old state, return new state. Gaps where these
  // edges cross the logits trunk make clear that they are not junctions.
  node(1775,1130,390,150,'本请求的 28 层 KV','初次 cache = None','green');
  text(1970,1228,'每层各一对 K / V',22,c.green);
  text(1970,1261,'下一轮使用返回的 new_cache',21,c.muted);
  edge('M1775 1160 H1707',true,false);
  edge('M1683 1160 H1535',true);
  edge('M1535 1210 H1683',true,false);
  edge('M1707 1210 H1775',true);
  text(1656,1144,'旧 KV',19,c.green);
  text(1656,1250,'新 KV',19,c.green);
  text(1778,1345,'只缓存 8 个 KV 头',23,c.green,500,'start');
  text(1778,1384,'历史 K 已归一化并旋转',21,c.muted,400,'start');
  text(1778,1420,'V 只投影，不做 RoPE',21,c.muted,400,'start');

  text(70,2895,'阅读主线',23,c.ink,600,'start');
  text(70,2935,'输入 ID → Embedding → 28 层 Block → Final RMSNorm → 词表 logits → 新 ID',24,c.muted,400,'start');
  text(70,2974,'prefill 与 decode 走同一张计算图；区别只有本次输入长度 n、位置偏移 t 和已有缓存。',22,c.muted,400,'start');
  p.push('</svg>');
  return p.join('\n')+'\n';
}
