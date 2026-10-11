"""生成本目录 SVG/PNG；需系统命令 rsvg-convert（仅重绘图解时需要）。"""

from html import escape
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
INK, MUTED, LINE = "#182e42", "#516579", "#73869a"
BLUE, GREEN, ORANGE = "#e8f1fa", "#e9f5ee", "#fff0df"
FONT = "PingFang SC,Microsoft YaHei,Noto Sans CJK SC,sans-serif"


class Figure:
    def __init__(self, width, height, title, subtitle):
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9" fill="{LINE}"/></marker></defs>
<rect width="100%" height="100%" fill="white"/>
<g font-family="{FONT}" fill="{INK}">''']
        self.text(35, 43, title, size=28, weight=600)
        self.text(35, 79, subtitle, size=19, color=MUTED)

    def text(self, x, y, value, *, size=20, anchor="start", color=INK, weight=400):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(value)}</text>')

    def box(self, x, y, w, h, lines, fill=BLUE):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#c6d3de"/>')
        line_height = 29
        first_y = y + h/2 - (len(lines)-1)*line_height/2 + 7
        for i, line in enumerate(lines):
            self.text(x+w/2, first_y+i*line_height, line, anchor="middle")

    def arrow(self, points, dashed=False):
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        self.parts.append(f'<polyline points="{points}" fill="none" stroke="{LINE}" stroke-width="2"{dash} marker-end="url(#arrow)"/>')

    def add(self, x, y):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="22" fill="white" stroke="{LINE}" stroke-width="2"/>')
        self.text(x, y+7, "+", anchor="middle", size=27)

    def save(self, name):
        svg = HERE / f"{name}.svg"
        svg.write_text("\n".join(self.parts) + "\n</g></svg>\n")
        subprocess.run(["rsvg-convert", str(svg), "-o", str(HERE / f"{name}.png")], check=True)


def architecture():
    f = Figure(1120, 1060, "MoE 在完整因果语言模型中的位置",
               "左：TinyMoELM 的完整数据流；右：放大一个 MoETransformerBlock。默认共 2 层。")
    f.text(195, 130, "完整模型", anchor="middle", weight=600)
    boxes = [
        (155, 65, ["input_ids  (B,T)"]),
        (255, 75, ["token embedding + position", "(B,T,C)"]),
        (370, 65, ["MoETransformerBlock 0"]),
        (480, 65, ["MoETransformerBlock 1"]),
        (590, 60, ["final LayerNorm"]),
        (695, 70, ["@ W_vocab  (C,V)", "输出词表分数"]),
        (810, 65, ["logits  (B,T,V)"]),
        (920, 65, ["下一 token 标签 → 交叉熵", "L_task"]),
    ]
    for y, h, lines in boxes:
        f.box(35, y, 325, h, lines, ORANGE if "Block" in lines[0] else BLUE)
    for (y, h, _), (next_y, _, _) in zip(boxes, boxes[1:]):
        f.arrow(f"197,{y+h} 197,{next_y}")
    f.text(197, 1025, "V = 词表大小；C = 隐藏宽度", anchor="middle", size=18, color=MUTED)

    f.box(465, 155, 620, 735, [], "#f7f9fc")
    f.text(775, 194, "块内部：MoE 接替原 FFN 的位置", anchor="middle", weight=600)
    f.box(650, 225, 260, 55, ["块输入 x  (B,T,C)"])
    f.box(620, 330, 320, 75, ["norm1 → CausalSelfAttention", "每个 token 仍执行所有头"])
    f.arrow("780,280 780,330")
    f.add(780, 453)
    f.arrow("780,405 780,431")
    f.arrow("650,252 530,252 530,453 758,453")
    f.text(538, 350, "残差 x", size=18, color=MUTED)
    f.text(818, 460, "h  (B,T,C)", size=18)
    f.box(650, 510, 260, 65, ["z = norm2(h)", "(B,T,C)"])
    f.arrow("780,475 780,510")
    f.box(650, 625, 260, 65, ["SparseMoE(z)", "delta  (B,T,C)"], ORANGE)
    f.arrow("780,575 780,625")
    f.add(780, 751)
    f.arrow("780,690 780,729")
    f.arrow("780,485 585,485 585,751 758,751")
    f.text(591, 619, "残差 h", size=18, color=MUTED)
    f.box(650, 810, 260, 55, ["块输出 h + delta  (B,T,C)"])
    f.arrow("780,773 780,810")
    f.arrow("360,403 425,403 425,194 465,194", dashed=True)
    f.text(428, 360, "放大", size=18, color=MUTED)
    f.box(485, 932, 590, 65, ["每层 L_balance 取平均后，乘 alpha", "L_total = L_task + alpha × mean(L_balance)"], GREEN)
    f.arrow("910,658 1050,658 1050,932")
    f.text(1040, 916, "路由辅助项", anchor="end", size=18, color=MUTED)
    f.arrow("360,953 485,953")
    f.save("architecture")


def routing():
    f = Figure(1120, 1180, "SparseMoE.forward：同一份向量，分成两条计算路径",
               "左边得到专家编号与标量权重；右边把原向量送进 FFN。两边在加权累加时汇合。")
    f.box(390, 120, 340, 70, ["x  (B,T,C)", "① reshape → flat  (N,C)"], BLUE)
    f.text(750, 136, "N = B × T", size=19, color=MUTED)
    f.box(45, 275, 450, 75, ["② logits = flat @ W_g", "(N,C) @ (C,E) → (N,E)"], BLUE)
    f.arrow("470,190 470,230 270,230 270,275")
    f.box(45, 415, 450, 75, ["③ topk(logits, k)", "selected_scores / expert_ids  (N,k)"], BLUE)
    f.arrow("270,350 270,415")
    f.box(45, 555, 450, 75, ["softmax(selected_scores)", "weights  (N,k)，每行和为 1"], BLUE)
    f.arrow("270,490 270,555")

    f.box(650, 415, 420, 75, ["④ 按专家 e 收集原向量", "selected_x = flat[token_rows]  (M,C)"], GREEN)
    f.arrow("730,155 1030,155 1030,380 860,380 860,415")
    f.arrow("495,452 650,452")
    f.text(573, 438, "按编号分发", anchor="middle", size=18, color=MUTED)
    f.box(650, 555, 420, 75, ["expert[e](selected_x)", "C → F → C，返回 (M,C)"], GREEN)
    f.arrow("860,490 860,555")
    f.text(860, 663, "M=0 时跳过；总计处理 N×k 行", anchor="middle", size=19, color=MUTED)

    f.box(350, 735, 460, 80, ["权重 × 专家输出", "index_add 按原 token 行号累加"], ORANGE)
    f.arrow("270,630 270,775 350,775")
    f.arrow("1070,593 1090,593 1090,775 810,775")
    f.box(390, 865, 380, 65, ["⑥ 恢复为 (B,T,C)", "输出给外层残差相加"], ORANGE)
    f.arrow("580,815 580,865")

    f.box(45, 990, 1025, 140, [], "#f7f9fc")
    f.text(70, 1024, "⑤ 训练统计旁路：每个专家一个 f、一个 P", weight=600)
    f.text(70, 1061, "expert_ids → counts → f = counts / (N×k)", size=20)
    f.text(70, 1097, "logits → 全 E 项 softmax → P = 按 token 取平均", size=20)
    f.text(840, 1061, "L_balance = E × Σ fₑPₑ", anchor="middle", size=21)
    f.text(840, 1097, "P 保留到 W_g 的梯度", anchor="middle", size=19, color=MUTED)
    f.save("routing")


if __name__ == "__main__":
    architecture()
    routing()
