# 从 Block 到 mini-GPT 讲义导出说明

本次新增一份讲义与一张图，并在 `exercises/README.md` 追加一条索引。
所有文件已与容器内版本校验 md5 一致。

## 基线：可直接应用

本次 patch 的父提交是 **`6e0b4a7`**（你远端 `main` 的当前 HEAD）。
所以不需要考虑先后顺序，直接应用即可：

```bash
git am /path/to/mini-gpt-lesson.patch
git push origin main
```

作者身份为 `hzeroto <53073935+hzeroto@users.noreply.github.com>`，与仓库历史一致。

> 与前两个压缩包的关系：`dropout-lesson.zip` 与 `block-rewrite.zip` 里的内容你已经
> 自行重写并推送（`1858757`、`aa8721d`、`6e0b4a7`），那两个 patch **不要再应用**，
> 否则会与你的版本冲突。本次是基于你最新代码的新增内容。

若 `git am` 失败，可只落地文件改动、自己写 commit：

```bash
git apply --stat /path/to/mini-gpt-lesson.patch
git apply /path/to/mini-gpt-lesson.patch
```

## 方式二：直接复制文件

```bash
cp -r mini-gpt-lesson-files/. /path/to/transformer-learning/
```

`notes/block-to-mini-gpt.md` 与 `notes/assets/block-to-gpt/` 都是**新增**，无覆盖风险。
`exercises/README.md` 是修改（只在文末追加一行索引），若你本地对该文件另有改动，
请用方式一或手动补那一行。

## 文件清单

| 文件 | 状态 |
|---|---|
| `notes/block-to-mini-gpt.md` | 新增，讲义正文 |
| `notes/assets/block-to-gpt/preln-scale-growth.svg` | 新增，尺度增长图图源 |
| `notes/assets/block-to-gpt/preln-scale-growth.png` | 新增，渲染结果 |
| `exercises/README.md` | 修改，+1 行索引 |

## 讲义覆盖的内容

起点是你 `ex008/block.py` 里的一句 docstring:「本层不加 embedding、位置编码或词表投影」。
沿这个缺口把块接成模型：

1. **堆叠不需要新机制，但输出不是预测** —— 两层堆叠实跑通过，`Y2` 最后一维是 C 而非 N
2. **输出投影可复用 E** —— 权重共享使 `logit[n] = h · E[n]`，省 `C*N` 参数；
   代价是输入向量与输出判别方向被绑定。标为结构选择，非定律
3. **并行训练为何不泄漏答案** —— 扰动末位置后，更早位置输出变化严格为 0
4. **位置切片取错的静默故障** —— 误取 `P[1:4]` 使贪心预测从 token 2 变成 token 1
5. **Pre-LN 堆栈末尾缺一次归一化** —— 12 层后 RMS 由 1.12 升至 4.47，末层输出从未被
   LN 处理；补 final LayerNorm 后回到 1.00。这对应 `modern.pre-post-ln` 验收标准里
   「理解常见 Pre-LN 堆栈末尾的归一化」
6. **完整数据流与三个理解检查**

## 核对情况

全部数值**直接调用你的 `exercises/ex008_transformer_block/block.py`** 实跑得到，
不是另写的等价实现。共 20 项断言逐条比对一致（PyTorch 2.14.0、CPU float64）：

- 堆叠 shape、因果扰动为 0、增量一致性 1.39e-17
- RMS 序列 1.118 / 1.567 / 2.317 / 3.204 / 4.471，倍率 2.85
- 位置切片错误的两组 logits 与 argmax 2 vs 1
- 检查题 C 的前提：普通 list 持有块时 `parameters()` 只有 E、`state_dict` 只有 E

其中检查题 C 的实证值得一提：**loss 从 2.983694 降到 1.343946，但 4 个块的参数
完全没变**，优化器只收到 embedding 一个张量。这是「loss 下降不足以说明参数在被训练」
的直接证据。

落地后建议在仓库根目录跑一次回归（容器内为 164 项通过、无跳过）：

```bash
python3 -m unittest discover -s . -p "test_*.py"
```

## 留白：本课未做的部分

按 `map.json` 的验收标准，以下要求需要练习而非讲义，本次**没有**代做：

- `modern.pre-post-ln` 要求「在相同子层上实际切换两种排列」并用实验讨论稳定性
- `architecture.decoder-only` 要求「给出小型 GPT 各模块接口，与 mini-GPT 同一份实现验收」
- `implementation.mini-gpt` 的训练、验证集对照、保存恢复

如需练习脚手架，可用 `transformer-exercise` skill 生成 ex009（教师给接口与测试，
核心实现留给你）。

## 未改动的文件

- `learning/progress.json`、`learning/records.jsonl`、`learning/map.json`
- `docs/learn_record.md`、`docs/assets/learning-data.generated.js`

读过讲义不构成任何节点的掌握证据；状态仍以 `progress.json` 为准。

## 图解如何重新生成

尺度增长图的坐标由实际前向计算得到（调用 ex008 的块、12 层、
权重按 `1/sqrt(输入宽度)` 初始化、`torch.manual_seed(2)`），不是手绘示意图。
改动 SVG 后重渲染：

```bash
python3 -c "
import cairosvg
cairosvg.svg2png(url='preln-scale-growth.svg', write_to='preln-scale-growth.png', output_width=640)
"
```

渲染中文需系统装有 CJK 字体（如 `fonts-noto-cjk`）。
