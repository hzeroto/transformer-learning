# 019：MoE 完整代码导读

本目录提供按请求写好的完整参考实现，供逐段阅读和运行；没有待填写的 TODO。

全部代码集中在 [moe.py](moe.py)，依次为：

1. `ExpertFFN` / `SparseMoE`：专家、路由、稀疏执行、加权合并和辅助损失。
2. `make_inspection_case` / `inspect_routing`：打印两个 token 的每一步中间值。
3. `MoETransformerBlock` / `TinyMoELM`：把 MoE 放回完整因果语言模型。
4. `train_demo` / `main`：小数据训练、自回归生成和运行入口。

架构图、公式与代码逐项对应的解释见 [MoE 代码导读](../../notes/moe-code-walkthrough.md)。

在项目根目录运行：

```bash
# 先看可以手算的路由与专家输出
.venv/bin/python -B -m exercises.ex019_moe.moe inspect

# 再看完整训练和生成
.venv/bin/python -B -m exercises.ex019_moe.moe train

# 也可直接执行这个文件；不指定模式会运行两部分
.venv/bin/python -B exercises/ex019_moe/moe.py

# 正确性验证
.venv/bin/python -B -m unittest tests.exercises.test_moe -v
```

仅依赖 PyTorch，CPU 即可运行。采用 ReLU 专家、Top-2、选中项内重新归一化；
所有输入位置有效，不含 PAD、容量裁剪、KV Cache 或分布式通信。
测试验证参考实现，阅读和运行本身不自动上报掌握状态。
