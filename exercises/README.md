# Transformer 学习作业

本目录集中保存需要学习者亲自完成的编程作业。每份作业使用独立目录，按
`exNNN_<主题>` 命名，例如：

```text
exercises/
└── ex001_tensor_shape_index/
    ├── README.md       # 任务、约束和完成标准
    └── tensor_index.py # 学习者填写的代码
```

为了遵守仓库的目录职责，自动验证统一放在 `tests/exercises/`，不和待完成代码
混在一起。运行某份作业前，先阅读其 `README.md`；除非任务说明另有要求，只修改
作业目录中的待完成文件，不修改测试来迎合实现。

## 作业列表

- [001：张量 shape 与索引转换](ex001_tensor_shape_index/README.md)
- [002：批处理与长度补齐](ex002_batch_padding/README.md)
- [003：内容与位置的相加表示](ex003_position_embedding/README.md)
- [004：稳定的逐组 Softmax](ex004_stable_softmax/README.md)
- [005：最小训练闭环（当前实现批量交叉熵）](ex005_training_loop/README.md)
