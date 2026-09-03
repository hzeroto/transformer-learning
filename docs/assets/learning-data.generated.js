window.LEARNING_DATA = {
  "schemaVersion": 1,
  "generatedAt": "2026-09-03T17:00:06.570Z",
  "goal": {
    "title": "从基础到独立手搓 Transformer",
    "description": "在理解数学、数据流和训练机制的基础上，独立实现、验证、排错并改造 Transformer 及常见变体。",
    "graduationCriteria": [
      "能从空文件实现核心 Transformer，不依赖高级 Transformer 封装",
      "能解释每个张量的来源、shape、数学作用和梯度路径",
      "能实现 Encoder-only、Decoder-only 和 Encoder–Decoder 三类架构",
      "能通过数值对齐、性质测试和极小数据过拟合验证实现",
      "能阅读一个新变体的结构说明并独立完成改造与实验"
    ]
  },
  "stages": [
    {
      "id": "foundation",
      "title": "基础语言：数学、张量与代码",
      "description": "建立后续所有公式和代码共同使用的语言。",
      "nodes": [
        {
          "id": "foundation.matrix-multiplication",
          "title": "普通矩阵乘法",
          "summary": "理解内维匹配、逐项点积和输出形状。",
          "prerequisites": [],
          "masteryCriteria": [
            "能判断两个二维矩阵能否相乘",
            "能解释结果中单个元素如何得到"
          ],
          "progress": {
            "status": "mastered",
            "updatedAt": "2026-09-03T11:58:07.448Z",
            "note": null,
            "evidence": [
              {
                "at": "2026-09-03T11:58:07.448Z",
                "text": "能正确使用普通矩阵乘法的内维匹配逻辑，并据此判断高阶乘法的核心二维部分"
              }
            ]
          }
        },
        {
          "id": "foundation.tensor-shape",
          "title": "张量、shape与索引",
          "summary": "把张量理解为需要多个下标访问的多维数值数组。",
          "prerequisites": [],
          "masteryCriteria": [
            "能区分轴的数量与轴的长度",
            "能从shape写出合法索引并解释各轴语义"
          ],
          "progress": {
            "status": "mastered",
            "updatedAt": "2026-09-03T16:12:50.035Z",
            "note": "进入向量化输入前完成高阶张量轴与索引语义验证；采用高信息密度任务，跳过定义复述",
            "evidence": [
              {
                "at": "2026-09-03T16:12:50.035Z",
                "text": "独立完成任意轴数张量的多轴索引与一维 offset 双向转换；指定样例、全部合法索引与全部 offset 的往返测试，以及非法 shape、索引数量和越界测试均通过"
              }
            ]
          }
        },
        {
          "id": "foundation.batch-matmul",
          "title": "批量矩阵乘法",
          "summary": "前置轴表示批次，最后两个轴执行对应的普通矩阵乘法。",
          "prerequisites": [
            "foundation.matrix-multiplication",
            "foundation.tensor-shape"
          ],
          "masteryCriteria": [
            "能把高阶矩阵乘法拆成若干次二维矩阵乘法",
            "能正确判断输出shape"
          ],
          "progress": {
            "status": "mastered",
            "updatedAt": "2026-09-03T11:58:08.049Z",
            "note": null,
            "evidence": [
              {
                "at": "2026-09-03T11:58:08.049Z",
                "text": "能把(7,3,4)@(7,4,6)解释为一个批次内7组(3,4)@(4,6)，并正确得到(7,3,6)"
              }
            ]
          }
        },
        {
          "id": "foundation.axis-transpose",
          "title": "交换张量轴",
          "summary": "理解交换两个轴相当于交换对应下标的位置。",
          "prerequisites": [
            "foundation.tensor-shape"
          ],
          "masteryCriteria": [
            "能用索引等式解释轴交换",
            "能判断交换前后的shape和元素位置"
          ],
          "progress": {
            "status": "mastered",
            "updatedAt": "2026-09-03T16:59:39.622Z",
            "note": "张量 shape 与索引已通过实现和测试验证；继续掌握交换轴的 shape 变化与精确索引关系",
            "evidence": [
              {
                "at": "2026-09-03T16:59:39.622Z",
                "text": "能正确判断 (2,3,4) 张量交换第 1、2 轴后 shape 为 (2,4,3)，正确给出 y[1,2,0] 对应 x[1,0,2]，并说明转置视图共享底层存储、修改该位置会同步影响原张量"
              }
            ]
          }
        },
        {
          "id": "foundation.broadcasting",
          "title": "广播规则",
          "summary": "理解不同shape在逐元素运算和批量矩阵乘法中的自动扩展。",
          "prerequisites": [
            "foundation.tensor-shape",
            "foundation.batch-matmul"
          ],
          "masteryCriteria": [
            "能从尾轴向前判断广播是否合法",
            "能预测广播后的shape"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "foundation.dot-product",
          "title": "向量点积与匹配程度",
          "summary": "连接点积的代数计算、几何含义和匹配分数。",
          "prerequisites": [
            "foundation.matrix-multiplication"
          ],
          "masteryCriteria": [
            "能解释点积与夹角的关系",
            "能说明点积为何可以充当匹配分数"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "foundation.softmax",
          "title": "Softmax与数值稳定",
          "summary": "把任意分数转为总和为1的非负权重，并避免指数溢出。",
          "prerequisites": [],
          "masteryCriteria": [
            "能解释Softmax的归一化轴",
            "能证明减去同一常数不改变结果",
            "能解释减最大值的数值意义"
          ],
          "progress": {
            "status": "verify",
            "updatedAt": "2026-09-03T11:58:10.537Z",
            "note": "能计算等分情况和减最大值后的结果，数值稳定原因尚需独立复述",
            "evidence": []
          }
        },
        {
          "id": "foundation.derivative",
          "title": "导数、偏导数与梯度",
          "summary": "把导数理解为局部变化率，把梯度理解为多参数敏感度集合。",
          "prerequisites": [],
          "masteryCriteria": [
            "能计算简单函数导数",
            "能区分导数、偏导数和梯度"
          ],
          "progress": {
            "status": "mastered",
            "updatedAt": "2026-09-03T11:58:08.683Z",
            "note": null,
            "evidence": [
              {
                "at": "2026-09-03T11:58:08.683Z",
                "text": "能正确求出(w*x-y)^2对w的偏导为2x(wx-y)，并把导数解释为变化率"
              }
            ]
          }
        },
        {
          "id": "foundation.chain-rule",
          "title": "链式法则与反向传播",
          "summary": "理解复合计算如何从最终误差向前传递梯度。",
          "prerequisites": [
            "foundation.derivative"
          ],
          "masteryCriteria": [
            "能手推一个多步标量计算的梯度",
            "能解释反向传播不是另一套独立公式"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "foundation.python-pytorch",
          "title": "Python与PyTorch语法桥接",
          "summary": "利用已有Go/C++能力，只补齐数值计算必需的Python和PyTorch写法。",
          "prerequisites": [
            "foundation.tensor-shape"
          ],
          "masteryCriteria": [
            "能独立阅读基础张量代码",
            "能创建、索引、变形和计算PyTorch张量"
          ],
          "progress": {
            "status": "pending"
          }
        }
      ]
    },
    {
      "id": "text-input",
      "title": "输入管线：文字如何变成向量",
      "description": "从原始文字出发，得到可以参加矩阵计算的批量向量。",
      "nodes": [
        {
          "id": "text-input.token-unit",
          "title": "文字切分单位",
          "summary": "理解为什么要把文本切成有限的可处理单位，以及字符、词和子词方案的差异。",
          "prerequisites": [],
          "masteryCriteria": [
            "能解释切分粒度的取舍",
            "能指出未知词和序列长度问题"
          ],
          "progress": {
            "status": "mastered",
            "updatedAt": "2026-09-03T13:48:48.610Z",
            "note": "从文字如何变成可计算输入开始补齐背景",
            "evidence": [
              {
                "at": "2026-09-03T13:48:48.610Z",
                "text": "能指出未收录的 playing 可由已知子词 play 和 ing 处理，并说明逐字符切分虽能处理但会产生更多 token、增加序列长度"
              }
            ]
          }
        },
        {
          "id": "text-input.vocabulary-id",
          "title": "词表与整数编号",
          "summary": "建立有限词表，把每个文字单位稳定映射为整数编号。",
          "prerequisites": [
            "text-input.token-unit"
          ],
          "masteryCriteria": [
            "能完成文本与编号序列的双向转换",
            "能解释未知、开始、结束和补齐编号"
          ],
          "progress": {
            "status": "mastered",
            "updatedAt": "2026-09-03T14:28:26.444Z",
            "note": "文字切分单位已掌握，继续学习如何把每个文字单位稳定映射为整数编号",
            "evidence": [
              {
                "at": "2026-09-03T14:28:26.444Z",
                "text": "能正确把带起止标记的 token 序列编码为 [2,4,5,6,7,3]，能指出未知单位映射为 ID 1，并明确说明 token 与普通 ID 之间必须保持固定映射"
              }
            ]
          }
        },
        {
          "id": "text-input.embedding",
          "title": "编号变向量",
          "summary": "使用一张可学习的向量表，把离散编号映射为连续特征。",
          "prerequisites": [
            "text-input.vocabulary-id",
            "foundation.tensor-shape"
          ],
          "masteryCriteria": [
            "能把该操作解释为按行查表",
            "能写出参数表和输出的shape"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "text-input.batch-padding",
          "title": "批处理与长度补齐",
          "summary": "把不同长度的文本组成规则张量，同时记录无效补齐位置。",
          "prerequisites": [
            "text-input.vocabulary-id",
            "foundation.tensor-shape"
          ],
          "masteryCriteria": [
            "能构造批量编号张量",
            "能区分真实内容和补齐位置"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "text-input.position",
          "title": "位置信息",
          "summary": "为向量补充顺序，使模型能够区分相同内容出现在不同位置。",
          "prerequisites": [
            "text-input.embedding"
          ],
          "masteryCriteria": [
            "能解释没有位置信息时会丢失什么",
            "能实现一种基础位置表示"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "text-input.complete-pipeline",
          "title": "完整输入管线",
          "summary": "把文本转换为带位置、批次和有效位置标记的向量张量。",
          "prerequisites": [
            "text-input.embedding",
            "text-input.batch-padding",
            "text-input.position"
          ],
          "masteryCriteria": [
            "能追踪一段文本在每一步的内容和shape",
            "能独立实现最小输入管线"
          ],
          "progress": {
            "status": "pending"
          }
        }
      ]
    },
    {
      "id": "learning-mechanics",
      "title": "学习机制：模型怎样从错误中更新",
      "description": "理解参数、中间结果、预测、误差和更新之间的完整闭环。",
      "nodes": [
        {
          "id": "learning-mechanics.parameter-activation",
          "title": "参数、输入与中间结果",
          "summary": "区分训练保存的参数、外部输入和每次运行临时产生的数值。",
          "prerequisites": [
            "foundation.tensor-shape"
          ],
          "masteryCriteria": [
            "能在一段计算中正确分类三类数据",
            "能解释为什么输入变化会让中间结果变化"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "learning-mechanics.forward",
          "title": "前向计算与计算图",
          "summary": "输入按照确定的数据流经过多步计算得到预测。",
          "prerequisites": [
            "learning-mechanics.parameter-activation",
            "foundation.python-pytorch"
          ],
          "masteryCriteria": [
            "能画出简单网络的数据流",
            "能标注每步张量shape"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "learning-mechanics.logits-probability",
          "title": "输出分数与概率",
          "summary": "为每个候选文字单位产生原始分数，再转换为概率。",
          "prerequisites": [
            "text-input.vocabulary-id",
            "foundation.softmax"
          ],
          "masteryCriteria": [
            "能区分原始分数和概率",
            "能解释最后一个轴为何等于词表大小"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "learning-mechanics.loss",
          "title": "损失与交叉熵",
          "summary": "用一个标量衡量当前预测与正确答案之间的差距。",
          "prerequisites": [
            "learning-mechanics.logits-probability"
          ],
          "masteryCriteria": [
            "能解释正确答案概率与损失的关系",
            "能手写稳定的交叉熵计算"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "learning-mechanics.backward-update",
          "title": "反向传播与参数更新",
          "summary": "计算每个参数对损失的影响并修改参数以降低损失。",
          "prerequisites": [
            "learning-mechanics.loss",
            "foundation.chain-rule"
          ],
          "masteryCriteria": [
            "能跟踪一次梯度产生和参数更新",
            "能用有限差分验证简单梯度"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "learning-mechanics.train-infer",
          "title": "训练与实际使用",
          "summary": "训练阶段拥有正确答案用于计算误差，实际使用阶段只能依赖输入和已生成结果。",
          "prerequisites": [
            "learning-mechanics.forward",
            "learning-mechanics.backward-update"
          ],
          "masteryCriteria": [
            "能分别画出训练和实际使用的数据流",
            "能说明参数何时改变、何时固定"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "learning-mechanics.next-token",
          "title": "预测下一个文字单位",
          "summary": "根据已有内容产生下一个单位，再把结果加入输入继续预测。",
          "prerequisites": [
            "learning-mechanics.logits-probability",
            "learning-mechanics.train-infer"
          ],
          "masteryCriteria": [
            "能解释逐步生成循环",
            "能说明训练时如何同时计算多个位置"
          ],
          "progress": {
            "status": "pending"
          }
        }
      ]
    },
    {
      "id": "attention",
      "title": "Attention：匹配、归一化与读取",
      "description": "从基础矩阵运算逐步构造完整Attention，不依赖任何架构术语。",
      "nodes": [
        {
          "id": "attention.qkv",
          "title": "Q、K、V的计算与分工",
          "summary": "从输入向量分别计算匹配请求、匹配依据和被读取内容。",
          "prerequisites": [
            "text-input.embedding",
            "learning-mechanics.parameter-activation",
            "foundation.matrix-multiplication"
          ],
          "masteryCriteria": [
            "能区分参数矩阵和动态计算结果",
            "能不用后续架构术语解释Q、K、V"
          ],
          "progress": {
            "status": "relearn",
            "updatedAt": "2026-09-03T11:58:11.140Z",
            "note": "曾在输入、参数和模型结构等先修未完成时提前引入",
            "evidence": []
          }
        },
        {
          "id": "attention.score",
          "title": "两两匹配分数",
          "summary": "通过Q乘K的转置，一次得到所有位置之间的点积分数。",
          "prerequisites": [
            "attention.qkv",
            "foundation.dot-product",
            "foundation.axis-transpose",
            "foundation.batch-matmul"
          ],
          "masteryCriteria": [
            "能解释分数矩阵每一行、列和元素的含义",
            "能正确推导shape"
          ],
          "progress": {
            "status": "relearn",
            "updatedAt": "2026-09-03T11:58:11.712Z",
            "note": "已接触Q乘K转置的shape，后续按完整先修链重学",
            "evidence": []
          }
        },
        {
          "id": "attention.scaling",
          "title": "分数缩放",
          "summary": "根据每个头的特征数缩放点积，避免Softmax过早饱和。",
          "prerequisites": [
            "attention.score",
            "foundation.softmax"
          ],
          "masteryCriteria": [
            "能从点积方差解释平方根缩放",
            "能解释不缩放对概率和梯度的影响"
          ],
          "progress": {
            "status": "relearn",
            "updatedAt": "2026-09-03T11:58:12.291Z",
            "note": "已听过平方根缩放原因，尚未建立在点积尺度和训练机制上",
            "evidence": []
          }
        },
        {
          "id": "attention.weights",
          "title": "分数变注意力权重",
          "summary": "沿被查询位置的轴使用Softmax，使每个查询位置得到一组读取比例。",
          "prerequisites": [
            "attention.scaling",
            "foundation.softmax"
          ],
          "masteryCriteria": [
            "能选择正确Softmax轴",
            "能解释每一行权重之和为何为1"
          ],
          "progress": {
            "status": "relearn",
            "updatedAt": "2026-09-03T11:58:12.895Z",
            "note": "已听过分数经Softmax变权重，尚未按依赖链验证",
            "evidence": []
          }
        },
        {
          "id": "attention.weighted-read",
          "title": "按权重读取V",
          "summary": "使用注意力权重对V进行加权求和，形成每个位置的新表示。",
          "prerequisites": [
            "attention.weights",
            "foundation.batch-matmul"
          ],
          "masteryCriteria": [
            "能解释输出长度和特征维度",
            "能说明V为什么不参与匹配分数计算"
          ],
          "progress": {
            "status": "relearn",
            "updatedAt": "2026-09-03T11:58:13.496Z",
            "note": "已听过权重读取V，尚未按依赖链验证",
            "evidence": []
          }
        },
        {
          "id": "attention.mask",
          "title": "屏蔽无效位置和未来位置",
          "summary": "在Softmax之前禁止读取补齐内容或尚未允许看到的位置。",
          "prerequisites": [
            "attention.weights",
            "text-input.batch-padding",
            "learning-mechanics.next-token"
          ],
          "masteryCriteria": [
            "能区分两类屏蔽",
            "能解释为什么屏蔽必须影响Softmax归一化"
          ],
          "progress": {
            "status": "relearn",
            "updatedAt": "2026-09-03T11:58:14.066Z",
            "note": "已提前接触屏蔽概念，等待输入补齐和逐步预测先修",
            "evidence": []
          }
        },
        {
          "id": "attention.single-head",
          "title": "单头Attention实现",
          "summary": "把投影、匹配、缩放、屏蔽、归一化和读取串成完整计算。",
          "prerequisites": [
            "attention.qkv",
            "attention.score",
            "attention.scaling",
            "attention.weights",
            "attention.weighted-read",
            "attention.mask"
          ],
          "masteryCriteria": [
            "能只用基础张量运算实现",
            "能通过数值、shape和屏蔽性质测试"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "attention.multi-head",
          "title": "多头Attention",
          "summary": "把特征拆成多组，让不同组学习不同的匹配和信息读取方式。",
          "prerequisites": [
            "attention.single-head",
            "foundation.axis-transpose"
          ],
          "masteryCriteria": [
            "能完成拆头和合头",
            "能解释头数与每头特征数的约束",
            "能实现输出投影"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "attention.source-types",
          "title": "同序列与跨序列Attention",
          "summary": "比较Q、K、V来自同一组输入和来自两组输入时的数据流。",
          "prerequisites": [
            "attention.single-head",
            "architecture.sequence-roles"
          ],
          "masteryCriteria": [
            "能明确指出每个张量的上游输入",
            "能解释两种数据流分别解决什么问题"
          ],
          "progress": {
            "status": "pending"
          }
        }
      ]
    },
    {
      "id": "block",
      "title": "Transformer 基础积木",
      "description": "在Attention之外补齐构成可训练深层网络的组件。",
      "nodes": [
        {
          "id": "block.linear",
          "title": "线性变换",
          "summary": "使用共享参数改变每个位置的特征表示。",
          "prerequisites": [
            "foundation.matrix-multiplication",
            "learning-mechanics.parameter-activation"
          ],
          "masteryCriteria": [
            "能手写前向计算",
            "能计算参数量并说明最后一轴的变化"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "block.activation",
          "title": "非线性激活",
          "summary": "打破多层线性变换可以合并为一层的限制。",
          "prerequisites": [
            "block.linear"
          ],
          "masteryCriteria": [
            "能解释为什么需要非线性",
            "能实现并比较基础激活函数"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "block.ffn",
          "title": "逐位置前馈网络",
          "summary": "每个位置独立经过同一套两层非线性变换。",
          "prerequisites": [
            "block.linear",
            "block.activation"
          ],
          "masteryCriteria": [
            "能解释它与Attention的分工",
            "能推导中间和输出shape"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "block.residual",
          "title": "残差连接",
          "summary": "把模块输入直接加入输出，保留信息并改善深层梯度传递。",
          "prerequisites": [
            "foundation.tensor-shape",
            "foundation.chain-rule"
          ],
          "masteryCriteria": [
            "能解释相加的shape约束",
            "能说明恒等路径对信息和梯度的作用"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "block.layer-norm",
          "title": "层归一化",
          "summary": "对每个位置的特征进行标准化和可学习的缩放平移。",
          "prerequisites": [
            "foundation.tensor-shape",
            "learning-mechanics.parameter-activation"
          ],
          "masteryCriteria": [
            "能选择正确归一化轴",
            "能手写计算并与参考实现对齐"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "block.dropout",
          "title": "随机失活",
          "summary": "训练时随机关闭部分路径，实际使用时保持确定性。",
          "prerequisites": [
            "learning-mechanics.train-infer"
          ],
          "masteryCriteria": [
            "能解释两个阶段行为差异",
            "能正确切换模型模式"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "block.complete",
          "title": "完整Transformer块",
          "summary": "组合Attention、前馈网络、残差和归一化形成可堆叠模块。",
          "prerequisites": [
            "attention.multi-head",
            "block.ffn",
            "block.residual",
            "block.layer-norm",
            "block.dropout"
          ],
          "masteryCriteria": [
            "能闭卷实现一个完整块",
            "能追踪所有子层的数据流和shape"
          ],
          "progress": {
            "status": "pending"
          }
        }
      ]
    },
    {
      "id": "architecture",
      "title": "完整架构与训练目标",
      "description": "先理解序列两侧各自承担的角色，再学习常见架构家族。",
      "nodes": [
        {
          "id": "architecture.sequence-roles",
          "title": "输入序列与输出序列",
          "summary": "区分被读取的源序列、正在生成的目标序列，以及只有一条序列的任务。",
          "prerequisites": [
            "text-input.complete-pipeline",
            "learning-mechanics.next-token"
          ],
          "masteryCriteria": [
            "能用翻译和语言生成分别画出输入输出",
            "能说明何时存在两条独立序列"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "architecture.encoder",
          "title": "Encoder",
          "summary": "读取完整输入序列，为每个位置产生包含上下文的表示。",
          "prerequisites": [
            "block.complete",
            "text-input.position",
            "architecture.sequence-roles"
          ],
          "masteryCriteria": [
            "能从输入到输出逐层说明数据流",
            "能解释为何通常允许读取输入两侧内容"
          ],
          "progress": {
            "status": "relearn",
            "updatedAt": "2026-09-03T11:58:14.683Z",
            "note": "名词被提前使用，等待输入管线、基础块和序列角色先修",
            "evidence": []
          }
        },
        {
          "id": "architecture.decoder",
          "title": "Decoder",
          "summary": "根据已经可见的目标内容产生用于预测下一个单位的表示。",
          "prerequisites": [
            "block.complete",
            "attention.mask",
            "learning-mechanics.next-token",
            "architecture.sequence-roles"
          ],
          "masteryCriteria": [
            "能解释未来信息为何必须屏蔽",
            "能区分训练输入和实际生成输入"
          ],
          "progress": {
            "status": "relearn",
            "updatedAt": "2026-09-03T11:58:15.276Z",
            "note": "名词被提前使用，等待训练机制、屏蔽和序列角色先修",
            "evidence": []
          }
        },
        {
          "id": "architecture.encoder-decoder",
          "title": "原版Encoder–Decoder",
          "summary": "先读取源序列，再由目标侧按需读取源表示并生成目标序列。",
          "prerequisites": [
            "architecture.encoder",
            "architecture.decoder",
            "attention.source-types"
          ],
          "masteryCriteria": [
            "能解释三处Attention的数据来源",
            "能实现小型序列转换任务"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "architecture.decoder-only",
          "title": "Decoder-only / GPT",
          "summary": "使用带未来屏蔽的模块，根据已有文本持续预测下一个单位。",
          "prerequisites": [
            "architecture.decoder",
            "learning-mechanics.next-token"
          ],
          "masteryCriteria": [
            "能画出训练和生成流程",
            "能从空文件实现小型GPT"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "architecture.encoder-only",
          "title": "Encoder-only / BERT",
          "summary": "利用输入两侧上下文完成内容理解、分类或被遮盖内容预测。",
          "prerequisites": [
            "architecture.encoder",
            "learning-mechanics.loss"
          ],
          "masteryCriteria": [
            "能解释它与生成模型的可见范围差异",
            "能实现掩码预测和分类头"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "architecture.t5",
          "title": "T5式文本到文本架构",
          "summary": "用Encoder–Decoder把分类、摘要、翻译等任务统一成文本输入和文本输出。",
          "prerequisites": [
            "architecture.encoder-decoder"
          ],
          "masteryCriteria": [
            "能解释任务统一方式",
            "能完成一个小型文本到文本任务"
          ],
          "progress": {
            "status": "pending"
          }
        }
      ]
    },
    {
      "id": "implementation",
      "title": "手搓、验证与排错",
      "description": "把理解变成可以独立重建和证明正确的实现能力。",
      "nodes": [
        {
          "id": "implementation.numpy-attention",
          "title": "NumPy单头Attention",
          "summary": "不使用自动求导或高级接口实现单头Attention。",
          "prerequisites": [
            "attention.single-head",
            "foundation.python-pytorch"
          ],
          "masteryCriteria": [
            "与参考结果数值对齐",
            "通过shape和屏蔽性质测试"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "implementation.pytorch-mha",
          "title": "PyTorch多头Attention",
          "summary": "只用基础张量运算和参数实现多头Attention。",
          "prerequisites": [
            "attention.multi-head",
            "foundation.python-pytorch"
          ],
          "masteryCriteria": [
            "禁止使用现成多头Attention接口",
            "前向和梯度与参考实现对齐"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "implementation.transformer-block",
          "title": "手写完整Transformer块",
          "summary": "把所有基础组件实现成可配置、可测试模块。",
          "prerequisites": [
            "block.complete",
            "implementation.pytorch-mha"
          ],
          "masteryCriteria": [
            "能从空文件重写",
            "能在极小数据上成功过拟合"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "implementation.mini-gpt",
          "title": "Mini-GPT",
          "summary": "实现数据处理、模型、训练、保存、恢复和文本生成。",
          "prerequisites": [
            "architecture.decoder-only",
            "implementation.transformer-block"
          ],
          "masteryCriteria": [
            "没有未来信息泄漏",
            "能让单批数据接近完全拟合",
            "能生成学习到的文本模式"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "implementation.original-transformer",
          "title": "原版Transformer",
          "summary": "实现完整的源序列到目标序列模型。",
          "prerequisites": [
            "architecture.encoder-decoder",
            "implementation.transformer-block"
          ],
          "masteryCriteria": [
            "能完成复制、反转或小型翻译任务",
            "所有补齐和未来屏蔽测试通过"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "implementation.bert",
          "title": "Mini-BERT",
          "summary": "实现双向表示、内容遮盖训练和下游分类。",
          "prerequisites": [
            "architecture.encoder-only",
            "implementation.transformer-block"
          ],
          "masteryCriteria": [
            "遮盖标签逻辑正确",
            "能在极小数据上验证学习能力"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "implementation.testing-debugging",
          "title": "测试与故障排查",
          "summary": "用数值对齐、梯度检查、性质测试和小数据过拟合证明实现正确。",
          "prerequisites": [
            "implementation.numpy-attention"
          ],
          "masteryCriteria": [
            "能定位shape、屏蔽、NaN和信息泄漏问题",
            "能为新模块自行设计测试"
          ],
          "progress": {
            "status": "pending"
          }
        }
      ]
    },
    {
      "id": "modern",
      "title": "现代常见变体与推理优化",
      "description": "在统一骨架上替换组件，理解每项改动解决的问题和代价。",
      "nodes": [
        {
          "id": "modern.pre-post-ln",
          "title": "Pre-LN与Post-LN",
          "summary": "比较归一化在残差分支中的位置以及深层训练差异。",
          "prerequisites": [
            "block.layer-norm",
            "block.residual",
            "implementation.transformer-block"
          ],
          "masteryCriteria": [
            "能切换两种实现",
            "能解释梯度路径和稳定性差异"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "modern.rmsnorm",
          "title": "RMSNorm",
          "summary": "使用均方根尺度替代完整均值方差标准化。",
          "prerequisites": [
            "block.layer-norm"
          ],
          "masteryCriteria": [
            "能手写并数值对齐",
            "能说明它省略了哪一步"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "modern.swiglu",
          "title": "SwiGLU",
          "summary": "用可学习门控替代传统前馈网络中的单一路径激活。",
          "prerequisites": [
            "block.ffn"
          ],
          "masteryCriteria": [
            "能实现门控前馈层",
            "能计算和比较参数量"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "modern.rope-alibi",
          "title": "RoPE与ALiBi",
          "summary": "通过旋转或分数偏置表达相对位置关系。",
          "prerequisites": [
            "text-input.position",
            "attention.score"
          ],
          "masteryCriteria": [
            "能实现至少一种方案",
            "能解释它如何影响位置关系和长度外推"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "modern.mqa-gqa",
          "title": "MQA与GQA",
          "summary": "减少K、V的头数，降低逐步生成时的存储和带宽成本。",
          "prerequisites": [
            "attention.multi-head"
          ],
          "masteryCriteria": [
            "能实现头的共享与分组",
            "能验证特殊配置退化为普通多头"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "modern.kv-cache",
          "title": "KV Cache",
          "summary": "复用历史位置已经计算的K和V，避免每次生成重复计算。",
          "prerequisites": [
            "learning-mechanics.next-token",
            "architecture.decoder-only",
            "attention.qkv"
          ],
          "masteryCriteria": [
            "缓存与无缓存输出严格对齐",
            "能分析计算量和存储量变化"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "modern.sampling",
          "title": "生成采样策略",
          "summary": "实现贪心、温度、top-k和top-p等候选选择策略。",
          "prerequisites": [
            "learning-mechanics.logits-probability",
            "learning-mechanics.next-token"
          ],
          "masteryCriteria": [
            "能实现各策略",
            "能解释随机性、质量和多样性的取舍"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "modern.sliding-window",
          "title": "滑动窗口Attention",
          "summary": "限制每个位置只读取局部范围，降低长序列计算量。",
          "prerequisites": [
            "attention.mask",
            "attention.single-head"
          ],
          "masteryCriteria": [
            "窗口外权重严格为零",
            "能分析时间和空间复杂度"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "modern.flash-attention",
          "title": "FlashAttention核心思想",
          "summary": "通过分块计算和在线Softmax减少显存读写，而不改变Attention数学结果。",
          "prerequisites": [
            "attention.single-head",
            "foundation.softmax",
            "implementation.testing-debugging"
          ],
          "masteryCriteria": [
            "能解释瓶颈为何不仅是计算次数",
            "能完成简化分块实现并数值对齐"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "modern.moe",
          "title": "稀疏专家模型",
          "summary": "让不同输入只激活部分前馈专家，以增加参数容量而控制单次计算量。",
          "prerequisites": [
            "block.ffn",
            "learning-mechanics.loss"
          ],
          "masteryCriteria": [
            "能实现玩具路由器和专家层",
            "能解释负载均衡问题"
          ],
          "progress": {
            "status": "pending"
          }
        }
      ]
    },
    {
      "id": "mastery",
      "title": "迁移与最终掌握",
      "description": "验证是否能离开熟悉代码，把原理迁移到新任务和新论文。",
      "nodes": [
        {
          "id": "mastery.vit",
          "title": "Vision Transformer",
          "summary": "把图像切成小块并映射成序列，复用Encoder完成分类。",
          "prerequisites": [
            "architecture.encoder",
            "text-input.position"
          ],
          "masteryCriteria": [
            "能实现图像分块与向量映射",
            "能说明它与文本输入真正不同的部分"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "mastery.closed-book",
          "title": "闭卷重建",
          "summary": "只根据接口规格，从空文件重建核心模型。",
          "prerequisites": [
            "implementation.mini-gpt",
            "implementation.original-transformer",
            "implementation.testing-debugging"
          ],
          "masteryCriteria": [
            "通过隐藏测试",
            "能逐层解释公式、shape和设计选择"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "mastery.paper-to-code",
          "title": "论文到代码",
          "summary": "从公式和结构说明识别改动点，在现有骨架中完成实现。",
          "prerequisites": [
            "mastery.closed-book",
            "modern.pre-post-ln",
            "modern.rope-alibi",
            "modern.mqa-gqa"
          ],
          "masteryCriteria": [
            "能复现一个此前未实现的变体",
            "能设计基线和消融实验"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "mastery.performance",
          "title": "参数量、计算量与性能分析",
          "summary": "从公式和实际测量分析参数量、计算量、内存、吞吐和生成延迟。",
          "prerequisites": [
            "implementation.testing-debugging",
            "modern.kv-cache",
            "modern.flash-attention"
          ],
          "masteryCriteria": [
            "能手算核心模块参数量和主要计算量",
            "能用基准测试验证优化收益"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "mastery.capstone",
          "title": "可配置Transformer毕业项目",
          "summary": "用一套代码支持多种基础架构、位置方案、归一化、前馈层和注意力头配置。",
          "prerequisites": [
            "mastery.paper-to-code",
            "mastery.performance",
            "mastery.vit"
          ],
          "masteryCriteria": [
            "能实现、训练、验证、排错并完整讲解",
            "能根据新需求选择和组合组件"
          ],
          "progress": {
            "status": "pending"
          }
        }
      ]
    }
  ],
  "progress": {
    "schemaVersion": 1,
    "currentNodeId": null,
    "updatedAt": "2026-09-03T16:59:39.622Z",
    "nodes": {
      "foundation.matrix-multiplication": {
        "status": "mastered",
        "updatedAt": "2026-09-03T11:58:07.448Z",
        "note": null,
        "evidence": [
          {
            "at": "2026-09-03T11:58:07.448Z",
            "text": "能正确使用普通矩阵乘法的内维匹配逻辑，并据此判断高阶乘法的核心二维部分"
          }
        ]
      },
      "foundation.batch-matmul": {
        "status": "mastered",
        "updatedAt": "2026-09-03T11:58:08.049Z",
        "note": null,
        "evidence": [
          {
            "at": "2026-09-03T11:58:08.049Z",
            "text": "能把(7,3,4)@(7,4,6)解释为一个批次内7组(3,4)@(4,6)，并正确得到(7,3,6)"
          }
        ]
      },
      "foundation.derivative": {
        "status": "mastered",
        "updatedAt": "2026-09-03T11:58:08.683Z",
        "note": null,
        "evidence": [
          {
            "at": "2026-09-03T11:58:08.683Z",
            "text": "能正确求出(w*x-y)^2对w的偏导为2x(wx-y)，并把导数解释为变化率"
          }
        ]
      },
      "foundation.tensor-shape": {
        "status": "mastered",
        "updatedAt": "2026-09-03T16:12:50.035Z",
        "note": "进入向量化输入前完成高阶张量轴与索引语义验证；采用高信息密度任务，跳过定义复述",
        "evidence": [
          {
            "at": "2026-09-03T16:12:50.035Z",
            "text": "独立完成任意轴数张量的多轴索引与一维 offset 双向转换；指定样例、全部合法索引与全部 offset 的往返测试，以及非法 shape、索引数量和越界测试均通过"
          }
        ]
      },
      "foundation.axis-transpose": {
        "status": "mastered",
        "updatedAt": "2026-09-03T16:59:39.622Z",
        "note": "张量 shape 与索引已通过实现和测试验证；继续掌握交换轴的 shape 变化与精确索引关系",
        "evidence": [
          {
            "at": "2026-09-03T16:59:39.622Z",
            "text": "能正确判断 (2,3,4) 张量交换第 1、2 轴后 shape 为 (2,4,3)，正确给出 y[1,2,0] 对应 x[1,0,2]，并说明转置视图共享底层存储、修改该位置会同步影响原张量"
          }
        ]
      },
      "foundation.softmax": {
        "status": "verify",
        "updatedAt": "2026-09-03T11:58:10.537Z",
        "note": "能计算等分情况和减最大值后的结果，数值稳定原因尚需独立复述",
        "evidence": []
      },
      "attention.qkv": {
        "status": "relearn",
        "updatedAt": "2026-09-03T11:58:11.140Z",
        "note": "曾在输入、参数和模型结构等先修未完成时提前引入",
        "evidence": []
      },
      "attention.score": {
        "status": "relearn",
        "updatedAt": "2026-09-03T11:58:11.712Z",
        "note": "已接触Q乘K转置的shape，后续按完整先修链重学",
        "evidence": []
      },
      "attention.scaling": {
        "status": "relearn",
        "updatedAt": "2026-09-03T11:58:12.291Z",
        "note": "已听过平方根缩放原因，尚未建立在点积尺度和训练机制上",
        "evidence": []
      },
      "attention.weights": {
        "status": "relearn",
        "updatedAt": "2026-09-03T11:58:12.895Z",
        "note": "已听过分数经Softmax变权重，尚未按依赖链验证",
        "evidence": []
      },
      "attention.weighted-read": {
        "status": "relearn",
        "updatedAt": "2026-09-03T11:58:13.496Z",
        "note": "已听过权重读取V，尚未按依赖链验证",
        "evidence": []
      },
      "attention.mask": {
        "status": "relearn",
        "updatedAt": "2026-09-03T11:58:14.066Z",
        "note": "已提前接触屏蔽概念，等待输入补齐和逐步预测先修",
        "evidence": []
      },
      "architecture.encoder": {
        "status": "relearn",
        "updatedAt": "2026-09-03T11:58:14.683Z",
        "note": "名词被提前使用，等待输入管线、基础块和序列角色先修",
        "evidence": []
      },
      "architecture.decoder": {
        "status": "relearn",
        "updatedAt": "2026-09-03T11:58:15.276Z",
        "note": "名词被提前使用，等待训练机制、屏蔽和序列角色先修",
        "evidence": []
      },
      "text-input.token-unit": {
        "status": "mastered",
        "updatedAt": "2026-09-03T13:48:48.610Z",
        "note": "从文字如何变成可计算输入开始补齐背景",
        "evidence": [
          {
            "at": "2026-09-03T13:48:48.610Z",
            "text": "能指出未收录的 playing 可由已知子词 play 和 ing 处理，并说明逐字符切分虽能处理但会产生更多 token、增加序列长度"
          }
        ]
      },
      "text-input.vocabulary-id": {
        "status": "mastered",
        "updatedAt": "2026-09-03T14:28:26.444Z",
        "note": "文字切分单位已掌握，继续学习如何把每个文字单位稳定映射为整数编号",
        "evidence": [
          {
            "at": "2026-09-03T14:28:26.444Z",
            "text": "能正确把带起止标记的 token 序列编码为 [2,4,5,6,7,3]，能指出未知单位映射为 ID 1，并明确说明 token 与普通 ID 之间必须保持固定映射"
          }
        ]
      }
    },
    "statusCounts": {
      "mastered": 7,
      "current": 0,
      "verify": 1,
      "relearn": 8,
      "pending": 52
    },
    "totalNodes": 68
  },
  "records": [
    {
      "id": "073f28f8-e0fc-454c-8f67-9533a42d2c58",
      "at": "2026-09-03T16:59:39.622Z",
      "nodeId": "foundation.axis-transpose",
      "nodeTitle": "交换张量轴",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "能正确判断 (2,3,4) 张量交换第 1、2 轴后 shape 为 (2,4,3)，正确给出 y[1,2,0] 对应 x[1,0,2]，并说明转置视图共享底层存储、修改该位置会同步影响原张量",
      "note": null
    },
    {
      "id": "b85bf2ad-4b75-461b-9dda-5ed73ec73ea1",
      "at": "2026-09-03T16:27:43.433Z",
      "nodeId": "foundation.axis-transpose",
      "nodeTitle": "交换张量轴",
      "action": "current",
      "fromStatus": "verify",
      "toStatus": "current",
      "evidence": null,
      "note": "张量 shape 与索引已通过实现和测试验证；继续掌握交换轴的 shape 变化与精确索引关系"
    },
    {
      "id": "2de14062-98cb-43ab-9463-a42e5c58d9d7",
      "at": "2026-09-03T16:12:50.035Z",
      "nodeId": "foundation.tensor-shape",
      "nodeTitle": "张量、shape与索引",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "独立完成任意轴数张量的多轴索引与一维 offset 双向转换；指定样例、全部合法索引与全部 offset 的往返测试，以及非法 shape、索引数量和越界测试均通过",
      "note": null
    },
    {
      "id": "b2b08f61-5658-477d-818a-572b3ac1a457",
      "at": "2026-09-03T14:28:26.715Z",
      "nodeId": "foundation.tensor-shape",
      "nodeTitle": "张量、shape与索引",
      "action": "current",
      "fromStatus": "verify",
      "toStatus": "current",
      "evidence": null,
      "note": "进入向量化输入前完成高阶张量轴与索引语义验证；采用高信息密度任务，跳过定义复述"
    },
    {
      "id": "6cf1bb0d-8314-4a21-ba53-cedc3494053a",
      "at": "2026-09-03T14:28:26.444Z",
      "nodeId": "text-input.vocabulary-id",
      "nodeTitle": "词表与整数编号",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "能正确把带起止标记的 token 序列编码为 [2,4,5,6,7,3]，能指出未知单位映射为 ID 1，并明确说明 token 与普通 ID 之间必须保持固定映射",
      "note": null
    },
    {
      "id": "1f0b9432-373a-47ce-b02b-b86e5a5c37dd",
      "at": "2026-09-03T13:49:11.971Z",
      "nodeId": "text-input.vocabulary-id",
      "nodeTitle": "词表与整数编号",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "文字切分单位已掌握，继续学习如何把每个文字单位稳定映射为整数编号"
    },
    {
      "id": "f826ec94-2841-4c44-b480-3e6f87983c1a",
      "at": "2026-09-03T13:48:48.610Z",
      "nodeId": "text-input.token-unit",
      "nodeTitle": "文字切分单位",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "能指出未收录的 playing 可由已知子词 play 和 ing 处理，并说明逐字符切分虽能处理但会产生更多 token、增加序列长度",
      "note": null
    },
    {
      "id": "ab78a141-f8a1-4ef4-982a-ec6ef55b64a3",
      "at": "2026-09-03T11:58:15.853Z",
      "nodeId": "text-input.token-unit",
      "nodeTitle": "文字切分单位",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "从文字如何变成可计算输入开始补齐背景"
    },
    {
      "id": "2de76720-6d11-432c-987a-6709ed209b33",
      "at": "2026-09-03T11:58:15.276Z",
      "nodeId": "architecture.decoder",
      "nodeTitle": "Decoder",
      "action": "relearn",
      "fromStatus": "pending",
      "toStatus": "relearn",
      "evidence": null,
      "note": "名词被提前使用，等待训练机制、屏蔽和序列角色先修"
    },
    {
      "id": "e7fb2b69-a121-408d-8e52-f9d1ab7017f6",
      "at": "2026-09-03T11:58:14.683Z",
      "nodeId": "architecture.encoder",
      "nodeTitle": "Encoder",
      "action": "relearn",
      "fromStatus": "pending",
      "toStatus": "relearn",
      "evidence": null,
      "note": "名词被提前使用，等待输入管线、基础块和序列角色先修"
    },
    {
      "id": "e1680d1a-c979-4630-9f33-13840c79bdc0",
      "at": "2026-09-03T11:58:14.066Z",
      "nodeId": "attention.mask",
      "nodeTitle": "屏蔽无效位置和未来位置",
      "action": "relearn",
      "fromStatus": "pending",
      "toStatus": "relearn",
      "evidence": null,
      "note": "已提前接触屏蔽概念，等待输入补齐和逐步预测先修"
    },
    {
      "id": "0c83b6de-a73f-46b1-adfd-c634fdd6a95c",
      "at": "2026-09-03T11:58:13.496Z",
      "nodeId": "attention.weighted-read",
      "nodeTitle": "按权重读取V",
      "action": "relearn",
      "fromStatus": "pending",
      "toStatus": "relearn",
      "evidence": null,
      "note": "已听过权重读取V，尚未按依赖链验证"
    },
    {
      "id": "9f4cac0f-f000-46c6-b5f2-e7ae8d3e9bb9",
      "at": "2026-09-03T11:58:12.895Z",
      "nodeId": "attention.weights",
      "nodeTitle": "分数变注意力权重",
      "action": "relearn",
      "fromStatus": "pending",
      "toStatus": "relearn",
      "evidence": null,
      "note": "已听过分数经Softmax变权重，尚未按依赖链验证"
    },
    {
      "id": "1eed9d29-1eac-4648-813d-730544f2f979",
      "at": "2026-09-03T11:58:12.291Z",
      "nodeId": "attention.scaling",
      "nodeTitle": "分数缩放",
      "action": "relearn",
      "fromStatus": "pending",
      "toStatus": "relearn",
      "evidence": null,
      "note": "已听过平方根缩放原因，尚未建立在点积尺度和训练机制上"
    },
    {
      "id": "c47e3cb4-9fcc-4660-96d2-f665b4507a06",
      "at": "2026-09-03T11:58:11.712Z",
      "nodeId": "attention.score",
      "nodeTitle": "两两匹配分数",
      "action": "relearn",
      "fromStatus": "pending",
      "toStatus": "relearn",
      "evidence": null,
      "note": "已接触Q乘K转置的shape，后续按完整先修链重学"
    },
    {
      "id": "e4bb2089-c094-41f9-8137-207cac97d25e",
      "at": "2026-09-03T11:58:11.140Z",
      "nodeId": "attention.qkv",
      "nodeTitle": "Q、K、V的计算与分工",
      "action": "relearn",
      "fromStatus": "pending",
      "toStatus": "relearn",
      "evidence": null,
      "note": "曾在输入、参数和模型结构等先修未完成时提前引入"
    },
    {
      "id": "97920632-2b6c-4b7d-b8bb-02549e61e390",
      "at": "2026-09-03T11:58:10.537Z",
      "nodeId": "foundation.softmax",
      "nodeTitle": "Softmax与数值稳定",
      "action": "verify",
      "fromStatus": "pending",
      "toStatus": "verify",
      "evidence": null,
      "note": "能计算等分情况和减最大值后的结果，数值稳定原因尚需独立复述"
    },
    {
      "id": "72bd9c94-f197-4add-a4bb-cb614b897059",
      "at": "2026-09-03T11:58:09.869Z",
      "nodeId": "foundation.axis-transpose",
      "nodeTitle": "交换张量轴",
      "action": "verify",
      "fromStatus": "pending",
      "toStatus": "verify",
      "evidence": null,
      "note": "已接触最后两个轴交换，尚未独立用索引关系完成验证"
    },
    {
      "id": "a2b96ee0-01e9-4f6d-bdd3-7d01e069909c",
      "at": "2026-09-03T11:58:09.277Z",
      "nodeId": "foundation.tensor-shape",
      "nodeTitle": "张量、shape与索引",
      "action": "verify",
      "fromStatus": "pending",
      "toStatus": "verify",
      "evidence": null,
      "note": "已能解释常见shape，但高阶张量的轴与索引语义仍需系统确认"
    },
    {
      "id": "c283b43c-4f58-472c-a563-d08b5d760797",
      "at": "2026-09-03T11:58:08.683Z",
      "nodeId": "foundation.derivative",
      "nodeTitle": "导数、偏导数与梯度",
      "action": "master",
      "fromStatus": "pending",
      "toStatus": "mastered",
      "evidence": "能正确求出(w*x-y)^2对w的偏导为2x(wx-y)，并把导数解释为变化率",
      "note": null
    },
    {
      "id": "1b2328e2-fd36-4b2e-bac6-6ced24c93f7c",
      "at": "2026-09-03T11:58:08.049Z",
      "nodeId": "foundation.batch-matmul",
      "nodeTitle": "批量矩阵乘法",
      "action": "master",
      "fromStatus": "pending",
      "toStatus": "mastered",
      "evidence": "能把(7,3,4)@(7,4,6)解释为一个批次内7组(3,4)@(4,6)，并正确得到(7,3,6)",
      "note": null
    },
    {
      "id": "2803cf61-857c-4d1f-af55-59709103f190",
      "at": "2026-09-03T11:58:07.448Z",
      "nodeId": "foundation.matrix-multiplication",
      "nodeTitle": "普通矩阵乘法",
      "action": "master",
      "fromStatus": "pending",
      "toStatus": "mastered",
      "evidence": "能正确使用普通矩阵乘法的内维匹配逻辑，并据此判断高阶乘法的核心二维部分",
      "note": null
    }
  ]
};
