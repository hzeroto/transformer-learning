window.LEARNING_DATA = {
  "schemaVersion": 1,
  "generatedAt": "2026-09-07T13:51:24.012Z",
  "goal": {
    "title": "从基础到独立手搓 Transformer",
    "description": "面向后端工程师转向 AI Infra，在理解数学、数据流和训练机制的基础上，独立实现 Transformer 及常见变体，验证增量推理并用可复现实验分析执行成本。主题分组不代表授课顺序，能力关卡见 learning/roadmap.md。",
    "graduationCriteria": [
      "能从空文件实现核心 Transformer，不依赖高级 Transformer 封装",
      "能解释每个张量的来源、shape、数学作用和梯度路径",
      "能实现 Encoder-only、Decoder-only 和 Encoder–Decoder 三类架构",
      "能通过数值对齐、性质测试和极小数据过拟合验证实现",
      "能阅读一个新变体的结构说明并独立完成改造与实验",
      "能实现小型 GPT、原版 Transformer、BERT、T5 和明确配置的 LLaMA 风格模型",
      "能验证缓存推理、计算参数和KV存储成本，并用基准解释一次优化的收益或无收益"
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
            "status": "mastered",
            "updatedAt": "2026-09-04T09:13:12.273Z",
            "note": "已验证右对齐的合法性与实际索引语义，能识别shape合法但数据语义错误的广播；API名称熟练度单独在Python/PyTorch语法节点跟踪",
            "evidence": [
              {
                "at": "2026-09-04T09:13:12.273Z",
                "text": "在X为(2,2,2)、valid为(2,2)的静默错误案例中，准确判断运算不报错且实际读取valid[t,c]；澄清修正意图是用unsqueeze(-1)增加末尾单例轴，使有效位置标记沿特征轴复用，先前仅混淆API名称"
              }
            ]
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
            "status": "mastered",
            "updatedAt": "2026-09-04T13:10:03.495Z",
            "note": "Softmax基础关卡已通过原理解释、独立实现与边界测试；包含逐组归一化、平移不变性及防止整组指数下溢。非有限输入与Attention mask留待相关节点处理",
            "evidence": [
              {
                "at": "2026-09-04T13:07:15.192Z",
                "text": "独立实现 exercises/ex004_stable_softmax/stable_softmax.py：沿末轴保留单例维取最大值与求和，平移后指数归一化，不修改输入且保持自动求导路径；本题10项和全仓36项Python测试全部通过、无跳过，覆盖跨组极端分数、逐组平移不变、不同shape/dtype、允许微小概率下溢及与官方实现的梯度对齐"
              },
              {
                "at": "2026-09-04T13:10:03.495Z",
                "text": "独立写出两候选平移后的Softmax表达式，并指出分子分母可以约去公共因子e^(-c)，完成平移不变性的代数解释；结合此前稳定Softmax独立实现、本题10项及全仓36项Python测试通过，完成归一化轴、数值稳定性与计算图保持的验收"
              }
            ]
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
            "status": "mastered",
            "updatedAt": "2026-09-04T12:05:02.607Z",
            "note": "已通过多步标量链式求导与共享参数多路径累加验证；后续进入PyTorch自动求导使用，不将API熟练度与数学理解混为一谈",
            "evidence": [
              {
                "at": "2026-09-04T12:05:02.607Z",
                "text": "对a=w*x、b=a*w、L=b*b的共享参数计算，独立给出grad_b=2*b、grad_a=grad_b*w、grad_w=grad_b*a+grad_a*x，明确沿局部导数传递并累加w的两条路径贡献"
              }
            ]
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
            "status": "mastered",
            "updatedAt": "2026-09-04T12:20:06.449Z",
            "note": "基础张量代码与自动求导边界理解已验证，不代表熟悉全部Python/PyTorch API；后续新API仍先解释，梯度累积及其他机制按实际需要学习",
            "evidence": [
              {
                "at": "2026-09-04T12:20:06.449Z",
                "text": "已独立完成基础Tensor创建、切片赋值、布尔标记、内容查表和位置广播相加，相关作业全仓26项测试通过；在自动求导反例中准确判断item取数后重建Tensor不改变L，但切断w经a的路径，w.grad仅剩grad_b*a，实测为216"
              }
            ]
          }
        },
        {
          "id": "foundation.mean-variance",
          "title": "均值、方差与尺度",
          "summary": "按需建立归一化与点积缩放所需的统计语言，不扩展成完整概率论课程。",
          "prerequisites": [
            "foundation.tensor-shape"
          ],
          "masteryCriteria": [
            "能区分均值、方差和标准差并解释按轴计算的语义",
            "能说明独立或不相关假设何时允许方差相加，以及乘常数时方差如何变化"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "foundation.tensor-layout",
          "title": "张量变形与存储布局",
          "summary": "连接逻辑shape、stride、共享存储和实际复制，在拆头合头时补齐。",
          "prerequisites": [
            "foundation.axis-transpose",
            "foundation.python-pytorch"
          ],
          "masteryCriteria": [
            "能解释reshape、view与contiguous的使用条件和可能的复制",
            "能定位非连续张量的变形错误，并用索引验证head与token没有混淆"
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
            "status": "mastered",
            "updatedAt": "2026-09-04T06:59:57.153Z",
            "note": "已通过编号重映射的边界验证，理解Embedding按ID查行及重映射不改变表与输出shape",
            "evidence": [
              {
                "at": "2026-09-04T06:59:57.153Z",
                "text": "在token ID 0与2重新编号的反例中，能说明编号虽可任意选择但查表对应关系必须一致，准确要求交换E的第0与第2行以保持输出数值不变，并指出输出shape不变"
              }
            ]
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
            "status": "mastered",
            "updatedAt": "2026-09-04T08:41:41.577Z",
            "note": "已通过实现与性质测试验证批处理补齐、输出shape/dtype及真实内容与PAD位置的区分；循环生成valid逻辑正确，向量化仅为可选改进",
            "evidence": [
              {
                "at": "2026-09-04T08:41:41.577Z",
                "text": "独立完成 exercises/ex002_batch_padding/batch_padding.py 的 make_batch：按最长序列右侧补齐，正确生成 long 编号张量和 bool 有效位置标记，支持非零PAD且保留UNK与真实ID 0，不修改输入；本题8项测试及全仓18项测试在项目PyTorch环境中全部通过且无跳过"
              }
            ]
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
            "status": "mastered",
            "updatedAt": "2026-09-04T11:34:02.422Z",
            "note": "基础绝对位置表示已通过解释、实现及长度边界和不修改输入等测试；正弦位置编码与其他位置机制留待对应阶段",
            "evidence": [
              {
                "at": "2026-09-04T11:34:02.422Z",
                "text": "能解释交换token后位置向量由当前位置决定而非跟随token移动，并独立实现按token ID查内容表、按前T行取位置表及广播相加；已自行修正shape属性调用错误，位置表示8项测试和全仓26项测试全部通过且无跳过"
              }
            ]
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
        },
        {
          "id": "text-input.sinusoidal-position",
          "title": "正弦位置编码",
          "summary": "在原版Transformer分支实现固定的正弦余弦位置表示，与已掌握的可学习位置表区分。",
          "prerequisites": [
            "text-input.position",
            "foundation.broadcasting"
          ],
          "masteryCriteria": [
            "在先理解频率与位置下标后实现编码，验证shape、位置偏移和奇偶维处理约定",
            "能区分可计算更长位置和模型能可靠外推到更长序列"
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
            "status": "mastered",
            "updatedAt": "2026-09-04T11:48:36.269Z",
            "note": "模型参数与请求数据生命周期已验证；这类基础程序概念后续不再安排定义复述式检查",
            "evidence": [
              {
                "at": "2026-09-04T11:48:36.269Z",
                "text": "能将E、P、W识别为跨请求长期保留的模型参数，将ids、valid、X、Y归为请求相关数据，并指出不同输入导致输出变化并不能说明模型发生学习或参数更新"
              }
            ]
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
            "status": "mastered",
            "updatedAt": "2026-09-07T13:49:45.305Z",
            "note": "前向计算与 shape 数据流已通过同一训练闭环实现验收，不要求额外重画流程图；不据此推定完整训练或生成能力。",
            "evidence": [
              {
                "at": "2026-09-07T13:49:45.305Z",
                "text": "独立实现 forward_logits：按 input_ids 查 E 得到 (B,T,C)，在代码中正确标注 shape，并以共享 W 做矩阵乘法得到 (B,T,N)；在 train_step 中继续接入已有交叉熵得到标量 loss，保留自动求导路径且不修改前向输入。教师复跑本题 20 项及全部 68 项练习测试均通过、无跳过，验证原始分数、单位置形状、参数共享和后续求导。"
              }
            ]
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
            "status": "verify",
            "updatedAt": "2026-09-07T13:49:45.649Z",
            "note": "原始分数的代码使用与候选轴形状已验证；词表候选列语义的独立解释可结合生成或配置检查完成，不以教师重编号测试通过代替全部解释证据。",
            "evidence": [
              {
                "at": "2026-09-07T13:49:45.649Z",
                "text": "在 forward_logits 中独立使用 embedding 查表结果与 W 相乘，正确返回 (B,T,N) 原始分数而不提前 Softmax，并直接交给稳定交叉熵；实现通过前向参考及交换 embedding 行和输出列的重编号性质测试。"
              }
            ]
          }
        },
        {
          "id": "learning-mechanics.loss",
          "title": "损失与交叉熵",
          "summary": "先解释对数与正确答案概率，再从logits计算稳定交叉熵，并明确有效标签的归约方式。",
          "prerequisites": [
            "learning-mechanics.logits-probability"
          ],
          "masteryCriteria": [
            "能解释正确答案概率与负对数损失的关系",
            "能用log-sum-exp手写稳定交叉熵并与参考对齐",
            "能排除PAD标签并按有效标签数量求平均，明确全无效输入的处理约定"
          ],
          "progress": {
            "status": "mastered",
            "updatedAt": "2026-09-07T08:21:13.122Z",
            "note": "已完成批量稳定交叉熵的原理、实现和边界验证；不据此推定已掌握参数更新或完整训练闭环。",
            "evidence": [
              {
                "at": "2026-09-06T10:43:43.047Z",
                "text": "能指出以最大预测概率构造损失会奖励确定性、可能使模型对错误答案过度自信，区分它与奖励正确答案概率的目标；关于固定错误目标时标签置乱不影响损失和梯度，已通过补充讲解澄清，尚不作为独立推导证据"
              },
              {
                "at": "2026-09-06T11:15:06.239Z",
                "text": "能独立说明稳定交叉熵通过数学等价变形适应有限精度计算；对把正确答案概率下限截断到1e-8的方案，准确指出进入截断区间后损失被固定、该损失项梯度为零，而原目标应随正确答案概率继续降低而增大，因而截断改变了训练目标"
              },
              {
                "at": "2026-09-07T06:51:10.461Z",
                "text": "能指出先逐序列求平均再平均会让短序列中的单个有效 token 因分母更小而获得更大权重，判断这与所有有效 token 等权的目标不同。"
              },
              {
                "at": "2026-09-07T08:21:13.122Z",
                "text": "独立完成 exercises/ex005_training_loop/loss.py：沿候选轴平移分数并计算稳定交叉熵，按 targets gather 正确答案分数，使用 target_valid 筛选后等权平均，整个 batch 无有效标签时抛出 ValueError；未使用高级封装、不修改输入且保留梯度。本题 12 项及全部 48 项练习测试通过且无跳过，覆盖极端错误预测、标签选择、PAD 不变性、全无效输入和参考梯度对齐。结合此前负对数目标、概率截断改变目标及短序列权重偏大的独立解释，完成本节点验收。"
              }
            ]
          }
        },
        {
          "id": "learning-mechanics.backward-update",
          "title": "反向传播与参数更新",
          "summary": "计算每个参数对损失的影响并修改参数以降低损失。",
          "prerequisites": [
            "learning-mechanics.loss",
            "foundation.chain-rule",
            "foundation.python-pytorch",
            "learning-mechanics.forward"
          ],
          "masteryCriteria": [
            "能手写一次SGD更新并解释学习率、清梯度和不记录更新计算图的原因",
            "能用有限差分验证简单梯度，并定位断图与意外梯度累积"
          ],
          "progress": {
            "status": "current",
            "updatedAt": "2026-09-07T13:49:46.172Z",
            "note": "SGD 实现与训练演示已验收；下一步讲解并验证有限差分，用独立数值变化检查梯度，再通过排错验证梯度累积和计算图边界，不重写已正确实现的函数。",
            "evidence": [
              {
                "at": "2026-09-07T13:49:46.172Z",
                "text": "独立完成 train_step：先将 E.grad/W.grad 置 None，再前向并调用 backward，在 no_grad 范围内原地更新 E、W，最后返回更新前 loss.item()。通过一次与连续两次精确参考更新、旧梯度污染、学习率缩放、PAD 不变性、全无效输入不改参数及参数继续求导测试。"
              }
            ]
          }
        },
        {
          "id": "learning-mechanics.train-infer",
          "title": "训练与实际使用",
          "summary": "区分拥有监督标签的训练与逐步预测，并分别理解模型模式和自动求导开关。",
          "prerequisites": [
            "learning-mechanics.forward",
            "learning-mechanics.backward-update"
          ],
          "masteryCriteria": [
            "能分别画出训练和实际使用的数据流",
            "能说明参数何时改变，并区分eval模式与no_grad的作用"
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
            "text-input.vocabulary-id",
            "text-input.batch-padding"
          ],
          "masteryCriteria": [
            "能构造错开一位的输入与标签并正确对齐有效位置",
            "能解释逐步生成循环及训练时并行计算多个位置所需的因果可见范围"
          ],
          "progress": {
            "status": "verify",
            "updatedAt": "2026-09-07T13:49:45.478Z",
            "note": "标签构造、EOS/PAD 对齐及因果可见性解释已验证；逐步生成循环仍待同一作业后续验收。",
            "evidence": [
              {
                "at": "2026-09-07T11:49:51.133Z",
                "text": "能独立指出训练时读取下一输入位置可以直接复制正确答案，解释低训练损失与逐步生成时缺少未来答案、表现差并不矛盾；准确区分 target_valid 仅排除无效标签损失，不能限制输入数据的可见性。"
              },
              {
                "at": "2026-09-07T13:49:45.478Z",
                "text": "独立实现 prepare_next_token_batch：以 ids[:, :-1] 和 ids[:, 1:] 构造错位输入与目标，使用 valid[:, 1:] 对齐目标有效性；通过 EOS 有效而 EOS 后 PAD 无效、真实 ID 0、单位置、全无效目标及不修改输入测试。结合此前独立解释答案泄漏与损失筛选的区别，标签构造部分验收通过。"
              }
            ]
          }
        },
        {
          "id": "learning-mechanics.tiny-training-loop",
          "title": "最小可训练语言模型",
          "summary": "用当前token的Embedding预测下一token，把分数、损失、反向、更新与生成串成首个可运行闭环。",
          "prerequisites": [
            "text-input.embedding",
            "block.linear",
            "learning-mechanics.next-token",
            "learning-mechanics.backward-update",
            "learning-mechanics.train-infer"
          ],
          "masteryCriteria": [
            "能在每个输入token都有确定后继的玩具数据上降低损失并学到转移规律",
            "能验证标签对齐、PAD排除、参数更新与清梯度，定位一个注入的训练错误",
            "能解释这个基线只看当前token，无法区分需要更早上下文的任务"
          ],
          "progress": {
            "status": "verify",
            "updatedAt": "2026-09-07T13:49:45.995Z",
            "note": "最小模型的训练执行闭环已跑通；演示和测试框架由教师提供，不能据此宣告已能独立设计排错测试。生成、表达能力限制解释及独立故障定位仍待验收。",
            "evidence": [
              {
                "at": "2026-09-07T13:49:45.995Z",
                "text": "学习者完成 ex005 的标签构造、前向及 train_step 核心实现，并运行教师提供的演示；教师独立复跑后，确定后继数据的损失从 1.6260374015677406 降至 0.0034785792437921105，有效位置预测 [2,3,4,3,4] 全部正确。本题 20 项及全部 68 项练习测试通过、无跳过。"
              }
            ]
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
            "block.linear"
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
            "foundation.softmax",
            "foundation.mean-variance",
            "foundation.chain-rule"
          ],
          "masteryCriteria": [
            "能在独立、零均值及单位方差假设下解释平方根缩放，不将假设当成所有真实特征的定律",
            "能用实验解释不缩放对概率集中程度和梯度的影响"
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
            "能区分两类屏蔽，并解释屏蔽如何影响归一化",
            "能用未来扰动和PAD扰动测试验证可见范围",
            "能区分key有效性、query有效性与损失有效性，明确全屏蔽行的处理约定"
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
            "foundation.tensor-layout"
          ],
          "masteryCriteria": [
            "能用精确索引关系完成拆头和合头",
            "能解释不同参数子空间产生独立注意力分布，以及C=H×Dh是当前等宽实现的约定",
            "能实现输出投影并说明它如何混合各头的内容"
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
            "status": "verify",
            "updatedAt": "2026-09-07T13:49:45.822Z",
            "note": "无偏置线性前向已验证；参数量与配置关系保留后续综合验收，不重复低价值定义题。",
            "evidence": [
              {
                "at": "2026-09-07T13:49:45.822Z",
                "text": "在 forward_logits 中独立手写 (B,T,C) @ (C,N) 的共享无偏置线性前向，并正确标注最后一轴由 C 变 N；通过参考计算、不同 dtype 和单位置形状测试。"
              }
            ]
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
            "learning-mechanics.parameter-activation",
            "foundation.mean-variance"
          ],
          "masteryCriteria": [
            "能选择特征轴并解释总体方差、epsilon与可学习缩放平移",
            "能手写计算并验证前向和梯度与参考实现对齐"
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
            "能说明各组件的作用和组合顺序",
            "能追踪所有子层的数据流和shape；与手写块使用同一份实现验收"
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
      "description": "优先形成Decoder-only训练闭环；双序列角色与跨序列读取在Encoder–Decoder分支补齐。",
      "nodes": [
        {
          "id": "architecture.sequence-roles",
          "title": "输入序列与输出序列",
          "summary": "区分被读取的源序列、正在生成的目标序列，以及只有一条序列的任务。",
          "prerequisites": [
            "text-input.vocabulary-id",
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
            "text-input.position"
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
            "text-input.position"
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
            "attention.source-types",
            "architecture.sequence-roles"
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
            "能画出训练和生成流程并解释因果可见范围",
            "能给出小型GPT各模块的接口；与Mini-GPT使用同一份实现验收"
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
          "summary": "用Encoder–Decoder统一文本任务，并明确T5的位置偏置、归一化和片段去噪目标。",
          "prerequisites": [
            "architecture.encoder-decoder",
            "modern.rmsnorm",
            "modern.pre-post-ln"
          ],
          "masteryCriteria": [
            "能解释文本到文本形式以及片段遮盖与哨兵token目标",
            "能说明选定T5版本的相对位置分桶、归一化和前馈层配置，不能只改模型名称"
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
          "title": "NumPy单头Attention（可选）",
          "summary": "仅在需要比较数组后端时复写；默认PyTorch主线不重复此作业，也不把它作为毕业先修。",
          "optional": true,
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
            "foundation.python-pytorch",
            "implementation.testing-debugging"
          ],
          "masteryCriteria": [
            "禁止使用现成多头Attention接口，前向与梯度在声明容差内对齐",
            "通过非连续输入、不同query/key长度、拆合头索引和mask性质测试"
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
            "能根据接口从空文件实现，与block.complete合并验收",
            "能验证前向、反向和参数注册，完整模型的过拟合在Mini-GPT关卡验收"
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
            "implementation.transformer-block",
            "learning-mechanics.tiny-training-loop",
            "text-input.complete-pipeline",
            "modern.pre-post-ln"
          ],
          "masteryCriteria": [
            "未来token扰动不改变更早位置logits，PAD标签不参与损失",
            "能在明确可拟合的小批数据上接近完全拟合，并在独立验证集观察记忆与泛化差异",
            "能生成学习到的模式，保存并恢复参数、配置和训练状态，解释初始化、参数注册、优化器与模式切换"
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
            "implementation.transformer-block",
            "text-input.sinusoidal-position",
            "learning-mechanics.tiny-training-loop"
          ],
          "masteryCriteria": [
            "实现Post-LN、正弦位置、目标右移与三处Attention，并声明与原论文训练配方的简化差异",
            "在未见组合上验证复制或反转任务，排除仅记忆固定样本",
            "源序列、目标序列长度不同，补齐与未来屏蔽均正确"
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
            "implementation.transformer-block",
            "learning-mechanics.tiny-training-loop"
          ],
          "masteryCriteria": [
            "实现双向可见范围、内容/位置/句段表示及仅在选中位置计算的遮盖预测损失",
            "能区分遮盖位置选择与输入替换策略，并实现原版BERT的句对预测小任务或明确标注省略后的模型范围",
            "能在极小数据上验证学习能力，避免把双向读取与标签泄漏混淆"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "implementation.t5",
          "title": "Mini-T5",
          "summary": "在Encoder–Decoder骨架上实现选定T5版本的结构与片段去噪训练。",
          "prerequisites": [
            "architecture.t5",
            "implementation.original-transformer"
          ],
          "masteryCriteria": [
            "实现相对位置分桶偏置、选定版本的归一化与前馈层，并列明具体配置",
            "片段遮盖、哨兵token顺序、目标右移与PAD损失处理正确",
            "在小型去噪任务上训练和生成，通过跨序列读取测试"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "implementation.llama",
          "title": "LLaMA风格模型",
          "summary": "在Mini-GPT上逐项替换Pre-LN、RMSNorm、SwiGLU和RoPE，构成明确配置的模型并验证缓存。",
          "prerequisites": [
            "implementation.mini-gpt",
            "modern.rmsnorm",
            "modern.swiglu",
            "modern.rope",
            "modern.mqa-gqa",
            "modern.kv-cache"
          ],
          "masteryCriteria": [
            "每次组件替换都有独立测试和参数量核对，并说明所选版本与自定义配置的区别",
            "使用所选MHA或GQA配置完成小数据训练，不能将GQA泛称为所有LLaMA版本的固有结构",
            "RoPE位置偏移与缓存追加正确，全量和增量logits在预设容差内对齐"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "implementation.testing-debugging",
          "title": "测试与故障排查",
          "summary": "从训练闭环开始使用数值、梯度和性质测试；Attention的屏蔽与泄漏排错在对应实现中继续验收。",
          "prerequisites": [
            "foundation.python-pytorch",
            "foundation.chain-rule",
            "foundation.softmax"
          ],
          "masteryCriteria": [
            "能为简单模块自行设计有区分力的性质测试，并解释它能捕获哪种错误",
            "能结合有限差分与参考对齐定位一个注入的梯度或数值错误；小数据过拟合在训练闭环后继续使用",
            "能声明容差、随机种子和输入约定，不把教师提供的测试通过自动等同于会设计测试"
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
            "block.residual"
          ],
          "masteryCriteria": [
            "能在相同子层上切换两种排列，并理解常见Pre-LN堆栈末尾的归一化",
            "能解释梯度路径差异，用实验讨论稳定性而非宣称某种排列总是更优"
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
          "id": "modern.rope",
          "title": "旋转位置编码RoPE",
          "summary": "先补齐二维旋转所需的三角关系，再对Q和K按位置旋转并验证相对位移性质。",
          "prerequisites": [
            "text-input.position",
            "attention.score"
          ],
          "masteryCriteria": [
            "能实现成对特征旋转并验证相对位置与旋转保持长度的性质",
            "能处理生成位置偏移并解释可计算长位置不保证模型可靠外推"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "modern.rope-alibi",
          "title": "ALiBi与RoPE对照",
          "summary": "保留原节点ID；在已实现RoPE后实现按头配置的线性距离偏置，比较两种位置机制。",
          "prerequisites": [
            "modern.rope",
            "attention.mask"
          ],
          "masteryCriteria": [
            "独立实现ALiBi并验证头斜率、距离方向和因果可见范围",
            "用实验比较RoPE与ALiBi，明确两种方案均已实现而非二选一"
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
            "能实现头的共享与分组，验证KV头数等于Q头数时退化为MHA、等于1时为MQA",
            "能计算KV参数及存储量，区分临时展开与把KV复制成所有Q头后永久缓存"
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
            "implementation.mini-gpt",
            "attention.qkv"
          ],
          "masteryCriteria": [
            "关闭随机失活，在相同位置与可见范围下逐步比较缓存和全量logits，满足事先声明的dtype/device容差",
            "验证不同前缀长度、分块追加、容量边界、缓存重置和请求隔离",
            "能区分整段预填充与逐token解码，并分析计算和存储变化"
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
          "summary": "限制局部可见范围，并区分稠密mask与真正跳过窗口外计算的实现。",
          "prerequisites": [
            "attention.mask",
            "attention.single-head"
          ],
          "masteryCriteria": [
            "窗口外权重严格为零，窗口起止和序列边界符合约定；缓存联调在KV Cache掌握后验证",
            "能实现不生成完整T×T矩阵的局部读取，并与稠密窗口mask参考对齐",
            "能解释单纯加mask不会降低稠密矩阵乘法复杂度，区分操作量减少与实际运行加速"
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
            "implementation.testing-debugging",
            "systems.cost-model"
          ],
          "masteryCriteria": [
            "能解释片上存储与设备内存读写，并说明完整精确Attention的二次算术量没有自动消失",
            "能实现在线归一化的简化分块Attention并在容差内对齐",
            "能区分CPU算法验证与融合GPU内核的性能验证，不把Python分块循环当成加速实现"
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
      "id": "systems",
      "title": "模型执行与AI Infra桥接",
      "description": "从自己的模型建立资源与性能基线；正确性可在CPU验证，硬件相关性能结论必须来自相应后端实测。",
      "nodes": [
        {
          "id": "systems.cost-model",
          "title": "参数、算术量与存储账本",
          "summary": "在MHA和FFN实现后计算主要乘加与张量字节数，后续随模型和缓存逐步扩充。",
          "prerequisites": [
            "implementation.pytorch-mha",
            "block.ffn",
            "foundation.tensor-layout"
          ],
          "masteryCriteria": [
            "能推导线性层、注意力矩阵和前馈层的主要算术量，并声明乘加计数约定",
            "能为已实现的参数、激活和已学梯度分别记账；优化器额外状态与KV账本在对应节点后扩充",
            "能用元素数乘每元素字节数核对张量存储，并区分视图、实际分配和分配器预留"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "systems.benchmarking",
          "title": "可复现性能基线与Profiling",
          "summary": "在优化前记录设备、版本、dtype和workload，用热身、重复测量和执行轨迹验证瓶颈假设。",
          "prerequisites": [
            "implementation.mini-gpt",
            "systems.cost-model"
          ],
          "masteryCriteria": [
            "提供可重跑的基准，包含shape、线程/设备、热身、重复次数和计时范围",
            "能解释异步设备为何需要同步或事件计时，并用算子轨迹支持瓶颈判断",
            "在固定输入条件下报告模型延迟和吞吐，缓存加入后分别测预填充与解码；不冒充端到端服务延迟"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "systems.mixed-precision",
          "title": "数值精度与混合精度实验",
          "summary": "比较FP32、FP16和BF16的范围与精度，按算子选择精度并检查误差和梯度。",
          "prerequisites": [
            "learning-mechanics.tiny-training-loop",
            "systems.benchmarking"
          ],
          "masteryCriteria": [
            "能解释浮点范围、舍入与归约误差，区分混合精度和把整个模型强转低精度",
            "在支持的后端完成一次autocast对照，验证logits、loss和梯度；若使用FP16训练，解释梯度缩放",
            "报告后端能力及未验证项目，不将CPU或MPS结果推广为CUDA性能结论"
          ],
          "progress": {
            "status": "pending"
          }
        },
        {
          "id": "systems.serving-bridge",
          "title": "有状态推理与服务边界",
          "summary": "把后端经验连接到请求持有的KV状态、批处理与资源约束，形成后续推理服务项目的接口说明。",
          "prerequisites": [
            "modern.kv-cache",
            "modern.mqa-gqa",
            "systems.benchmarking"
          ],
          "masteryCriteria": [
            "能给出请求进入、预填充、解码、结束释放的状态流，并验证两个请求不会串用缓存",
            "能解释上下文长度与并发对KV预算的影响，区别模型计时、首token延迟和持续输出间隔",
            "能说明连续批处理与分页KV要解决的问题及后续服务实验方案；不要求在此实现生产调度器"
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
            "systems.cost-model",
            "systems.benchmarking",
            "modern.kv-cache",
            "modern.mqa-gqa",
            "modern.flash-attention"
          ],
          "masteryCriteria": [
            "能把缓存字节数与实际张量核对，并解释预填充和解码的不同成本",
            "围绕一个瓶颈做单变量改动，在同一workload与精度约定下复测正确性、内存和性能",
            "能解释加速或无收益的证据，明确硬件性能未验证的部分，不设置必须达到的加速倍数"
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
            "mastery.vit",
            "implementation.bert",
            "implementation.t5",
            "implementation.llama",
            "modern.sampling",
            "modern.sliding-window",
            "modern.moe"
          ],
          "masteryCriteria": [
            "按roadmap覆盖表展示所需架构与变体的实现、训练、性质测试和独立排错证据",
            "能根据新需求选择组件，并声明所选架构版本与训练配方简化",
            "能闭卷重建核心模型；AI Infra桥接实验单独记录，不以未知GPU条件阻塞核心正确性毕业"
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
    "currentNodeId": "learning-mechanics.backward-update",
    "updatedAt": "2026-09-07T13:49:46.172Z",
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
        "status": "mastered",
        "updatedAt": "2026-09-04T13:10:03.495Z",
        "note": "Softmax基础关卡已通过原理解释、独立实现与边界测试；包含逐组归一化、平移不变性及防止整组指数下溢。非有限输入与Attention mask留待相关节点处理",
        "evidence": [
          {
            "at": "2026-09-04T13:07:15.192Z",
            "text": "独立实现 exercises/ex004_stable_softmax/stable_softmax.py：沿末轴保留单例维取最大值与求和，平移后指数归一化，不修改输入且保持自动求导路径；本题10项和全仓36项Python测试全部通过、无跳过，覆盖跨组极端分数、逐组平移不变、不同shape/dtype、允许微小概率下溢及与官方实现的梯度对齐"
          },
          {
            "at": "2026-09-04T13:10:03.495Z",
            "text": "独立写出两候选平移后的Softmax表达式，并指出分子分母可以约去公共因子e^(-c)，完成平移不变性的代数解释；结合此前稳定Softmax独立实现、本题10项及全仓36项Python测试通过，完成归一化轴、数值稳定性与计算图保持的验收"
          }
        ]
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
      },
      "text-input.embedding": {
        "status": "mastered",
        "updatedAt": "2026-09-04T06:59:57.153Z",
        "note": "已通过编号重映射的边界验证，理解Embedding按ID查行及重映射不改变表与输出shape",
        "evidence": [
          {
            "at": "2026-09-04T06:59:57.153Z",
            "text": "在token ID 0与2重新编号的反例中，能说明编号虽可任意选择但查表对应关系必须一致，准确要求交换E的第0与第2行以保持输出数值不变，并指出输出shape不变"
          }
        ]
      },
      "text-input.batch-padding": {
        "status": "mastered",
        "updatedAt": "2026-09-04T08:41:41.577Z",
        "note": "已通过实现与性质测试验证批处理补齐、输出shape/dtype及真实内容与PAD位置的区分；循环生成valid逻辑正确，向量化仅为可选改进",
        "evidence": [
          {
            "at": "2026-09-04T08:41:41.577Z",
            "text": "独立完成 exercises/ex002_batch_padding/batch_padding.py 的 make_batch：按最长序列右侧补齐，正确生成 long 编号张量和 bool 有效位置标记，支持非零PAD且保留UNK与真实ID 0，不修改输入；本题8项测试及全仓18项测试在项目PyTorch环境中全部通过且无跳过"
          }
        ]
      },
      "foundation.broadcasting": {
        "status": "mastered",
        "updatedAt": "2026-09-04T09:13:12.273Z",
        "note": "已验证右对齐的合法性与实际索引语义，能识别shape合法但数据语义错误的广播；API名称熟练度单独在Python/PyTorch语法节点跟踪",
        "evidence": [
          {
            "at": "2026-09-04T09:13:12.273Z",
            "text": "在X为(2,2,2)、valid为(2,2)的静默错误案例中，准确判断运算不报错且实际读取valid[t,c]；澄清修正意图是用unsqueeze(-1)增加末尾单例轴，使有效位置标记沿特征轴复用，先前仅混淆API名称"
          }
        ]
      },
      "foundation.python-pytorch": {
        "status": "mastered",
        "updatedAt": "2026-09-04T12:20:06.449Z",
        "note": "基础张量代码与自动求导边界理解已验证，不代表熟悉全部Python/PyTorch API；后续新API仍先解释，梯度累积及其他机制按实际需要学习",
        "evidence": [
          {
            "at": "2026-09-04T12:20:06.449Z",
            "text": "已独立完成基础Tensor创建、切片赋值、布尔标记、内容查表和位置广播相加，相关作业全仓26项测试通过；在自动求导反例中准确判断item取数后重建Tensor不改变L，但切断w经a的路径，w.grad仅剩grad_b*a，实测为216"
          }
        ]
      },
      "text-input.position": {
        "status": "mastered",
        "updatedAt": "2026-09-04T11:34:02.422Z",
        "note": "基础绝对位置表示已通过解释、实现及长度边界和不修改输入等测试；正弦位置编码与其他位置机制留待对应阶段",
        "evidence": [
          {
            "at": "2026-09-04T11:34:02.422Z",
            "text": "能解释交换token后位置向量由当前位置决定而非跟随token移动，并独立实现按token ID查内容表、按前T行取位置表及广播相加；已自行修正shape属性调用错误，位置表示8项测试和全仓26项测试全部通过且无跳过"
          }
        ]
      },
      "learning-mechanics.parameter-activation": {
        "status": "mastered",
        "updatedAt": "2026-09-04T11:48:36.269Z",
        "note": "模型参数与请求数据生命周期已验证；这类基础程序概念后续不再安排定义复述式检查",
        "evidence": [
          {
            "at": "2026-09-04T11:48:36.269Z",
            "text": "能将E、P、W识别为跨请求长期保留的模型参数，将ids、valid、X、Y归为请求相关数据，并指出不同输入导致输出变化并不能说明模型发生学习或参数更新"
          }
        ]
      },
      "foundation.chain-rule": {
        "status": "mastered",
        "updatedAt": "2026-09-04T12:05:02.607Z",
        "note": "已通过多步标量链式求导与共享参数多路径累加验证；后续进入PyTorch自动求导使用，不将API熟练度与数学理解混为一谈",
        "evidence": [
          {
            "at": "2026-09-04T12:05:02.607Z",
            "text": "对a=w*x、b=a*w、L=b*b的共享参数计算，独立给出grad_b=2*b、grad_a=grad_b*w、grad_w=grad_b*a+grad_a*x，明确沿局部导数传递并累加w的两条路径贡献"
          }
        ]
      },
      "learning-mechanics.logits-probability": {
        "status": "verify",
        "updatedAt": "2026-09-07T13:49:45.649Z",
        "note": "原始分数的代码使用与候选轴形状已验证；词表候选列语义的独立解释可结合生成或配置检查完成，不以教师重编号测试通过代替全部解释证据。",
        "evidence": [
          {
            "at": "2026-09-07T13:49:45.649Z",
            "text": "在 forward_logits 中独立使用 embedding 查表结果与 W 相乘，正确返回 (B,T,N) 原始分数而不提前 Softmax，并直接交给稳定交叉熵；实现通过前向参考及交换 embedding 行和输出列的重编号性质测试。"
          }
        ]
      },
      "learning-mechanics.loss": {
        "status": "mastered",
        "updatedAt": "2026-09-07T08:21:13.122Z",
        "note": "已完成批量稳定交叉熵的原理、实现和边界验证；不据此推定已掌握参数更新或完整训练闭环。",
        "evidence": [
          {
            "at": "2026-09-06T10:43:43.047Z",
            "text": "能指出以最大预测概率构造损失会奖励确定性、可能使模型对错误答案过度自信，区分它与奖励正确答案概率的目标；关于固定错误目标时标签置乱不影响损失和梯度，已通过补充讲解澄清，尚不作为独立推导证据"
          },
          {
            "at": "2026-09-06T11:15:06.239Z",
            "text": "能独立说明稳定交叉熵通过数学等价变形适应有限精度计算；对把正确答案概率下限截断到1e-8的方案，准确指出进入截断区间后损失被固定、该损失项梯度为零，而原目标应随正确答案概率继续降低而增大，因而截断改变了训练目标"
          },
          {
            "at": "2026-09-07T06:51:10.461Z",
            "text": "能指出先逐序列求平均再平均会让短序列中的单个有效 token 因分母更小而获得更大权重，判断这与所有有效 token 等权的目标不同。"
          },
          {
            "at": "2026-09-07T08:21:13.122Z",
            "text": "独立完成 exercises/ex005_training_loop/loss.py：沿候选轴平移分数并计算稳定交叉熵，按 targets gather 正确答案分数，使用 target_valid 筛选后等权平均，整个 batch 无有效标签时抛出 ValueError；未使用高级封装、不修改输入且保留梯度。本题 12 项及全部 48 项练习测试通过且无跳过，覆盖极端错误预测、标签选择、PAD 不变性、全无效输入和参考梯度对齐。结合此前负对数目标、概率截断改变目标及短序列权重偏大的独立解释，完成本节点验收。"
          }
        ]
      },
      "learning-mechanics.next-token": {
        "status": "verify",
        "updatedAt": "2026-09-07T13:49:45.478Z",
        "note": "标签构造、EOS/PAD 对齐及因果可见性解释已验证；逐步生成循环仍待同一作业后续验收。",
        "evidence": [
          {
            "at": "2026-09-07T11:49:51.133Z",
            "text": "能独立指出训练时读取下一输入位置可以直接复制正确答案，解释低训练损失与逐步生成时缺少未来答案、表现差并不矛盾；准确区分 target_valid 仅排除无效标签损失，不能限制输入数据的可见性。"
          },
          {
            "at": "2026-09-07T13:49:45.478Z",
            "text": "独立实现 prepare_next_token_batch：以 ids[:, :-1] 和 ids[:, 1:] 构造错位输入与目标，使用 valid[:, 1:] 对齐目标有效性；通过 EOS 有效而 EOS 后 PAD 无效、真实 ID 0、单位置、全无效目标及不修改输入测试。结合此前独立解释答案泄漏与损失筛选的区别，标签构造部分验收通过。"
          }
        ]
      },
      "learning-mechanics.backward-update": {
        "status": "current",
        "updatedAt": "2026-09-07T13:49:46.172Z",
        "note": "SGD 实现与训练演示已验收；下一步讲解并验证有限差分，用独立数值变化检查梯度，再通过排错验证梯度累积和计算图边界，不重写已正确实现的函数。",
        "evidence": [
          {
            "at": "2026-09-07T13:49:46.172Z",
            "text": "独立完成 train_step：先将 E.grad/W.grad 置 None，再前向并调用 backward，在 no_grad 范围内原地更新 E、W，最后返回更新前 loss.item()。通过一次与连续两次精确参考更新、旧梯度污染、学习率缩放、PAD 不变性、全无效输入不改参数及参数继续求导测试。"
          }
        ]
      },
      "learning-mechanics.forward": {
        "status": "mastered",
        "updatedAt": "2026-09-07T13:49:45.305Z",
        "note": "前向计算与 shape 数据流已通过同一训练闭环实现验收，不要求额外重画流程图；不据此推定完整训练或生成能力。",
        "evidence": [
          {
            "at": "2026-09-07T13:49:45.305Z",
            "text": "独立实现 forward_logits：按 input_ids 查 E 得到 (B,T,C)，在代码中正确标注 shape，并以共享 W 做矩阵乘法得到 (B,T,N)；在 train_step 中继续接入已有交叉熵得到标量 loss，保留自动求导路径且不修改前向输入。教师复跑本题 20 项及全部 68 项练习测试均通过、无跳过，验证原始分数、单位置形状、参数共享和后续求导。"
          }
        ]
      },
      "block.linear": {
        "status": "verify",
        "updatedAt": "2026-09-07T13:49:45.822Z",
        "note": "无偏置线性前向已验证；参数量与配置关系保留后续综合验收，不重复低价值定义题。",
        "evidence": [
          {
            "at": "2026-09-07T13:49:45.822Z",
            "text": "在 forward_logits 中独立手写 (B,T,C) @ (C,N) 的共享无偏置线性前向，并正确标注最后一轴由 C 变 N；通过参考计算、不同 dtype 和单位置形状测试。"
          }
        ]
      },
      "learning-mechanics.tiny-training-loop": {
        "status": "verify",
        "updatedAt": "2026-09-07T13:49:45.995Z",
        "note": "最小模型的训练执行闭环已跑通；演示和测试框架由教师提供，不能据此宣告已能独立设计排错测试。生成、表达能力限制解释及独立故障定位仍待验收。",
        "evidence": [
          {
            "at": "2026-09-07T13:49:45.995Z",
            "text": "学习者完成 ex005 的标签构造、前向及 train_step 核心实现，并运行教师提供的演示；教师独立复跑后，确定后继数据的损失从 1.6260374015677406 降至 0.0034785792437921105，有效位置预测 [2,3,4,3,4] 全部正确。本题 20 项及全部 68 项练习测试通过、无跳过。"
          }
        ]
      }
    },
    "statusCounts": {
      "mastered": 17,
      "current": 1,
      "verify": 4,
      "relearn": 8,
      "pending": 49
    },
    "totalNodes": 79
  },
  "records": [
    {
      "id": "1948d301-b737-4e6e-b816-012e1f6b4423",
      "at": "2026-09-07T13:49:46.172Z",
      "nodeId": "learning-mechanics.backward-update",
      "nodeTitle": "反向传播与参数更新",
      "action": "current",
      "fromStatus": "current",
      "toStatus": "current",
      "evidence": "独立完成 train_step：先将 E.grad/W.grad 置 None，再前向并调用 backward，在 no_grad 范围内原地更新 E、W，最后返回更新前 loss.item()。通过一次与连续两次精确参考更新、旧梯度污染、学习率缩放、PAD 不变性、全无效输入不改参数及参数继续求导测试。",
      "note": "SGD 实现与训练演示已验收；下一步讲解并验证有限差分，用独立数值变化检查梯度，再通过排错验证梯度累积和计算图边界，不重写已正确实现的函数。"
    },
    {
      "id": "e19cfb1e-4362-4c6f-97d0-71584ed550ad",
      "at": "2026-09-07T13:49:45.995Z",
      "nodeId": "learning-mechanics.tiny-training-loop",
      "nodeTitle": "最小可训练语言模型",
      "action": "verify",
      "fromStatus": "pending",
      "toStatus": "verify",
      "evidence": "学习者完成 ex005 的标签构造、前向及 train_step 核心实现，并运行教师提供的演示；教师独立复跑后，确定后继数据的损失从 1.6260374015677406 降至 0.0034785792437921105，有效位置预测 [2,3,4,3,4] 全部正确。本题 20 项及全部 68 项练习测试通过、无跳过。",
      "note": "最小模型的训练执行闭环已跑通；演示和测试框架由教师提供，不能据此宣告已能独立设计排错测试。生成、表达能力限制解释及独立故障定位仍待验收。"
    },
    {
      "id": "c44938ac-8ae7-49ad-9d2d-1fd81532053a",
      "at": "2026-09-07T13:49:45.822Z",
      "nodeId": "block.linear",
      "nodeTitle": "线性变换",
      "action": "verify",
      "fromStatus": "pending",
      "toStatus": "verify",
      "evidence": "在 forward_logits 中独立手写 (B,T,C) @ (C,N) 的共享无偏置线性前向，并正确标注最后一轴由 C 变 N；通过参考计算、不同 dtype 和单位置形状测试。",
      "note": "无偏置线性前向已验证；参数量与配置关系保留后续综合验收，不重复低价值定义题。"
    },
    {
      "id": "a4590d26-0cfd-4381-a0e8-bb4684b94541",
      "at": "2026-09-07T13:49:45.649Z",
      "nodeId": "learning-mechanics.logits-probability",
      "nodeTitle": "输出分数与概率",
      "action": "verify",
      "fromStatus": "verify",
      "toStatus": "verify",
      "evidence": "在 forward_logits 中独立使用 embedding 查表结果与 W 相乘，正确返回 (B,T,N) 原始分数而不提前 Softmax，并直接交给稳定交叉熵；实现通过前向参考及交换 embedding 行和输出列的重编号性质测试。",
      "note": "原始分数的代码使用与候选轴形状已验证；词表候选列语义的独立解释可结合生成或配置检查完成，不以教师重编号测试通过代替全部解释证据。"
    },
    {
      "id": "dc7fdcaf-0ae8-4a0f-82fa-bfb4d1d167b1",
      "at": "2026-09-07T13:49:45.478Z",
      "nodeId": "learning-mechanics.next-token",
      "nodeTitle": "预测下一个文字单位",
      "action": "verify",
      "fromStatus": "verify",
      "toStatus": "verify",
      "evidence": "独立实现 prepare_next_token_batch：以 ids[:, :-1] 和 ids[:, 1:] 构造错位输入与目标，使用 valid[:, 1:] 对齐目标有效性；通过 EOS 有效而 EOS 后 PAD 无效、真实 ID 0、单位置、全无效目标及不修改输入测试。结合此前独立解释答案泄漏与损失筛选的区别，标签构造部分验收通过。",
      "note": "标签构造、EOS/PAD 对齐及因果可见性解释已验证；逐步生成循环仍待同一作业后续验收。"
    },
    {
      "id": "278ea025-fc62-4571-87ca-4fe8ec572741",
      "at": "2026-09-07T13:49:45.305Z",
      "nodeId": "learning-mechanics.forward",
      "nodeTitle": "前向计算与计算图",
      "action": "master",
      "fromStatus": "pending",
      "toStatus": "mastered",
      "evidence": "独立实现 forward_logits：按 input_ids 查 E 得到 (B,T,C)，在代码中正确标注 shape，并以共享 W 做矩阵乘法得到 (B,T,N)；在 train_step 中继续接入已有交叉熵得到标量 loss，保留自动求导路径且不修改前向输入。教师复跑本题 20 项及全部 68 项练习测试均通过、无跳过，验证原始分数、单位置形状、参数共享和后续求导。",
      "note": "前向计算与 shape 数据流已通过同一训练闭环实现验收，不要求额外重画流程图；不据此推定完整训练或生成能力。"
    },
    {
      "id": "af8b8a34-59e3-466c-9d57-c8ce10e167da",
      "at": "2026-09-07T12:44:48.153Z",
      "nodeId": "learning-mechanics.backward-update",
      "nodeTitle": "反向传播与参数更新",
      "action": "current",
      "fromStatus": "current",
      "toStatus": "current",
      "evidence": null,
      "note": "已进入 ex005 第二部分独立实现：training.py 提供标签构造、当前 token 前向和单次 SGD 更新三个 TODO；配套行为测试及复用旧 make_batch/loss 的训练演示。重点验收精确更新、连续两轮的清梯度、参数身份及后续求导、PAD 不变性和一致后继数据拟合。学习者表示大概理解不作为掌握证据；实现、有限差分、独立排错及生成仍待验收。"
    },
    {
      "id": "24970b35-4fbe-4619-a168-94ce387975b7",
      "at": "2026-09-07T11:49:51.373Z",
      "nodeId": "learning-mechanics.backward-update",
      "nodeTitle": "反向传播与参数更新",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "下一知识块为一次参数更新：先用已有查表、矩阵乘法和交叉熵串联前向计算，再解释 SGD 的学习率、backward 只求梯度、梯度累积/清理及 no_grad 更新边界。前向数据流、输出词表列对应、标签构造与更新将用同一训练闭环分别验收，不因已讲解自动判定掌握。"
    },
    {
      "id": "93e4cc36-0a60-44b9-9709-228553627621",
      "at": "2026-09-07T11:49:51.133Z",
      "nodeId": "learning-mechanics.next-token",
      "nodeTitle": "预测下一个文字单位",
      "action": "verify",
      "fromStatus": "current",
      "toStatus": "verify",
      "evidence": "能独立指出训练时读取下一输入位置可以直接复制正确答案，解释低训练损失与逐步生成时缺少未来答案、表现差并不矛盾；准确区分 target_valid 仅排除无效标签损失，不能限制输入数据的可见性。",
      "note": "因果可见性与损失筛选的区别已通过解释验证，不重复定义题。错位输入/标签及有效标记的独立构造、生成循环仍待在同一训练闭环中验收，暂不标记整个节点掌握。"
    },
    {
      "id": "b0f19270-8580-438e-be48-02154b63aa67",
      "at": "2026-09-07T08:21:13.300Z",
      "nodeId": "learning-mechanics.next-token",
      "nodeTitle": "预测下一个文字单位",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "下一步在 ex005 同一训练闭环中学习输入与目标错开一位，以及有效标签标记的对齐。输出分数与词表列对应关系仍待综合验证；生成循环和因果可见范围尚未验收，不提前标记掌握。"
    },
    {
      "id": "159b79e6-fbff-43af-a79e-d53785db1710",
      "at": "2026-09-07T08:21:13.122Z",
      "nodeId": "learning-mechanics.loss",
      "nodeTitle": "损失与交叉熵",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "独立完成 exercises/ex005_training_loop/loss.py：沿候选轴平移分数并计算稳定交叉熵，按 targets gather 正确答案分数，使用 target_valid 筛选后等权平均，整个 batch 无有效标签时抛出 ValueError；未使用高级封装、不修改输入且保留梯度。本题 12 项及全部 48 项练习测试通过且无跳过，覆盖极端错误预测、标签选择、PAD 不变性、全无效输入和参考梯度对齐。结合此前负对数目标、概率截断改变目标及短序列权重偏大的独立解释，完成本节点验收。",
      "note": "已完成批量稳定交叉熵的原理、实现和边界验证；不据此推定已掌握参数更新或完整训练闭环。"
    },
    {
      "id": "18564a37-2e68-427a-ad5e-7a8e1a1a669f",
      "at": "2026-09-07T06:51:10.461Z",
      "nodeId": "learning-mechanics.loss",
      "nodeTitle": "损失与交叉熵",
      "action": "current",
      "fromStatus": "current",
      "toStatus": "current",
      "evidence": "能指出先逐序列求平均再平均会让短序列中的单个有效 token 因分母更小而获得更大权重，判断这与所有有效 token 等权的目标不同。",
      "note": "进入 exercises/ex005_training_loop 第一部分：批量稳定交叉熵独立实现。负对数目标、稳定变形和有效标签归约已有解释证据；实现、PAD 不变性、全无效输入处理及梯度对齐待验收，完整训练闭环尚未完成。"
    },
    {
      "id": "be81a5dc-33ed-493b-b973-f6a1adff6148",
      "at": "2026-09-06T11:15:06.239Z",
      "nodeId": "learning-mechanics.loss",
      "nodeTitle": "损失与交叉熵",
      "action": "current",
      "fromStatus": "current",
      "toStatus": "current",
      "evidence": "能独立说明稳定交叉熵通过数学等价变形适应有限精度计算；对把正确答案概率下限截断到1e-8的方案，准确指出进入截断区间后损失被固定、该损失项梯度为零，而原目标应随正确答案概率继续降低而增大，因而截断改变了训练目标",
      "note": "负对数目标、数值等价变形与截断造成零梯度的边界已通过解释验证；下一步推广到批量按标签取分数和有效标签归约，再通过独立实现与测试验收完整损失节点"
    },
    {
      "id": "68da62c4-12b9-4ba7-92ec-dc41253f6d9c",
      "at": "2026-09-06T10:43:43.047Z",
      "nodeId": "learning-mechanics.loss",
      "nodeTitle": "损失与交叉熵",
      "action": "current",
      "fromStatus": "current",
      "toStatus": "current",
      "evidence": "能指出以最大预测概率构造损失会奖励确定性、可能使模型对错误答案过度自信，区分它与奖励正确答案概率的目标；关于固定错误目标时标签置乱不影响损失和梯度，已通过补充讲解澄清，尚不作为独立推导证据",
      "note": "继续同一损失知识块：从负对数正确答案概率推导逐组平移后的稳定交叉熵，先验证单位置数值；批量按标签取分数、PAD归约及学习者独立实现仍待完成，不提前标记掌握"
    },
    {
      "id": "1032033f-9e33-4573-a9a9-d53471d293e7",
      "at": "2026-09-05T16:12:36.485Z",
      "nodeId": "learning-mechanics.loss",
      "nodeTitle": "损失与交叉熵",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "在最小训练闭环中继续：先连接词表分数、逐组概率与训练标签，再解释单标签交叉熵的负对数目标；输出分数节点保留待综合验证，不提前上报掌握。稳定log-sum-exp实现及PAD归约随后讲解并验收"
    },
    {
      "id": "f2a098b8-5d8f-4c0a-b8ba-d03f7c419d73",
      "at": "2026-09-05T16:12:36.485Z",
      "nodeId": "learning-mechanics.logits-probability",
      "nodeTitle": "输出分数与概率",
      "action": "focus-shift",
      "fromStatus": "current",
      "toStatus": "verify",
      "evidence": null,
      "note": "学习焦点切换到 损失与交叉熵"
    },
    {
      "id": "0feb4d12-d264-48fa-8568-ac73b6be51f6",
      "at": "2026-09-04T13:10:03.669Z",
      "nodeId": "learning-mechanics.logits-probability",
      "nodeTitle": "输出分数与概率",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "Softmax和词表编号先修已掌握；继续连接每位置特征(B,T,C)、词表输出参数(C,N)、原始分数logits与候选概率(B,T,N)，通过输出列的词表对应关系验证理解"
    },
    {
      "id": "e6c68eee-fdd5-4715-9546-2266248e8fe5",
      "at": "2026-09-04T13:10:03.495Z",
      "nodeId": "foundation.softmax",
      "nodeTitle": "Softmax与数值稳定",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "独立写出两候选平移后的Softmax表达式，并指出分子分母可以约去公共因子e^(-c)，完成平移不变性的代数解释；结合此前稳定Softmax独立实现、本题10项及全仓36项Python测试通过，完成归一化轴、数值稳定性与计算图保持的验收",
      "note": "Softmax基础关卡已通过原理解释、独立实现与边界测试；包含逐组归一化、平移不变性及防止整组指数下溢。非有限输入与Attention mask留待相关节点处理"
    },
    {
      "id": "fbea914b-3b6a-413e-bc1a-98395642d17d",
      "at": "2026-09-04T13:07:15.192Z",
      "nodeId": "foundation.softmax",
      "nodeTitle": "Softmax与数值稳定",
      "action": "current",
      "fromStatus": "current",
      "toStatus": "current",
      "evidence": "独立实现 exercises/ex004_stable_softmax/stable_softmax.py：沿末轴保留单例维取最大值与求和，平移后指数归一化，不修改输入且保持自动求导路径；本题10项和全仓36项Python测试全部通过、无跳过，覆盖跨组极端分数、逐组平移不变、不同shape/dtype、允许微小概率下溢及与官方实现的梯度对齐",
      "note": "稳定Softmax的代码实现与数值边界验收已完成；同组分数平移不变性的代数证明已讲解，但尚未取得学习者独立解释的证据，当前保留此项待确认后再上报完整掌握"
    },
    {
      "id": "1f6d4952-3148-4cbe-acca-76f70a5835b9",
      "at": "2026-09-04T12:49:03.451Z",
      "nodeId": "foundation.softmax",
      "nodeTitle": "Softmax与数值稳定",
      "action": "current",
      "fromStatus": "current",
      "toStatus": "current",
      "evidence": null,
      "note": "已讲解沿候选轴独立归一化、同组分数平移不变性与逐组减最大值的浮点稳定性；进入 exercises/ex004_stable_softmax 的独立实现与边界验证，尚未上报掌握"
    },
    {
      "id": "60b2a286-b83c-4093-b281-37aa2f989466",
      "at": "2026-09-04T12:20:06.652Z",
      "nodeId": "foundation.softmax",
      "nodeTitle": "Softmax与数值稳定",
      "action": "current",
      "fromStatus": "verify",
      "toStatus": "current",
      "evidence": null,
      "note": "补齐Softmax归一化轴、减去常数的不变性及逐行减最大值的浮点稳定性，为预测分数与训练误差计算做准备"
    },
    {
      "id": "dbee6688-dbc9-4a82-b15a-3511b882ecb5",
      "at": "2026-09-04T12:20:06.449Z",
      "nodeId": "foundation.python-pytorch",
      "nodeTitle": "Python与PyTorch语法桥接",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "已独立完成基础Tensor创建、切片赋值、布尔标记、内容查表和位置广播相加，相关作业全仓26项测试通过；在自动求导反例中准确判断item取数后重建Tensor不改变L，但切断w经a的路径，w.grad仅剩grad_b*a，实测为216",
      "note": "基础张量代码与自动求导边界理解已验证，不代表熟悉全部Python/PyTorch API；后续新API仍先解释，梯度累积及其他机制按实际需要学习"
    },
    {
      "id": "83b501a4-7b85-40f8-8557-e1f6c3aaad8f",
      "at": "2026-09-04T12:05:02.779Z",
      "nodeId": "foundation.python-pytorch",
      "nodeTitle": "Python与PyTorch语法桥接",
      "action": "current",
      "fromStatus": "verify",
      "toStatus": "current",
      "evidence": null,
      "note": "基于已手推的共享参数计算，学习requires_grad、backward、grad及数值提取对求导路径的影响；Tensor创建、索引、广播已有练习证据"
    },
    {
      "id": "929d1df6-5f69-42db-846e-b3148f6d429f",
      "at": "2026-09-04T12:05:02.607Z",
      "nodeId": "foundation.chain-rule",
      "nodeTitle": "链式法则与反向传播",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "对a=w*x、b=a*w、L=b*b的共享参数计算，独立给出grad_b=2*b、grad_a=grad_b*w、grad_w=grad_b*a+grad_a*x，明确沿局部导数传递并累加w的两条路径贡献",
      "note": "已通过多步标量链式求导与共享参数多路径累加验证；后续进入PyTorch自动求导使用，不将API熟练度与数学理解混为一谈"
    },
    {
      "id": "e43372b4-f5d4-44a4-8403-53dfdce7d3c0",
      "at": "2026-09-04T11:48:36.454Z",
      "nodeId": "foundation.chain-rule",
      "nodeTitle": "链式法则与反向传播",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "已有基础导数能力；当前学习复合计算的局部导数传递与共享参数多路径贡献累加，并用实现或反例验证而非重复定义"
    },
    {
      "id": "70a4c2f7-1b5e-4692-ab42-b0da54fedce6",
      "at": "2026-09-04T11:48:36.269Z",
      "nodeId": "learning-mechanics.parameter-activation",
      "nodeTitle": "参数、输入与中间结果",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "能将E、P、W识别为跨请求长期保留的模型参数，将ids、valid、X、Y归为请求相关数据，并指出不同输入导致输出变化并不能说明模型发生学习或参数更新",
      "note": "模型参数与请求数据生命周期已验证；这类基础程序概念后续不再安排定义复述式检查"
    },
    {
      "id": "61c2b3ab-dd05-4455-a63e-0c158cb9a412",
      "at": "2026-09-04T11:34:02.582Z",
      "nodeId": "learning-mechanics.parameter-activation",
      "nodeTitle": "参数、输入与中间结果",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "结合已完成的Embedding、Padding与位置表示，学习持久参数、外部输入及每次运行中间结果的区别；完整文本输入管线尚待综合验证，不提前标记掌握"
    },
    {
      "id": "ab5635db-d31d-4924-8b55-fc0822656333",
      "at": "2026-09-04T11:34:02.422Z",
      "nodeId": "text-input.position",
      "nodeTitle": "位置信息",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "能解释交换token后位置向量由当前位置决定而非跟随token移动，并独立实现按token ID查内容表、按前T行取位置表及广播相加；已自行修正shape属性调用错误，位置表示8项测试和全仓26项测试全部通过且无跳过",
      "note": "基础绝对位置表示已通过解释、实现及长度边界和不修改输入等测试；正弦位置编码与其他位置机制留待对应阶段"
    },
    {
      "id": "59c6ac4f-50ac-4841-8375-f8f9b3a7c0d5",
      "at": "2026-09-04T09:14:59.705Z",
      "nodeId": "text-input.position",
      "nodeTitle": "位置信息",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "Embedding与广播均已掌握；当前学习内容向量与位置向量的区别，并实现基于位置表的基础绝对位置表示"
    },
    {
      "id": "fb7638cb-9cf3-4e6e-84f2-16c2ebf9c0b0",
      "at": "2026-09-04T09:13:12.447Z",
      "nodeId": "foundation.python-pytorch",
      "nodeTitle": "Python与PyTorch语法桥接",
      "action": "verify",
      "fromStatus": "pending",
      "toStatus": "verify",
      "evidence": null,
      "note": "已独立使用Tensor创建、切片赋值、整数与布尔dtype等完成Padding并通过测试；对Python/PyTorch仍不熟悉，新API需先解释，函数名熟练度不能替代原理判断"
    },
    {
      "id": "56be772e-3480-48a9-8d51-c96f992b69c2",
      "at": "2026-09-04T09:13:12.273Z",
      "nodeId": "foundation.broadcasting",
      "nodeTitle": "广播规则",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "在X为(2,2,2)、valid为(2,2)的静默错误案例中，准确判断运算不报错且实际读取valid[t,c]；澄清修正意图是用unsqueeze(-1)增加末尾单例轴，使有效位置标记沿特征轴复用，先前仅混淆API名称",
      "note": "已验证右对齐的合法性与实际索引语义，能识别shape合法但数据语义错误的广播；API名称熟练度单独在Python/PyTorch语法节点跟踪"
    },
    {
      "id": "9a2af2ad-8795-4810-b960-1c46e8b6179e",
      "at": "2026-09-04T08:44:55.647Z",
      "nodeId": "foundation.broadcasting",
      "nodeTitle": "广播规则",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "张量shape和批量矩阵乘法先修已掌握；在位置向量相加前学习尾轴对齐、长度1扩展及广播的索引语义"
    },
    {
      "id": "427e8003-555c-4217-a0d0-b62a9f088c99",
      "at": "2026-09-04T08:41:41.577Z",
      "nodeId": "text-input.batch-padding",
      "nodeTitle": "批处理与长度补齐",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "独立完成 exercises/ex002_batch_padding/batch_padding.py 的 make_batch：按最长序列右侧补齐，正确生成 long 编号张量和 bool 有效位置标记，支持非零PAD且保留UNK与真实ID 0，不修改输入；本题8项测试及全仓18项测试在项目PyTorch环境中全部通过且无跳过",
      "note": "已通过实现与性质测试验证批处理补齐、输出shape/dtype及真实内容与PAD位置的区分；循环生成valid逻辑正确，向量化仅为可选改进"
    },
    {
      "id": "c8a6eebc-6161-4863-af0a-c103f8443b4e",
      "at": "2026-09-04T06:59:57.323Z",
      "nodeId": "text-input.batch-padding",
      "nodeTitle": "批处理与长度补齐",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "词表编号、张量shape与Embedding已掌握；当前学习不同长度序列的批量补齐与有效位置标记"
    },
    {
      "id": "7c4dd052-130f-47f2-b2d1-9f7c538f5ed4",
      "at": "2026-09-04T06:59:57.153Z",
      "nodeId": "text-input.embedding",
      "nodeTitle": "编号变向量",
      "action": "master",
      "fromStatus": "current",
      "toStatus": "mastered",
      "evidence": "在token ID 0与2重新编号的反例中，能说明编号虽可任意选择但查表对应关系必须一致，准确要求交换E的第0与第2行以保持输出数值不变，并指出输出shape不变",
      "note": "已通过编号重映射的边界验证，理解Embedding按ID查行及重映射不改变表与输出shape"
    },
    {
      "id": "b7ed2a38-8816-4a20-b01f-528cd5432397",
      "at": "2026-09-04T03:43:59.732Z",
      "nodeId": "text-input.embedding",
      "nodeTitle": "编号变向量",
      "action": "current",
      "fromStatus": "pending",
      "toStatus": "current",
      "evidence": null,
      "note": "词表编号与张量shape先修均已通过；当前学习按token ID查表得到向量，区分固定编号、可学习参数表与输出张量"
    },
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
