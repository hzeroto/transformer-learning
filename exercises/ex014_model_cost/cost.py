"""学习者填写成本账本；当前只有一个 TODO，题面见同目录 README.md。"""


def estimate_matmul_flops(
    *, B: int, n: int, S: int, C: int, Hq: int, Hkv: int,
    G: int, layers: int, N: int,
) -> dict[str, int]:
    """估算 ex013 模型一次前向的主要矩阵乘法操作数。

    参数全部是 Python 正整数，由调用者保证合法，不考参数校验：
      B：等长、无 PAD 序列的 batch 大小。
      n：本次新输入的位置数；
      S：追加后的总长度，1 <= n <= S。
         进入时缓存长度 t=S-n；prefill 为 n=S，单步 decode 为 n=1。
      C：主干特征宽；
      Hq/Hkv：query/KV 头数。
         C 能被 Hq 整除，Hq 能被 Hkv 整除，每头宽 D=C//Hq 为偶数。
         紧凑 KV 投影宽 K=Hkv*D；这里 K 是整数，不是张量。
      G：SwiGLU 中间宽；
      layers：块数；
      N：词表大小。

    返回下列八个键，各值均为 Python int。前六项都已经包含全部 layers：
      q_proj：Q 投影。
      kv_proj：仅新位置的 K、V 两次投影之和。
      score：所有 query 头的 QK 打分。
      value_read：注意力权重乘 V。
      out_proj：合头后的 Wo 投影。
      ffn：SwiGLU 的 gate、up、down 三次矩阵乘法之和。
      vocab：最终词表投影，仅一次，计算本次全部 n 个位置。
      total：上述七项之和；不要再次乘 layers。

    口径：一次乘加约计 2 FLOPs，沿用先稠密矩阵乘法再加因果 mask 的
    ex013 实现。单层矩阵形状和各张量来源见 README 的表格。
    不计 RMSNorm、RoPE、Softmax、SiLU、逐元素乘法、残差、mask、
    查表及复制，不计反向、优化器更新，不估算耗时或字节数。

    只做整数算术；不需要创建或运行模型，不读写参数、缓存或文件。
    不从 tests 或 examples 的教师参照导入计算，不写死示例结果。
    """
    D = C // Hq
    K = Hkv * D
    # Q = x @ Wq (B, n, C) @ (C, C)
    flops_proj = 2 * B * n * C * C * layers
    # K_new = x @ Wk (B, n, C) @ (C, K)
    # V_new = x @ Wv (B, n, C) @ (C, K)
    flops_kv_proj = 4 * B * n * C * K * layers
    # score = Q (B, Hq, n, D) @ K^T (B, Hkv, D, S) Hq -> Hkv Hkv按00..11..Hkv-1分组
    # score (B, Hq, n, S)
    flops_score = 2 * B  * Hq * n * S * D * layers
    # grade(B, Hq, n, D) = score(B, Hq, n, S) @ V(B, Hkv, S, D) 
    flops_read = 2 * B * Hq * n * S * D * layers
    # out_proj(B, Hq, n, K) = grade_merge(B, n, C) @ Wo (C, C)
    flops_out_proj = 2 * B * n * C * C * layers
    # ffn_1(B, n, G) = X(B, n, C) @ W_gate(C, G)
    flops_ffn = 2 * B * n * G * C * layers
    # up = X(B, n, C) @ W_up(C, G)
    flops_ffn += 2 * B * n * G * C * layers
    # # ffn_2(B, n, G) = ffn_1(B, n, G) * up(B, n, G)
    # flops_ffn += B * n * G * layers 暂时不计
    # ffn_3(B, n, C) = ffn_2(B, n, G) @ W_down(G, C)
    flops_ffn += 2 * B * n * G * C * layers
    # vocab = X(B, n, C) @ W_vocab(C, N)
    flops_vocab = 2 * B * n * C * N
    # total = q_proj + kv_proj + score + value_read + out_proj + ffn + vocab
    flops_total = flops_proj + flops_kv_proj + flops_score + flops_read + flops_out_proj + flops_ffn + flops_vocab

    return {
        "q_proj": flops_proj,
        "kv_proj": flops_kv_proj,
        "score": flops_score,
        "value_read": flops_read,
        "out_proj": flops_out_proj,
        "ffn": flops_ffn,
        "vocab": flops_vocab,
        "total": flops_total,
    }





    
