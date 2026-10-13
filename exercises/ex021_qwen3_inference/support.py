"""教师提供的文件格式适配；不实现模型数学，不在导入时依赖 Hugging Face。"""
from dataclasses import dataclass, fields
import json
from pathlib import Path

import torch


MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_MODEL_DIR = Path("checkpoints/Qwen3-0.6B")


@dataclass(frozen=True)
class QwenConfig:
    """只读配置；head_dim 独立于 hidden_size / num_attention_heads。"""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    tie_word_embeddings: bool = True

    def __post_init__(self):
        for name in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "max_position_embeddings",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Q 头数必须是 KV 头数的整数倍")
        if self.head_dim % 2 or self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("head_dim 必须为偶数；epsilon 和 theta 必须为正数")


@dataclass(frozen=True)
class LayerKV:
    """一个请求的一层缓存：k/v 均为 (B,Hkv,S,D)，k 已经做过 RoPE。"""

    k: torch.Tensor
    v: torch.Tensor


def read_config(path: str | Path) -> QwenConfig:
    """读取目录内 config.json 或该文件路径；拒绝本练习没有实现的变体。"""
    path = Path(path)
    data = json.loads((path / "config.json" if path.is_dir() else path).read_text())
    if data.get("model_type") != "qwen3":
        raise ValueError("本练习只支持 dense Qwen3，不支持 Qwen2/MoE")
    if data.get("hidden_act", "silu") != "silu":
        raise ValueError("本练习只实现 SiLU 门控 FFN")
    if data.get("attention_bias", False) or data.get("mlp_bias", False):
        raise ValueError("本练习不支持投影 bias")
    if data.get("use_sliding_window", False) or data.get("sliding_window") is not None:
        raise ValueError("本练习使用完整因果注意力，不支持滑窗")
    if any(kind != "full_attention" for kind in data.get("layer_types", [])):
        raise ValueError("本练习只支持 full_attention 层")
    scaling = data.get("rope_scaling")
    if scaling not in (None, {}) and scaling != {"rope_type": "default"}:
        raise ValueError("本练习不支持 RoPE scaling")
    names = {field.name for field in fields(QwenConfig)}
    # 不推导 head_dim：官方 0.6B 的 Q 总宽度不等于 hidden_size。
    if "head_dim" not in data:
        raise ValueError("config 必须显式提供 head_dim")
    return QwenConfig(**{key: value for key, value in data.items() if key in names})


def expected_shapes(config: QwenConfig) -> dict[str, tuple[int, ...]]:
    """官方权重名称与磁盘 shape；Linear 权重顺序为 (out_features,in_features)。"""
    c, f, d = config.hidden_size, config.intermediate_size, config.head_dim
    q, kv = config.num_attention_heads * d, config.num_key_value_heads * d
    shapes = {
        "model.embed_tokens.weight": (config.vocab_size, c),
        "model.norm.weight": (c,),
        "lm_head.weight": (config.vocab_size, c),
    }
    layer = {
        "input_layernorm.weight": (c,), "post_attention_layernorm.weight": (c,),
        "self_attn.q_proj.weight": (q, c), "self_attn.k_proj.weight": (kv, c),
        "self_attn.v_proj.weight": (kv, c), "self_attn.o_proj.weight": (c, q),
        "self_attn.q_norm.weight": (d,), "self_attn.k_norm.weight": (d,),
        "mlp.gate_proj.weight": (f, c), "mlp.up_proj.weight": (f, c),
        "mlp.down_proj.weight": (c, f),
    }
    for index in range(config.num_hidden_layers):
        shapes.update({f"model.layers.{index}.{name}": shape for name, shape in layer.items()})
    return shapes


def _check_keys(keys, config):
    expected = set(expected_shapes(config))
    missing, extra = expected - set(keys), set(keys) - expected
    if config.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if missing or extra:
        raise ValueError(f"checkpoint key 不匹配：missing={sorted(missing)}, extra={sorted(extra)}")


def _check_tensor(name, tensor, shape):
    if tuple(tensor.shape) != shape or not tensor.is_floating_point():
        raise ValueError(f"{name} 应为浮点张量 {shape}，实际 {tuple(tensor.shape)} / {tensor.dtype}")


def validate_weights(weights: dict[str, torch.Tensor], config: QwenConfig) -> dict[str, torch.Tensor]:
    """严格检查名称/shape，返回 CPU FP32 字典；不转置、不 reshape，不修改入参。

    tied LM head 可省略；若文件双存两个键，必须数值相等，返回值使用同一张量引用。
    """
    _check_keys(weights, config)
    shapes = expected_shapes(config)
    for name, value in weights.items():
        _check_tensor(name, value, shapes[name])
    if config.tie_word_embeddings and "lm_head.weight" in weights:
        if not torch.equal(weights["model.embed_tokens.weight"], weights["lm_head.weight"]):
            raise ValueError("tie_word_embeddings=True，但 embedding 与 lm_head 数值不同")
    result = {
        name: value.detach().to(device="cpu", dtype=torch.float32)
        for name, value in weights.items()
        if not (config.tie_word_embeddings and name == "lm_head.weight")
    }
    if config.tie_word_embeddings:
        result["lm_head.weight"] = result["model.embed_tokens.weight"]
    return result


def load_weights(model_dir: str | Path, config: QwenConfig | None = None) -> dict[str, torch.Tensor]:
    """逐 tensor 加载固定模型的单个 model.safetensors；不导入 pickle/远端代码。

    本练习不支持分片、量化权重或任意架构。可选依赖仅在真实加载时导入。
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    config = config or read_config(model_dir)
    path = model_dir / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}；先显式运行下载入口，本函数不会联网")
    shapes, result = expected_shapes(config), {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        _check_keys(keys, config)
        for name in keys:
            if tuple(handle.get_slice(name).get_shape()) != shapes[name]:
                raise ValueError(f"{name} shape 与 config 不匹配")
        if config.tie_word_embeddings and "lm_head.weight" in keys:
            # 先在文件的 BF16 dtype 下比较，避免分配两份巨大的 FP32 词表。
            if not torch.equal(handle.get_tensor("model.embed_tokens.weight"),
                               handle.get_tensor("lm_head.weight")):
                raise ValueError("文件中的 tied embedding 与 lm_head 数值不同")
        for name in sorted(keys):
            if config.tie_word_embeddings and name == "lm_head.weight":
                continue
            tensor = handle.get_tensor(name)
            _check_tensor(name, tensor, shapes[name])
            result[name] = tensor.to(dtype=torch.float32)
    if config.tie_word_embeddings:
        result["lm_head.weight"] = result["model.embed_tokens.weight"]
    return result


def encode_prompt(model_dir: str | Path, prompt: str):
    """官方 tokenizer 只处理文本协议；模型前向仍由学习者实现。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
    )
    return tokenizer, ids


def eos_ids(model_dir: str | Path, tokenizer) -> tuple[int, ...]:
    """优先采用 checkpoint 的生成结束符集合，而不是只读 tokenizer 的单个 EOS。"""
    path = Path(model_dir) / "generation_config.json"
    value = json.loads(path.read_text()).get("eos_token_id") if path.is_file() else None
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        raise ValueError("本地模型没有配置 EOS")
    return (value,) if isinstance(value, int) else tuple(value)
