# -*- coding: utf-8 -*-
"""解析 Qwen2.5-1.5B-Instruct ONNX 图结构：算子统计 / 参数账目 / KV cache 签名

只依赖 onnx，且不加载权重（load_external_data=False），几秒就能跑完。
用法:
    python analyze_qwen_onnx.py [模型目录]
默认模型目录 = 脚本所在目录下的 qwen2.5-1.5b-instruct-onnx-fp16
"""
import os
import sys
import collections
import functools
import operator

import onnx


def prod(dims):
    return functools.reduce(operator.mul, dims, 1) if dims else 1


TYPES = {1: "float32", 10: "float16", 7: "int64", 6: "int32", 9: "bool"}


def main(model_dir):
    onnx_path = os.path.join(model_dir, "onnx", "model_fp16.onnx")
    data_path = os.path.join(model_dir, "onnx", "model_fp16.onnx_data")

    # ---- 一、两个文件的大小 ----
    print("=" * 62)
    print("一、模型在硬盘上的两个文件")
    print("=" * 62)
    if os.path.exists(onnx_path):
        print(f"  图结构 model_fp16.onnx      {os.path.getsize(onnx_path) / 1e6:8.2f} MB")
    if os.path.exists(data_path):
        print(f"  权重   model_fp16.onnx_data {os.path.getsize(data_path) / 1e9:8.2f} GB")

    # ---- 二、图结构 + 算子统计 ----
    m = onnx.load(onnx_path, load_external_data=False)
    g = m.graph
    print()
    print("=" * 62)
    print("二、图结构")
    print("=" * 62)
    print(f"  opset 版本 : {[o.version for o in m.opset_import]}")
    print(f"  节点数     : {len(g.node)}")
    print(f"  输入/输出  : {len(g.input)} 进 / {len(g.output)} 出")

    ops = collections.Counter(n.op_type for n in g.node)
    print("\n  算子统计 (top 15):")
    for op, cnt in ops.most_common(15):
        print(f"    {op:20s} {cnt}")

    # ---- 三、权重统计 ----
    print()
    print("=" * 62)
    print("三、权重(initializer)统计")
    print("=" * 62)
    fp16 = [i for i in g.initializer if i.data_type == 10]
    others = [i for i in g.initializer if i.data_type != 10]
    total_fp16 = sum(prod(i.dims) for i in fp16)
    print(f"  初始器总数      : {len(g.initializer)}")
    print(f"    float16 权重  : {len(fp16)} 个，共 {total_fp16:,} 参数")
    print(f"    其它(常量等)  : {len(others)} 个")

    print("\n  float16 权重按形状归类:")
    shapes = collections.Counter(tuple(i.dims) for i in fp16)
    for shape, cnt in sorted(shapes.items(), key=lambda x: -prod(x[0])):
        print(f"    shape={list(shape)!s:22s} x{cnt:3d}  = {prod(shape) * cnt:>14,}")

    groups = collections.Counter()
    for i in fp16:
        nm = i.name
        if nm.endswith("embed_tokens.weight"):
            groups["embed_tokens"] += prod(i.dims)
        elif "layernorm.weight" in nm or nm.endswith("model.norm.weight"):
            groups["RMSNorm"] += prod(i.dims)
        elif "rotary" in nm:
            groups["RoPE cos/sin 表"] += prod(i.dims)
        elif nm.endswith(".bias"):
            groups["导出带的 bias"] += prod(i.dims)
        else:
            groups["投影权重(注意力+FFN)"] += prod(i.dims)

    print("\n  按成分归类:")
    for k, v in groups.most_common():
        print(f"    {k:22s} {v:>15,}")

    # ---- 四、输入输出签名 ----
    print()
    print("=" * 62)
    print("四、输入输出签名(KV cache)")
    print("=" * 62)

    def sig(t):
        dims = ["?" if d.dim_value == 0 else (d.dim_param or d.dim_value)
                for d in t.type.tensor_type.shape.dim]
        return TYPES.get(t.type.tensor_type.elem_type, t.type.tensor_type.elem_type), dims

    past_in = [i for i in g.input if i.name.startswith("past_key_values")]
    present = [o for o in g.output if o.name.startswith("present")]
    print(f"  KV cache 输入 : {len(past_in)} 个 (28 层 × key/value)")
    print(f"  KV cache 输出 : {len(present)} 个 present (28 层 × key/value)")
    for name in ["input_ids", "attention_mask", "position_ids"]:
        t = next(i for i in g.input if i.name == name)
        dt, dims = sig(t)
        print(f"    {name:16s} {dt:8s} {dims}")
    t = next(i for i in g.input if i.name.startswith("past_key_values"))
    dt, dims = sig(t)
    print(f"    past_key_values.N.key/value  {dt:8s} {dims}")
    t = next(o for o in g.output if o.name == "logits")
    dt, dims = sig(t)
    print(f"    logits   {dt:8s} {dims}")

    # ---- 五、手算参数对账 ----
    print()
    print("=" * 62)
    print("五、手算参数量对账")
    print("=" * 62)
    vocab, hidden = 151936, 1536
    n_layers, inter = 28, 8960
    n_heads, n_kv = 12, 2
    head_dim = hidden // n_heads
    emb = vocab * hidden
    attn = hidden * hidden * 2 + hidden * n_kv * head_dim * 2          # q,o + k,v
    mlp = hidden * inter * 3
    per_layer = attn + mlp                                             # 不含 norm，norm 单独算
    rope = 2 * 32768 * head_dim
    bias = (hidden + n_kv * head_dim * 2) * n_layers
    norm = hidden * (2 * n_layers + 1)
    manual = emb + per_layer * n_layers + norm + rope + bias
    print(f"  词嵌入 embed_tokens : {emb:>15,}")
    print(f"  注意力投影(28层)    : {attn * n_layers:>15,}")
    print(f"  FFN(28层)           : {mlp * n_layers:>15,}")
    print(f"  RMSNorm(57个)       : {norm:>15,}")
    print(f"  RoPE cos/sin 表     : {rope:>15,}")
    print(f"  导出带的 bias       : {bias:>15,}")
    print("  " + "-" * 32)
    print(f"  手算合计            : {manual:>15,}")
    print(f"  文件里实际 float16  : {total_fp16:>15,}")
    print(f"  差值                : {total_fp16 - manual:>15,}")
    print()
    print(f"  权重文件应 ≈ 参数 × 2 字节(fp16): {total_fp16 * 2 / 1e9:.2f} GB")


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "qwen2.5-1.5b-instruct-onnx-fp16",
    )
    main(model_dir)
