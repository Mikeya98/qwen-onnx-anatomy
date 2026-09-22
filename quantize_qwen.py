# -*- coding: utf-8 -*-
"""对 fp32 的 Qwen ONNX 模型做 int8 动态量化。

注意：不能直接量化 fp16 模型。onnxruntime 的量化工具按 fp32 设计，
直接量化 fp16 模型会撞两个类型 bug（scale 被存成 fp16、Gather 输出 fp16
对不上 DequantizeLinear 的 float）。所以先跑 to_fp32.py 把模型转成 fp32，
再跑这个脚本。

用法:
    python to_fp32.py <fp16模型> <fp32模型>
    python quantize_qwen.py <fp32模型> <输出int8.onnx>
"""
import os
import sys

import onnxruntime.quantization as q
from onnxruntime.quantization import QuantType


def main(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    q.quantize_dynamic(
        src,
        dst,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    # 输出文件大小对比
    src_size = os.path.getsize(src)
    dst_dir = os.path.dirname(dst)
    dst_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(dst_dir)
        for f in files
    )
    print(f"输入: {src}  {src_size / 1e9:.2f} GB")
    print(f"输出: {dst}  (目录 {dst_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
