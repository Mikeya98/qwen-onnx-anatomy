# -*- coding: utf-8 -*-
"""把全 fp16 的 ONNX 模型转成 fp32。

onnxruntime 动态量化按 fp32 模型设计，直接量化 fp16 模型会撞类型 bug
（scale 被存成 fp16、Gather 输出 fp16 对不上 DequantizeLinear 期望的 float）。

做法：所有 fp16 initializer / value_info 转 fp32，所有 Cast 的 to 统一改成
fp32（变成 no-op，交给 onnxruntime 优化器消除），再重算类型。
用法:
    python to_fp32.py <输入.onnx> <输出.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def main(src, dst):
    m = onnx.load(src, load_external_data=True)  # 需要读权重才能转 dtype
    g = m.graph

    # 1. initializer: fp16 -> fp32
    n_init = 0
    for init in g.initializer:
        if init.data_type == 10:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new)
            n_init += 1

    # 2. value_info / graph input / graph output: elem_type 10 -> 1
    def fix_tensor_type(t):
        if t.HasField("tensor_type") and t.tensor_type.elem_type == 10:
            t.tensor_type.elem_type = 1

    n_vi = 0
    for v in list(g.value_info) + list(g.input) + list(g.output):
        if v.type.tensor_type.elem_type == 10:
            fix_tensor_type(v.type)
            n_vi += 1

    # 3. Cast 节点: to=10 -> to=1
    n_cast = 0
    for node in g.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == 10:
                    attr.i = 1
                    n_cast += 1

    # 4. Constant / ConstantOfShape 节点属性里的 fp16 常量 -> fp32
    n_const = 0
    for node in g.node:
        if node.op_type in ("Constant", "ConstantOfShape"):
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == 10:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))
                    n_const += 1

    onnx.save(m, dst, save_as_external_data=True,
              all_tensors_to_one_file=True, location="model_fp32_data.bin",
              size_threshold=1024)
    print(f"initializer 转 fp32: {n_init} 个")
    print(f"value/io 转 fp32:    {n_vi} 个")
    print(f"Cast 目标改 fp32:    {n_cast} 个")
    print(f"Constant 转 fp32:    {n_const} 个")
    print(f"已保存: {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
