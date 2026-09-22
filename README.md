# qwen-onnx-anatomy

把 Qwen2.5-1.5B-Instruct 的 ONNX 模型拆开、跑起来、量化的全过程。起因是想搞清楚"一个 15 亿参数的大模型，在硬盘上到底长什么样、纯 CPU 上到底能不能跑"，而不是停在 `model.generate()` 一行的黑盒。

配套文章：

- [拆开 Qwen2.5-1.5B 的 ONNX 文件：一个 15 亿参数的模型长什么样](https://zhuanlan.zhihu.com/p/2085479762013729545)
- 把 Qwen2.5-1.5B 在纯 CPU 上跑起来（本篇，含 fp16/fp32/int8 实测耗时 + 量化踩坑）

## 四个脚本

| 脚本 | 干什么 | 依赖 |
|---|---|---|
| `analyze_qwen_onnx.py` | 拆图：算子统计 / 参数量对账 / KV cache 签名，不加载权重 | `onnx` |
| `run_qwen_cpu.py` | 纯 CPU 跑通生成循环 + 测单 token 耗时 | `onnxruntime` + `tokenizers` |
| `to_fp32.py` | 把全 fp16 模型转成 fp32 | `onnx` |
| `quantize_qwen.py` | 对 fp32 模型做 int8 动态量化 | `onnxruntime` |

## 怎么跑

先下模型（走 hf-mirror 镜像，直连下不动）：

```python
from huggingface_hub import snapshot_download
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

snapshot_download(
    "onnx-community/Qwen2.5-1.5B-Instruct",
    local_dir="qwen2.5-1.5b-instruct-onnx-fp16",
)
```

**拆图**（只依赖 onnx，不加载权重，几秒出结果）：

```bash
pip install onnx
python analyze_qwen_onnx.py qwen2.5-1.5b-instruct-onnx-fp16
```

**纯 CPU 跑通 + 测耗时**（线程数可加第三个参数）：

```bash
pip install onnxruntime tokenizers
python run_qwen_cpu.py qwen2.5-1.5b-instruct-onnx-fp16 30 16
```

**量化**（一定要先转 fp32，直接量化 fp16 会报类型错误）：

```bash
python to_fp32.py qwen2.5-1.5b-instruct-onnx-fp16/onnx/model_fp16.onnx model_fp32.onnx
python quantize_qwen.py model_fp32.onnx int8/model_int8.onnx
```

## 实测结果（16 核纯 CPU）

| 版本 | 权重文件 | prefill（24 token） | decode 单 token |
|---|---|---|---|
| fp16（原始） | 3.10GB | 2.87s | 1561ms |
| fp32（转的） | 6.20GB | 0.43s | 347ms |
| int8（量化的） | 1.56GB | 0.52s | 281ms |

几个有意思的点：

- **fp16 在纯 CPU 上反而最慢**。CPU 没有 fp16 算数单元，图里 115 个 Cast 全是来回转换的开销。fp16 是给 GPU 的。
- **decode 不随线程数变**（内存带宽受限），prefill 随线程数变（计算密集）。加核心救不了 decode。
- **int8 砍半体积、提 5 倍速度，但小模型动态量化会崩精度**。让 int8 模型回答"1+1 等于几"直接答非所问。要保精度得静态量化 + 校准集。

详细拆解在文章里。
