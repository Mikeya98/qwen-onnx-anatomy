# qwen-onnx-anatomy

把 Qwen2.5-1.5B-Instruct 的 ONNX 模型拆开看了一遍：图结构、参数量账目、KV cache 签名。起因是想搞清楚"一个 15 亿参数的大模型，在硬盘上到底长什么样"，而不是停在 `model.generate()` 一行的黑盒。

配套文章：拆开 Qwen2.5-1.5B 的 ONNX 文件（链接发布后补）。

## 这脚本干什么

大模型的 ONNX 文件本质是一个 protobuf 序列化的计算图，图里既装算子节点，也装权重。这个脚本把图读出来，回答几个问题：

- `model_fp16.onnx` 和 `model_fp16.onnx_data` 两个文件分别装了什么，为什么一个 1.1MB 一个 3.1GB
- 3207 个节点都在算什么，算子有哪些
- "15 亿参数"这 15 亿摊在哪：词嵌入、注意力、FFN、归一化各占多少
- 为什么输入 59 个、输出 57 个，KV cache 长什么样

## 怎么跑

只依赖 `onnx`，而且不加载权重（`load_external_data=False`），几秒出结果：

```bash
pip install onnx
python analyze_qwen_onnx.py <模型目录>
```

`<模型目录>` 是下载下来的那个文件夹，里面要有 `onnx/model_fp16.onnx`。模型从 onnx-community 拿（只跑这个脚本的话，下 onnx 两个文件就够；要真正跑推理还得带上 config.json、tokenizer.json 那些）：

```python
from huggingface_hub import snapshot_download
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

snapshot_download(
    "onnx-community/Qwen2.5-1.5B-Instruct",
    allow_patterns=["onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data"],
    local_dir="qwen2.5-1.5b-instruct-onnx-fp16",
)
```

不传参数的话，脚本会在自己所在目录下找 `qwen2.5-1.5b-instruct-onnx-fp16` 这个文件夹。

## 跑出来是什么

关键几行：

```
float16 权重 : 569 个，共 1,552,103,141 参数
图结构       : 1.10 MB
权重         : 3.10 GB

手算合计            : 1,552,102,912
文件里实际 float16  : 1,552,103,141
差值                : 229
```

手算和文件里实际存的只差 229 个参数，是几个零散的常量。能对上，说明"15 亿参数"不是厂家拍脑袋报的数，自己一步步能算出来。

## 几个有意思的点

- FFN 占了大头：11.6 亿 / 15.5 亿（约 75%），参数主要堆在 feed-forward，不在注意力
- GQA 省参数：12 个 query 头只有 2 个 KV 头，K/V 投影从 1536×1536 缩到 1536×256
- 权重是 fp16，但 KV cache 和 logits 是 fp32，别被文件名里的 "fp16" 骗了
- RoPE 的 cos/sin 表被展开存了 840 万参数，比所有归一化权重加起来还大两个数量级

详细拆解在文章里。
