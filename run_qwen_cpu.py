# -*- coding: utf-8 -*-
"""纯 CPU 跑 Qwen2.5-1.5B-Instruct (ONNX fp16)：完整生成循环 + 单 token 耗时统计

只用 onnxruntime + tokenizers，不依赖 transformers。
用法:
    python run_qwen_cpu.py [模型目录] [max_new_tokens] [线程数]
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

N_LAYERS = 28
N_KV = 2
HEAD_DIM = 128


def main(model_dir, max_new_tokens, threads):
    onnx_path = os.path.join(model_dir, "onnx", "model_fp16.onnx")
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    t0 = time.time()
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    print(f"session 加载: {time.time() - t0:.1f}s (线程数={threads or '默认'})", flush=True)

    output_names = [o.name for o in sess.get_outputs()]
    logits_idx = output_names.index("logits")

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n用一句话介绍你自己。<|im_end|>\n"
              "<|im_start|>assistant\n")
    ids = tok.encode(prompt).ids
    print(f"prompt: {len(ids)} tokens", flush=True)

    past_len = 0
    generated = []

    # 第一轮 prefill：past 全零
    feeds = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        "position_ids": np.arange(len(ids), dtype=np.int64)[None, :],
    }
    for i in range(N_LAYERS):
        for kv in ("key", "value"):
            feeds[f"past_key_values.{i}.{kv}"] = np.zeros((1, N_KV, 0, HEAD_DIM), dtype=np.float32)

    t = time.time()
    outs = sess.run(output_names, feeds)
    t_prefill = time.time() - t
    logits = outs[logits_idx]
    next_id = int(np.argmax(logits[0, -1, :]))
    print(f"prefill({len(ids)} tokens): {t_prefill:.3f}s  -> id {next_id}", flush=True)

    # 后续 decode：把上一轮 present 回填为 past
    past_len = len(ids)
    decode_ms = []
    for step in range(max_new_tokens):
        # 上一轮的 present 就是这一轮的 past
        feeds = {
            "input_ids": np.array([[next_id]], dtype=np.int64),
            "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
            "position_ids": np.array([[past_len]], dtype=np.int64),
        }
        for i in range(N_LAYERS):
            for kv in ("key", "value"):
                feeds[f"past_key_values.{i}.{kv}"] = outs[output_names.index(f"present.{i}.{kv}")]

        t = time.time()
        outs = sess.run(output_names, feeds)
        dt = time.time() - t
        decode_ms.append(dt * 1000)

        logits = outs[logits_idx]
        next_id = int(np.argmax(logits[0, -1, :]))
        print(f"decode #{step+1}: {dt*1000:.0f} ms  -> id {next_id}", flush=True)

        if next_id in (151645, 151643):                     # <|im_end|> / <|endoftext|>
            break
        generated.append(next_id)
        past_len += 1

    print("\n=== 输出 ===")
    print(tok.decode(generated))
    if decode_ms:
        arr = np.array(decode_ms)
        print(f"\ndecode: {len(arr)} tokens, 平均 {arr.mean():.0f} ms/token, "
              f"最慢 {arr.max():.0f}, 最快 {arr.min():.0f} ms/token")


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "E:/Cluade_Code/models/qwen2.5-1.5b-instruct-onnx-fp16"
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(model_dir, max_new, threads)
