"""Сравнение скорости X16 @ W4^T vs X16 @ W16^T на размерах Llama-3.2-1B-Instruct.

X и W16 — bfloat16 (как в модели и в Triton-ядре); W4 — int4 после квантизации.
Строки активаций M ∈ {128, 512, 2048}.

Запуск из корня репозитория:
  PYTHONPATH=src python src/bench_llama.py
"""

from __future__ import annotations

import statistics

import torch

from llama_shapes import ACTIVATION_ROWS, LLAMA_LINEAR_SHAPES, LM_HEAD_SHAPE, MODEL_ID
from matmul import matmul_bf16_x_int4
from quantize import quantize_fp16_to_int4
from utils import PACK_BITS_INT8

SEED = 0
WARMUP = 10
ITERS = 50
WEIGHT_STD = 0.02
PACK_BITS = PACK_BITS_INT8
INCLUDE_LM_HEAD = True


def time_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def bench_one(name: str, n_out: int, k_dim: int, m_dim: int) -> dict:
    """Сравнивает Triton int4 и cuBLAS bf16 для одной формы (M, N, K)."""
    W_fp16 = (torch.randn(n_out, k_dim, device="cuda", dtype=torch.float32) * WEIGHT_STD).to(torch.float16)
    W_bf16 = W_fp16.to(torch.bfloat16)
    X = torch.randn(m_dim, k_dim, device="cuda", dtype=torch.bfloat16)

    packed, scale, shape = quantize_fp16_to_int4(W_fp16, pack_bits=PACK_BITS, per_row=True)

    t_w4 = time_ms(lambda: matmul_bf16_x_int4(X, packed, scale, shape, pack_bits=PACK_BITS))
    t_w16 = time_ms(lambda: torch.matmul(X, W_bf16.T))

    return {
        "layer": name,
        "M": m_dim,
        "N": n_out,
        "K": k_dim,
        "ms_w4": t_w4,
        "ms_w16": t_w16,
        "speedup_w4_vs_w16": t_w16 / t_w4,
        "bytes_w4": packed.numel() * packed.element_size(),
        "bytes_w16": W_bf16.numel() * W_bf16.element_size(),
    }


def run_bench(include_lm_head: bool = INCLUDE_LM_HEAD) -> list[dict]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    shapes = list(LLAMA_LINEAR_SHAPES)
    if include_lm_head:
        shapes.append(LM_HEAD_SHAPE)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Модель (размеры весов): {MODEL_ID}")
    print(f"X, W16: bfloat16 | W4: int4 pack={PACK_BITS}, per-row scale")
    print(f"M (токены): {ACTIVATION_ROWS}")
    print()
    print(f"{'layer':<12} {'M':>5} {'N':>6} {'K':>6} {'W4 ms':>9} {'W16 ms':>9} {'W4/W16':>8} {'mem W4/W16':>12}")
    print("-" * 80)

    rows = []
    for name, n_out, k_dim in shapes:
        for m_dim in ACTIVATION_ROWS:
            r = bench_one(name, n_out, k_dim, m_dim)
            rows.append(r)
            mem_ratio = r["bytes_w4"] / r["bytes_w16"]
            print(
                f"{r['layer']:<12} {r['M']:>5} {r['N']:>6} {r['K']:>6} "
                f"{r['ms_w4']:9.4f} {r['ms_w16']:9.4f} {r['speedup_w4_vs_w16']:7.2f}x "
                f"{mem_ratio:11.2%}"
            )
        print()

    # Сводка: среднее ускорение по M
    print("== Сводка: среднее W16_ms / W4_ms по слоям ==")
    for m_dim in ACTIVATION_ROWS:
        subset = [r for r in rows if r["M"] == m_dim]
        avg = statistics.mean(r["speedup_w4_vs_w16"] for r in subset)
        print(f"  M={m_dim:>4}: mean speedup {avg:.2f}x  "
              f"(>1 → int4 быстрее cuBLAS bf16)")
    return rows


if __name__ == "__main__":
    run_bench()
