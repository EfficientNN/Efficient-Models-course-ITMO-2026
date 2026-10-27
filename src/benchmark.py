"""Бенчмарк: время (cuda.Event) и пропускная способность, сравнение с наивным torch.

Запуск: python benchmark.py
"""

import statistics

import torch

from matmul import matmul_bf16_x_int4
from quantize import quantize_fp16_to_int4
from utils import (
    PACK_BITS_INT8,
    SUPPORTED_PACK_BITS,
    compute_scale,
    dequantize_int4_to_fp16,
    pack_int4_reference,
    quantize_codes_reference,
)


SEED = 0
BENCH_WARMUP = 10                    # прогрев: компиляция и кэши
BENCH_ITERS = 50                     # число замеров; берём медиану
BYTES_PER_GB = 1e9
MS_PER_S = 1e3
QUANT_SHAPE = (4096, 4096)           # (N, K) типичного слоя LLM
MATMUL_M_LIST = (1, 16, 2048)        # декодинг (memory-bound) и prefill (compute-bound)
BF16_BYTES = 2
FP32_BYTES = 4
SWEEP_M = 16
SWEEP_CONFIGS = [                    # (block_n, block_k, num_warps, num_stages)
    (32, 64, 4, 3), (32, 128, 4, 3), (64, 64, 4, 3), (64, 128, 4, 3),
    (64, 128, 8, 3), (128, 64, 8, 3), (32, 64, 2, 4), (64, 64, 4, 2),
]


def time_ms(fn, warmup: int = BENCH_WARMUP, iters: int = BENCH_ITERS) -> float:
    """Медианное время вызова fn() в мс по torch.cuda.Event.

    Вход: fn — функция без аргументов (запускает GPU-работу); warmup/iters — число прогревов/замеров.
    Выход: медиана времени одного вызова, мс.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def gbps(num_bytes: int, ms: float) -> float:
    """Пропускная способность в ГБ/с: байты / время."""
    return num_bytes / BYTES_PER_GB / (ms / MS_PER_S)


def bench_quantize() -> None:
    """Сравнивает Triton-квантизацию с torch-пайплайном (коды + упаковка) для pack=8 и pack=32."""
    rows, cols = QUANT_SHAPE
    W = (torch.randn(rows, cols, device="cuda") * 0.1).to(torch.float16)
    print(f"== Квантизация fp16 -> int4, W {QUANT_SHAPE} ==")
    print(f"{'pack':>5} {'impl':>8} {'ms':>9} {'GB/s':>9} {'speedup':>8}")
    for pb in SUPPORTED_PACK_BITS:
        packed, _, _ = quantize_fp16_to_int4(W, pack_bits=pb)
        moved = W.numel() * W.element_size() + packed.numel() * packed.element_size()   # чтение W + запись packed
        t_tri = time_ms(lambda: quantize_fp16_to_int4(W, pack_bits=pb))

        def naive():
            _, inv = compute_scale(W, False)
            return pack_int4_reference(quantize_codes_reference(W, inv), pb)
        t_ref = time_ms(naive, warmup=3, iters=10)
        print(f"{pb:>5} {'triton':>8} {t_tri:9.4f} {gbps(moved, t_tri):9.1f} {t_ref / t_tri:7.1f}x")
        print(f"{pb:>5} {'torch':>8} {t_ref:9.4f} {gbps(moved, t_ref):9.1f} {'1.0x':>8}")
    print("(время Triton включает вычисление scale: abs().amax() в torch)")


def bench_matmul() -> None:
    """Сравнивает Triton-матмул с наивным X @ dequant(W).T и с cuBLAS на bf16-весах."""
    n_dim, k_dim = QUANT_SHAPE
    W = (torch.randn(n_dim, k_dim, device="cuda") * 0.1).to(torch.float16)
    W_bf16 = W.to(torch.bfloat16)
    print(f"\n== Матмул bf16 x int4 -> bf16, W ({n_dim}, {k_dim}) ==")
    print(f"{'pack':>5} {'M':>5} {'impl':>16} {'ms':>9} {'GB/s':>9} {'vs naive':>9}")
    for pb in SUPPORTED_PACK_BITS:
        packed, scale, shape = quantize_fp16_to_int4(W, pack_bits=pb)
        for m_dim in MATMUL_M_LIST:
            X = torch.randn(m_dim, k_dim, device="cuda").to(torch.bfloat16)
            moved = (X.numel() * BF16_BYTES + packed.numel() * packed.element_size() + m_dim * n_dim * BF16_BYTES)
            t_tri = time_ms(lambda: matmul_bf16_x_int4(X, packed, scale, shape, pack_bits=pb))
            t_naive = time_ms(lambda: X @ dequantize_int4_to_fp16(packed, scale, shape, pb).to(torch.bfloat16).T,
                              warmup=3, iters=10)
            moved_bf16 = X.numel() * BF16_BYTES + W_bf16.numel() * BF16_BYTES + m_dim * n_dim * BF16_BYTES
            t_cublas = time_ms(lambda: X @ W_bf16.T)
            print(f"{pb:>5} {m_dim:>5} {'triton int4':>16} {t_tri:9.4f} {gbps(moved, t_tri):9.1f} {t_naive / t_tri:8.1f}x")
            print(f"{pb:>5} {m_dim:>5} {'torch naive':>16} {t_naive:9.4f} {gbps(moved, t_naive):9.1f} {'1.0x':>9}")
            print(f"{pb:>5} {m_dim:>5} {'cuBLAS bf16 W':>16} {t_cublas:9.4f} {gbps(moved_bf16, t_cublas):9.1f} {'-':>9}")
    print("GB/s считается по «полезным» байтам: X + упакованные W + выход (для cuBLAS — W в bf16).")


def sweep_matmul_configs() -> None:
    """Небольшой перебор (BLOCK_N, BLOCK_K, num_warps, num_stages) при M=SWEEP_M, pack=8.

    Нужен, чтобы обосновать дефолты на конкретном GPU; печатает варианты по возрастанию времени.
    """
    n_dim, k_dim = QUANT_SHAPE
    W = (torch.randn(n_dim, k_dim, device="cuda") * 0.1).to(torch.float16)
    packed, scale, shape = quantize_fp16_to_int4(W, pack_bits=PACK_BITS_INT8)
    X = torch.randn(SWEEP_M, k_dim, device="cuda").to(torch.bfloat16)
    results = []
    for block_n, block_k, warps, stages in SWEEP_CONFIGS:
        try:
            ms = time_ms(lambda: matmul_bf16_x_int4(X, packed, scale, shape, block_n=block_n, block_k=block_k,
                                                    num_warps=warps, num_stages=stages), warmup=5, iters=20)
            results.append((ms, block_n, block_k, warps, stages))
        except Exception as exc:                              # конфиг может не влезть в shared memory
            print(f"  BLOCK_N={block_n} BLOCK_K={block_k} warps={warps} stages={stages}: пропущен ({type(exc).__name__})")
    print(f"\n== Перебор конфигов, M={SWEEP_M}, pack=8 ==")
    for ms, block_n, block_k, warps, stages in sorted(results):
        print(f"  BLOCK_N={block_n:>3} BLOCK_K={block_k:>3} warps={warps} stages={stages}: {ms:.4f} мс")


def run_benchmarks() -> None:
    """Запускает все бенчмарки."""
    torch.manual_seed(SEED)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    bench_quantize()
    bench_matmul()
    sweep_matmul_configs()


if __name__ == "__main__":
    run_benchmarks()
