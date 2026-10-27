"""Triton-матмул bf16 x int4 -> bf16: out = X @ W4^T (с масштабом scale)."""

import torch
import triton
import triton.language as tl

from utils import (
    INT4_BITS,
    PACK_BITS_INT8,
    cdiv,
    elems_per_word,
    packed_torch_dtype,
)


# --------------------------------------------------------------------------
# Константы ядра
# --------------------------------------------------------------------------
INT32_BITS = 32
NIBBLE_TOP_SHIFT = INT32_BITS - INT4_BITS      # 28: nibble выравнивается к старшим битам int32
_NIBBLE_BITS = tl.constexpr(INT4_BITS)
_NIBBLE_TOP_SHIFT = tl.constexpr(NIBBLE_TOP_SHIFT)

# --------------------------------------------------------------------------
# Параметры запуска по умолчанию (обоснование)
# --------------------------------------------------------------------------
MATMUL_MIN_TILE = 16          # tl.dot требует размеров тайла >= 16 по каждой оси
MATMUL_MAX_BLOCK_M = 64       # 64x64 fp32-аккумулятор = 32 рег./поток при 4 варпах: без спиллов
MATMUL_BLOCK_N = 64           # ширина тайла по N для M > MATMUL_SKINNY_M
MATMUL_BLOCK_N_SKINNY = 32    # для «тонких» M (декодинг) — больше программ, лучше загрузка SM
MATMUL_SKINNY_M = 16
MATMUL_BLOCK_K = 64           # кратно ELEMS (2 и 8); 64 bf16 = 128 Б на строку X — полная кэш-линия
MATMUL_NUM_WARPS = 4          # тайл 64x64 покрывается 4 варпами без избыточности
MATMUL_NUM_STAGES = 3         # 3 стадии программного конвейера; в shared-памяти T4 (64 КиБ) помещается
BF16_MIN_CAPABILITY_MAJOR = 8  # tensor cores с bf16 — начиная с Ampere (sm_80)


@triton.jit
def matmul_bf16_x_int4_kernel(
    x_ptr,                      # *bf16, (M, K)
    w_ptr,                      # *int8 / *int32, (N, WORDS_PER_ROW) — упакованные веса
    scale_ptr,                  # *fp32, (1,) при PER_ROW=False или (N,) при PER_ROW=True
    out_ptr,                    # *bf16, (M, N)
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_ww,
    stride_om, stride_on,
    PER_ROW: tl.constexpr,      # scale по N (True) или один на тензор (False)
    ELEMS: tl.constexpr,        # значений int4 в слове: 2 (int8) или 8 (int32)
    UPCAST_FP32: tl.constexpr,  # True для GPU < sm_80: dot в fp32 вместо bf16
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,      # обязан быть кратен ELEMS
):
    """out[m, n] = scale[n] * sum_k X[m, k] * code(W4[n, k]), аккумулятор в fp32.

    Входы: X — bf16 (M, K) row-major; W4 — упакованные int4 (N, ceil(K / ELEMS)),
    int8 или int32, row-major, строки дополнены нулевыми nibble'ами; scale — fp32.
    Выход: bf16 (M, N) row-major.
    Ограничения: BLOCK_K % ELEMS == 0, BLOCK_M/N/K >= 16 и степени двойки; M, N, K — любые
    (границы закрыты масками). Для UPCAST_FP32 tl.dot идёт по FMA/TF32-пути.
    Трюки: знаковое расширение nibble'а без ветвлений —
        code = (word << (28 - 4e)) >> 28   (арифметический сдвиг int32),
    что эквивалентно lo = (packed << 4) >> 4 и hi = packed >> 4 из int8-случая, но
    работает и для int32. Каждый K-индекс адресует своё слово k // ELEMS (повторные
    чтения одного слова попадают в L1) — это даёт простой, портабельный по версиям
    Triton код без tl.join/interleave.
    """
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Положение nibble'а внутри слова; не зависит от итерации, т.к. BLOCK_K % ELEMS == 0
    word_in_block = offs_k // ELEMS                              # [BLOCK_K]: индекс слова в блоке
    bit_shift = (offs_k % ELEMS) * _NIBBLE_BITS                  # [BLOCK_K]: 0, 4, 8, ...
    left_shift = -bit_shift + _NIBBLE_TOP_SHIFT                  # 28 - 4e

    offs_m64 = offs_m.to(tl.int64)
    offs_n64 = offs_n.to(tl.int64)
    x_ptrs = x_ptr + offs_m64[:, None] * stride_xm + offs_k[None, :] * stride_xk          # [BM, BK]
    w_ptrs = w_ptr + offs_n64[None, :] * stride_wn + word_in_block[:, None] * stride_ww   # [BK, BN]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        packed = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)

        # Распаковка: сдвиг влево выносит нужный nibble в биты 28..31, арифметический
        # сдвиг вправо возвращает его на место с расширением знака
        w = (packed.to(tl.int32) << left_shift[:, None]) >> _NIBBLE_TOP_SHIFT
        w = tl.where(k_mask[:, None], w, 0)                      # хвост K — строго нули

        if UPCAST_FP32:
            acc = tl.dot(x.to(tl.float32), w.to(tl.float32), acc)
        else:
            acc = tl.dot(x, w.to(tl.bfloat16), acc)              # int4 в bf16 представим точно

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += (BLOCK_K // ELEMS) * stride_ww

    # Эпилог: масштаб (broadcast по N) и запись в bf16
    if PER_ROW:
        scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc * scale[None, :]
    else:
        acc = acc * tl.load(scale_ptr)

    out_ptrs = out_ptr + offs_m64[:, None] * stride_om + offs_n64[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def select_matmul_config(M: int) -> dict:
    """Эвристика выбора тайлов по M (значения обоснованы в константах модуля).

    Вход: M — число строк X. Выход: dict(block_m, block_n, block_k).
    M <= 16 (декодинг, memory-bound): BLOCK_M=16, BLOCK_N=32 — больше программ по N.
    M > 16: BLOCK_M = ближайшая степень двойки (<= 64), BLOCK_N=64.
    """
    block_m = min(max(triton.next_power_of_2(M), MATMUL_MIN_TILE), MATMUL_MAX_BLOCK_M)
    block_n = MATMUL_BLOCK_N_SKINNY if M <= MATMUL_SKINNY_M else MATMUL_BLOCK_N
    return dict(block_m=block_m, block_n=block_n, block_k=MATMUL_BLOCK_K)


def matmul_bf16_x_int4(x: torch.Tensor, w_packed: torch.Tensor, scale: torch.Tensor, shape,
                       pack_bits: int = PACK_BITS_INT8, block_m: int = None, block_n: int = None,
                       block_k: int = None, num_warps: int = MATMUL_NUM_WARPS,
                       num_stages: int = MATMUL_NUM_STAGES) -> torch.Tensor:
    """Считает X @ W4^T, где W4 — упакованная int4-матрица (N, K) (Triton).

    Вход:
      x        — (M, K) torch.bfloat16, CUDA (не-contiguous будет скопирован);
      w_packed — (N, ceil(K / ELEMS)) int8 / int32 из quantize_fp16_to_int4;
      scale    — float32, (1,) (per-tensor) или (N,) (per-row по N);
      shape    — (N, K) исходная форма W;
      pack_bits — 8 или 32 (должен совпадать с упаковкой);
      block_m/n/k, num_warps, num_stages — параметры запуска (None -> эвристика).
    Выход: (M, N) torch.bfloat16.
    Ограничения: block_k кратен ELEMS и >= 16, степени двойки; на GPU < sm_80 dot
    выполняется в fp32 (bf16 tensor cores отсутствуют) — корректно, но медленнее.
    """
    n_out, k_dim = shape
    epw = elems_per_word(pack_bits)
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2):
        raise ValueError("x должен быть 2D torch.bfloat16 тензором на CUDA")
    if x.shape[1] != k_dim:
        raise ValueError(f"K не совпадает: x.shape[1]={x.shape[1]}, shape[1]={k_dim}")
    if w_packed.dtype != packed_torch_dtype(pack_bits) or tuple(w_packed.shape) != (n_out, cdiv(k_dim, epw)):
        raise ValueError(f"w_packed: ожидалось {packed_torch_dtype(pack_bits)} {(n_out, cdiv(k_dim, epw))}, "
                         f"получено {w_packed.dtype} {tuple(w_packed.shape)}")
    if scale.dtype != torch.float32 or scale.numel() not in (1, n_out):
        raise ValueError("scale: float32 с numel 1 или N")
    if x.shape[0] == 0 or n_out == 0 or k_dim == 0:
        raise ValueError("пустые размеры не поддерживаются")

    x = x.contiguous()
    scale = scale.to(x.device).contiguous()
    m_dim = x.shape[0]

    cfg = select_matmul_config(m_dim)
    block_m = block_m or cfg["block_m"]
    block_n = block_n or cfg["block_n"]
    block_k = block_k or cfg["block_k"]
    if block_k % epw or min(block_m, block_n, block_k) < MATMUL_MIN_TILE:
        raise ValueError(f"block_k должен быть кратен {epw}, все блоки >= {MATMUL_MIN_TILE}")
    for blk in (block_m, block_n, block_k):
        if blk & (blk - 1):
            raise ValueError("block_m/n/k должны быть степенями двойки")

    per_row = scale.numel() != 1
    upcast = torch.cuda.get_device_capability(x.device)[0] < BF16_MIN_CAPABILITY_MAJOR
    out = torch.empty((m_dim, n_out), dtype=torch.bfloat16, device=x.device)

    grid = (triton.cdiv(n_out, block_n), triton.cdiv(m_dim, block_m))
    matmul_bf16_x_int4_kernel[grid](
        x, w_packed, scale, out,
        m_dim, n_out, k_dim,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        out.stride(0), out.stride(1),
        PER_ROW=per_row, ELEMS=epw, UPCAST_FP32=upcast,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
