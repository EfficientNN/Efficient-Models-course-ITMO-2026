"""Triton-квантизация fp16 -> int4 с упаковкой в int8 (2 значения) или int32 (8 значений)."""

import torch
import triton
import triton.language as tl

from utils import (
    INT4_BITS,
    INT4_MAX,
    INT4_MIN,
    NIBBLE_MASK,
    PACK_BITS_INT8,
    ROUND_HALF,
    cdiv,
    check_packed_memory,
    compute_scale,
    elems_per_word,
    packed_torch_dtype,
)


# --------------------------------------------------------------------------
# Константы ядра (tl.constexpr — единственный способ читать глобалы внутри @triton.jit)
# --------------------------------------------------------------------------
_QMIN_F = tl.constexpr(float(INT4_MIN))
_QMAX_F = tl.constexpr(float(INT4_MAX))
_NIBBLE_MASK = tl.constexpr(NIBBLE_MASK)
_NIBBLE_BITS = tl.constexpr(INT4_BITS)
_ROUND_HALF = tl.constexpr(ROUND_HALF)

# --------------------------------------------------------------------------
# Параметры запуска по умолчанию (обоснование)
# --------------------------------------------------------------------------
# Одна программа обрабатывает ~2048 исходных fp16-элементов (4 КиБ чтения):
#  * при num_warps=4 (128 потоков) это 16 элементов/поток — хватает для скрытия
#    латентности HBM, но регистровое давление мало (тайл [BLOCK, ELEMS] в fp32);
#  * для pack=8 это BLOCK_SIZE=1024 слов, для pack=32 — 256 слов.
QUANT_ELEMS_PER_PROGRAM = 2048
QUANT_NUM_WARPS = 4


@triton.jit
def quantize_fp16_to_int4_kernel(
    w_ptr,                      # *fp16, (R, C) row-major
    inv_scale_ptr,              # *fp32, (1,) при PER_ROW=False или (R,) при PER_ROW=True
    out_ptr,                    # *int8 / *int32, (R, WORDS_PER_ROW) row-major
    n_cols,                     # C
    words_per_row,              # ceil(C / ELEMS)
    total_words,                # R * words_per_row
    PER_ROW: tl.constexpr,      # per-row (True) или per-tensor (False) scale
    ELEMS: tl.constexpr,        # значений int4 в слове: 2 (int8) или 8 (int32)
    PACK_BITS: tl.constexpr,    # 8 или 32 — ширина выходного слова
    BLOCK_SIZE: tl.constexpr,   # сколько выходных слов обрабатывает одна программа
):
    """Квантизует fp16 -> int4 (симметрично) и упаковывает nibble'ы в слова.

    Назначение: out[r, j] = pack( clamp(round(W[r, j*ELEMS + e] * inv_scale), -8, 7) for e in 0..ELEMS-1 ).
    Входы: W — fp16 (R, C) row-major; inv_scale — fp32 (1,) или (R,).
    Выход: int8 (PACK_BITS=8) или int32 (PACK_BITS=32), форма (R, ceil(C / ELEMS)).
    Layout: каждая строка независимо дополняется нулями до кратности ELEMS; элемент e
    слова лежит в битах [4e, 4e+4): чётный -> младшие 4 бита, нечётный -> старшие.
    Ограничения: ELEMS * 4 == PACK_BITS; BLOCK_SIZE и ELEMS — степени двойки.
    Трюки: nibble = q & 0xF (двоичное дополнение в 4 битах); упаковка = tl.sum от
    (nibble << 4e) — биты не пересекаются, поэтому сумма равна побитовому OR
    (переполнение int32 на старшем nibble'е — корректный wrap-around);
    floor(x + 0.5) реализован через усечение к нулю + tl.where (без libdevice).
    """
    pid = tl.program_id(axis=0)
    # int64, чтобы не переполнить индексы на больших матрицах
    word_idx = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # [BLOCK]
    word_mask = word_idx < total_words                                      # хвост сетки

    row = word_idx // words_per_row                                         # строка слова
    word_in_row = word_idx - row * words_per_row                            # номер слова в строке
    elem = tl.arange(0, ELEMS)                                              # [ELEMS]

    # Исходный столбец каждого элемента: [BLOCK, ELEMS]
    col = word_in_row[:, None] * ELEMS + elem[None, :]
    # Маска: слово существует И столбец не выходит за C (паддинг строки -> нули)
    in_bounds = word_mask[:, None] & (col < n_cols)

    x = tl.load(w_ptr + row[:, None] * n_cols + col, mask=in_bounds, other=0.0).to(tl.float32)

    if PER_ROW:
        inv_scale = tl.load(inv_scale_ptr + row, mask=word_mask, other=0.0)   # [BLOCK]
        x = x * inv_scale[:, None]
    else:
        x = x * tl.load(inv_scale_ptr)                                        # скаляр

    # clamp в float до округления: результат гарантированно в [-8, 7]
    x = tl.minimum(tl.maximum(x, _QMIN_F), _QMAX_F)

    # floor(x + 0.5): усечение к нулю, затем поправка для отрицательных нецелых
    shifted = x + _ROUND_HALF
    truncated = shifted.to(tl.int32)
    q = tl.where(shifted < truncated.to(tl.float32), truncated - 1, truncated)  # int32 в [-8, 7]

    # Упаковка: двоичное дополнение в 4 битах, сдвиг на 4*e, суммирование по ELEMS
    nibble = q & _NIBBLE_MASK                                               # [BLOCK, ELEMS], 0..15
    bit_shift = elem * _NIBBLE_BITS                                         # [ELEMS]: 0, 4, 8, ...
    word = tl.sum(nibble << bit_shift[None, :], axis=1)                     # [BLOCK] int32

    if PACK_BITS == 8:
        # приведение int32 -> int8 отбрасывает старшие биты (нужные 8 бит уже в слове)
        tl.store(out_ptr + word_idx, word.to(tl.int8), mask=word_mask)
    else:
        tl.store(out_ptr + word_idx, word, mask=word_mask)


def quantize_fp16_to_int4(W: torch.Tensor, pack_bits: int = PACK_BITS_INT8, per_row: bool = False,
                          block_size: int = None, num_warps: int = QUANT_NUM_WARPS):
    """Квантизует fp16-матрицу в int4 и упаковывает (Triton).

    Вход:
      W         — (M, N) torch.float16, CUDA, row-major (не-contiguous будет скопирован);
      pack_bits — 8 (2 значения в int8) или 32 (8 значений в int32);
      per_row   — False: scale = max|W| / 7 на весь тензор (форма (1,));
                  True:  отдельный scale на каждую строку (форма (M,));
      block_size — число упакованных слов на программу (степень двойки);
                  по умолчанию QUANT_ELEMS_PER_PROGRAM // ELEMS;
      num_warps — число варпов (по умолчанию 4).
    Выход: (W_packed, scale, shape):
      W_packed — (M, ceil(N / ELEMS)) int8 / int32;
      scale    — float32, (1,) или (M,);
      shape    — (M, N) для деквантизации.
    Ограничения: строки дополняются нулями до кратности ELEMS (N % 2 == 0 для pack=8,
    N % 8 == 0 для pack=32 — условие «ровно 4x» без паддинга). Свойства: ноль -> 0,
    диапазон кодов [-8, 7], округление half-up.
    """
    if not (W.is_cuda and W.dtype == torch.float16 and W.dim() == 2):
        raise ValueError("W должен быть 2D torch.float16 тензором на CUDA")
    if W.numel() == 0:
        raise ValueError("пустая матрица не поддерживается")
    W = W.contiguous()
    rows, cols = W.shape
    epw = elems_per_word(pack_bits)
    words_per_row = cdiv(cols, epw)
    total_words = rows * words_per_row

    if block_size is None:
        block_size = max(QUANT_ELEMS_PER_PROGRAM // epw, 1)
    if block_size & (block_size - 1):
        raise ValueError("block_size должен быть степенью двойки")

    scale, inv_scale = compute_scale(W, per_row)
    out = torch.empty((rows, words_per_row), dtype=packed_torch_dtype(pack_bits), device=W.device)

    grid = (triton.cdiv(total_words, block_size),)
    quantize_fp16_to_int4_kernel[grid](
        W, inv_scale, out, cols, words_per_row, total_words,
        PER_ROW=per_row, ELEMS=epw, PACK_BITS=pack_bits, BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    check_packed_memory(W, out, pack_bits)
    return out, scale, (rows, cols)
