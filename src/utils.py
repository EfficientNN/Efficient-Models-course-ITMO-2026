"""Общие константы и эталонные (чистый torch) утилиты для int4-квантизации.

Содержит:
  * именованные константы формата int4 и упаковки (никаких «магических» чисел в ядрах);
  * вычисление scale (per-tensor / per-row) для симметричной квантизации;
  * эталонные квантизатор, упаковщик и распаковщик на torch (для валидации ядер);
  * dequantize_int4_to_fp16 — обратное преобразование (torch, без отдельного ядра);
  * проверку «ровно 4x» по памяти.

Формат упакованных данных (layout):
  Матрица (R, C) хранится построчно (row-major). Каждая строка независимо
  дополняется нулевыми nibble'ами до кратности ELEMS = PACK_BITS // 4, поэтому
  упакованный тензор имеет форму (R, ceil(C / ELEMS)) и dtype int8 (PACK_BITS=8)
  или int32 (PACK_BITS=32). Внутри слова элемент с индексом e (0..ELEMS-1)
  лежит в битах [4*e, 4*e + 4): чётный индекс -> младшие 4 бита байта,
  нечётный -> старшие 4 бита. Значение хранится в 4-битном двоичном дополнении.
"""

import torch

# --------------------------------------------------------------------------
# Константы формата int4
# --------------------------------------------------------------------------
INT4_BITS = 4                      # ширина одного значения в битах
INT4_MIN = -8                      # минимум знакового int4
INT4_MAX = 7                       # максимум знакового int4
SYMMETRIC_QMAX = 7                 # scale = max|W| / 7 (симметричный диапазон [-7, 7])
NIBBLE_MASK = 0xF                  # маска младших 4 бит
NIBBLE_SIGN_BIT = 0x8              # знаковый бит nibble'а
ROUND_HALF = 0.5                   # округление «half up»: floor(x + 0.5)
UNIT_SCALE = 1.0                   # scale для нулевого тензора (избегаем деления на 0)

# --------------------------------------------------------------------------
# Константы упаковки
# --------------------------------------------------------------------------
PACK_BITS_INT8 = 8                 # 2 значения int4 в одном int8
PACK_BITS_INT32 = 32               # 8 значений int4 в одном int32
SUPPORTED_PACK_BITS = (PACK_BITS_INT8, PACK_BITS_INT32)
BITS_PER_BYTE = 8
FP16_BITS = 16
EXPECTED_COMPRESSION = FP16_BITS // INT4_BITS   # fp16 -> int4: ровно 4x

DEFAULT_SEED = 0


def cdiv(a: int, b: int) -> int:
    """Целочисленное деление с округлением вверх.

    Вход: a, b — целые, b > 0. Выход: ceil(a / b).
    """
    return (a + b - 1) // b


def validate_pack_bits(pack_bits: int) -> None:
    """Проверяет, что pack_bits ∈ {8, 32}; иначе бросает ValueError."""
    if pack_bits not in SUPPORTED_PACK_BITS:
        raise ValueError(f"pack_bits должен быть одним из {SUPPORTED_PACK_BITS}, получено {pack_bits}")


def elems_per_word(pack_bits: int) -> int:
    """Сколько значений int4 помещается в одно упакованное слово.

    Вход: pack_bits ∈ {8, 32}. Выход: 2 (для int8) или 8 (для int32).
    """
    validate_pack_bits(pack_bits)
    return pack_bits // INT4_BITS


def packed_torch_dtype(pack_bits: int) -> torch.dtype:
    """torch-dtype упакованного тензора: int8 для pack_bits=8, int32 для pack_bits=32."""
    validate_pack_bits(pack_bits)
    return torch.int8 if pack_bits == PACK_BITS_INT8 else torch.int32


def compute_scale(W: torch.Tensor, per_row: bool = False):
    """Считает scale симметричной квантизации и обратный scale (оба float32).

    Вход:
      W       — (R, C) fp16 (или любой float) тензор на любом устройстве;
      per_row — False: один scale на тензор, форма (1,); True: scale на строку, форма (R,).
    Выход: (scale, inv_scale) — float32, scale = max|W| / 7, inv_scale = 1 / scale.
    Ограничения: если max|W| == 0, scale = 1.0 (все коды будут нулями).
    Замечание: ядро умножает на inv_scale (а не делит) — так результат бит-в-бит
    совпадает с torch-эталоном (IEEE-умножение, без приближённого деления).
    """
    abs_w = W.abs()
    if per_row:
        amax = abs_w.amax(dim=1).to(torch.float32)
    else:
        amax = abs_w.amax().to(torch.float32).reshape(1)
    scale = torch.where(amax > 0, amax / SYMMETRIC_QMAX, torch.full_like(amax, UNIT_SCALE))
    inv_scale = 1.0 / scale
    return scale.contiguous(), inv_scale.contiguous()


def quantize_codes_reference(W: torch.Tensor, inv_scale: torch.Tensor) -> torch.Tensor:
    """Эталонная квантизация на torch (те же fp32-операции, что и в ядре).

    Вход: W (R, C) fp16; inv_scale float32 формы (1,) или (R,).
    Выход: (R, C) int32 коды в [-8, 7].
    Алгоритм: x = W * inv_scale -> clamp[-8, 7] -> floor(x + 0.5) (round half up).
    Ноль остаётся нулём: 0 * inv_scale = 0 -> floor(0.5) = 0.
    """
    x = W.to(torch.float32) * inv_scale.reshape(-1, 1)
    x = x.clamp(float(INT4_MIN), float(INT4_MAX))
    return torch.floor(x + ROUND_HALF).to(torch.int32)


def pack_int4_reference(codes: torch.Tensor, pack_bits: int) -> torch.Tensor:
    """Эталонная упаковка int4-кодов на torch.

    Вход: codes (R, C) целые значения в [-8, 7]; pack_bits ∈ {8, 32}.
    Выход: (R, ceil(C / ELEMS)) int8 / int32 (layout — см. docstring модуля).
    Трюк: nibble = code & 0xF (двоичное дополнение в 4 битах), затем
    word = sum(nibble << 4*e); результат считается в int64 и приводится к
    знаковому диапазону слова вычитанием 2**pack_bits.
    """
    epw = elems_per_word(pack_bits)
    rows, cols = codes.shape
    words_per_row = cdiv(cols, epw)
    pad = words_per_row * epw - cols
    codes = codes.to(torch.int64)
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad))
    nibbles = (codes & NIBBLE_MASK).reshape(rows, words_per_row, epw)
    shifts = torch.arange(epw, device=codes.device, dtype=torch.int64) * INT4_BITS
    word = (nibbles << shifts).sum(dim=-1)                  # беззнаковое слово [0, 2**pack_bits)
    modulus = 1 << pack_bits
    word = torch.where(word >= (modulus >> 1), word - modulus, word)
    return word.to(packed_torch_dtype(pack_bits))


def unpack_int4_reference(packed: torch.Tensor, cols: int, pack_bits: int) -> torch.Tensor:
    """Эталонная распаковка: упакованный тензор -> (R, cols) int32 знаковые коды.

    Вход: packed (R, W) int8/int32; cols — исходное число столбцов (без паддинга).
    Трюк: nibble = (word >> 4*e) & 0xF, знаковое расширение (nibble ^ 8) - 8.
    """
    epw = elems_per_word(pack_bits)
    rows, words_per_row = packed.shape
    word = packed.to(torch.int64) & ((1 << pack_bits) - 1)  # перевод в беззнаковое слово
    shifts = torch.arange(epw, device=packed.device, dtype=torch.int64) * INT4_BITS
    nibbles = (word.unsqueeze(-1) >> shifts) & NIBBLE_MASK
    codes = (nibbles ^ NIBBLE_SIGN_BIT) - NIBBLE_SIGN_BIT
    return codes.reshape(rows, words_per_row * epw)[:, :cols].to(torch.int32)


def dequantize_int4_to_fp16(W_packed: torch.Tensor, scale: torch.Tensor, shape, pack_bits: int = PACK_BITS_INT8) -> torch.Tensor:
    """Деквантизация int4 -> fp16 (чистый torch, для валидации).

    Вход:
      W_packed — (R, ceil(C / ELEMS)) int8 / int32;
      scale    — float32, форма (1,) (per-tensor) или (R,) (per-row);
      shape    — (R, C) исходная форма; pack_bits ∈ {8, 32}.
    Выход: (R, C) fp16, row-major. W ≈ code * scale.
    """
    rows, cols = shape
    if scale.numel() not in (1, rows):
        raise ValueError(f"scale.numel() должен быть 1 или {rows}, получено {scale.numel()}")
    codes = unpack_int4_reference(W_packed, cols, pack_bits)
    return (codes.to(torch.float32) * scale.to(torch.float32).reshape(-1, 1)).to(torch.float16)


def check_packed_memory(W: torch.Tensor, W_packed: torch.Tensor, pack_bits: int) -> None:
    """Проверяет выигрыш по памяти ровно в 4x.

    Если C кратно ELEMS (нет паддинга) — строгое равенство
        W_packed.numel() * W_packed.element_size() * 4 == W.numel() * W.element_size().
    Иначе проверяется точный размер с учётом выравнивания строк (паддинг < 1 слова на строку).
    """
    rows, cols = W.shape
    epw = elems_per_word(pack_bits)
    packed_bytes = W_packed.numel() * W_packed.element_size()
    if cols % epw == 0:
        if packed_bytes * EXPECTED_COMPRESSION != W.numel() * W.element_size():
            raise AssertionError(
                f"Память не уменьшилась ровно в {EXPECTED_COMPRESSION}x: "
                f"{packed_bytes} Б (int4) против {W.numel() * W.element_size()} Б (fp16)")
    else:
        expected = rows * cdiv(cols, epw) * (pack_bits // BITS_PER_BYTE)
        if packed_bytes != expected:
            raise AssertionError(f"Неверный размер упаковки: {packed_bytes} Б, ожидалось {expected} Б")
