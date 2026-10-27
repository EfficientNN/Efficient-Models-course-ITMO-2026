"""Тесты корректности квантизации и матмула. Запуск: python tests.py"""

import itertools
import math

import torch

from matmul import matmul_bf16_x_int4
from quantize import quantize_fp16_to_int4
from utils import (
    INT4_BITS,
    INT4_MAX,
    INT4_MIN,
    NIBBLE_MASK,
    PACK_BITS_INT8,
    PACK_BITS_INT32,
    SUPPORTED_PACK_BITS,
    check_packed_memory,
    compute_scale,
    dequantize_int4_to_fp16,
    pack_int4_reference,
    quantize_codes_reference,
    unpack_int4_reference,
)

# --------------------------------------------------------------------------
# Константы тестов
# --------------------------------------------------------------------------
TEST_SEED = 0
DEVICE = "cuda"
EDGE_SIZES = (1, 17, 128, 1000)                     # «некруглые» размеры для M, N, K
QUANT_SHAPES = ((1, 1), (17, 33), (128, 128), (1000, 17), (64, 1000))
ACCURACY_SHAPE = (1024, 1024)
WEIGHT_STD = 0.1                                     # W = randn * 0.1
MATMUL_ATOL = 1e-2
MATMUL_RTOL = 1e-2
PREDICTED_ERR_TOL = 0.10                             # допуск к теоретической оценке ошибки квантизации
UNIFORM_NOISE_DIVISOR = 12.0                         # дисперсия равномерного шума округления = шаг^2 / 12
PER_ROW_MATMUL_SHAPES = ((1, 17, 1000), (17, 1000, 128), (128, 128, 17), (1000, 1, 1))  # (M, N, K)


def make_weights(rows: int, cols: int) -> torch.Tensor:
    """Гауссовы fp16-веса N(0, 0.1^2) формы (rows, cols) на CUDA (генерация в fp32, затем fp16)."""
    return (torch.randn(rows, cols, device=DEVICE, dtype=torch.float32) * WEIGHT_STD).to(torch.float16)


# ==========================================================================
# Тесты. Каждый тест — функция с assert; возвращает короткую строку с деталями.
# ==========================================================================
def test_pack_roundtrip(rows: int, cols: int, pack_bits: int) -> str:
    """Что проверяем: pack -> unpack возвращает исходные коды (все значения от -8 до 7).
    Зачем: ловит ошибки знака и паддинга строк в эталонных torch-функциях (CPU)."""
    codes = torch.arange(INT4_MIN, INT4_MAX + 1, dtype=torch.int32).repeat(rows * cols)[: rows * cols].reshape(rows, cols)
    packed = pack_int4_reference(codes, pack_bits)
    assert torch.equal(unpack_int4_reference(packed, cols, pack_bits), codes), "коды изменились после pack -> unpack"
    return f"{rows}x{cols}"


def test_pack_layout_known_vector() -> str:
    """Что проверяем: на вектор [1, -2, 3, -4, 5, -6, 7, -8] упаковка даёт ровно те слова,
    которые мы считаем вручную: чётный индекс -> младшие 4 бита, нечётный -> старшие, знак в
    двоичном дополнении. Зачем: фиксирует формат хранения, от которого зависит матмул."""
    codes_list = [1, -2, 3, -4, 5, -6, 7, -8]
    codes = torch.tensor([codes_list], dtype=torch.int32)
    nibbles = [c & NIBBLE_MASK for c in codes_list]

    expected_int8 = []
    for i in range(0, len(nibbles), 2):
        byte = nibbles[i] | (nibbles[i + 1] << INT4_BITS)
        expected_int8.append(byte - 256 if byte >= 128 else byte)
    got_int8 = pack_int4_reference(codes, PACK_BITS_INT8).tolist()
    assert got_int8 == [expected_int8], f"int8: {got_int8} != {[expected_int8]}"

    word = sum(n << (INT4_BITS * i) for i, n in enumerate(nibbles))
    word = word - (1 << 32) if word >= (1 << 31) else word
    got_int32 = pack_int4_reference(codes, PACK_BITS_INT32).tolist()
    assert got_int32 == [[word]], f"int32: {got_int32} != {[[word]]}"
    return f"int8 -> {expected_int8}, int32 -> {word}"


def test_quant_kernel_equals_torch(shape, pack_bits: int, per_row: bool) -> str:
    """Что проверяем: упакованные данные Triton-ядра совпадают бит-в-бит с независимой
    torch-реализацией (scale -> округление -> упаковка). Зачем: основной тест корректности
    ядра квантизации, включая паддинг строк и хвосты блоков."""
    rows, cols = shape
    W = make_weights(rows, cols)
    packed, scale, out_shape = quantize_fp16_to_int4(W, pack_bits=pack_bits, per_row=per_row)
    assert out_shape == (rows, cols), f"shape: {out_shape}"
    _, inv_scale = compute_scale(W, per_row)
    codes_ref = quantize_codes_reference(W, inv_scale)
    expected = pack_int4_reference(codes_ref, pack_bits)
    assert packed.dtype == expected.dtype and packed.shape == expected.shape, "не совпали dtype/shape упаковки"
    assert torch.equal(packed, expected), "упакованные данные отличаются от эталона"
    assert INT4_MIN <= int(codes_ref.min()) and int(codes_ref.max()) <= INT4_MAX, "код вне диапазона int4"
    return f"{rows}x{cols}"


def test_quant_error_is_within_one_step(pack_bits: int, per_row: bool) -> str:
    """Что проверяем на W ~ N(0, 0.1^2), 1024x1024 (quantize -> dequantize):
    1) |W - W_deq| <= scale.max() (allclose с atol=scale.max(), rtol=0);
    2) нули остаются нулями (симметричная квантизация);
    3) относительная ошибка по Фробениусу совпадает с теорией: шум округления равномерный,
       поэтому ||err|| / ||W|| ≈ sqrt(mean(scale^2) / 12) / rms(W) (допуск 10 %)."""
    rows, cols = ACCURACY_SHAPE
    W = make_weights(rows, cols)
    W[::7, ::5] = 0                                          # заведомые нули
    packed, scale, shape = quantize_fp16_to_int4(W, pack_bits=pack_bits, per_row=per_row)
    W_deq = dequantize_int4_to_fp16(packed, scale, shape, pack_bits)

    assert torch.allclose(W, W_deq, atol=scale.max().item(), rtol=0), "ошибка больше одного шага scale"
    assert bool((W_deq[W == 0] == 0).all()), "ноль перестал быть нулём"

    W32, D32 = W.float(), W_deq.float()
    measured = ((W32 - D32).norm() / W32.norm()).item()
    noise_ms = (scale.float() ** 2).mean().item() / UNIFORM_NOISE_DIVISOR
    predicted = math.sqrt(noise_ms / (W32 ** 2).mean().item())
    assert abs(measured - predicted) / predicted < PREDICTED_ERR_TOL, \
        f"ошибка {measured:.4f} не совпала с теорией {predicted:.4f}"
    return f"rel_fro={measured:.4f} (теория {predicted:.4f}), max|err|={(W32 - D32).abs().max().item():.4f} <= scale={scale.max().item():.4f}"


def test_memory_is_4x_smaller(pack_bits: int) -> str:
    """Что проверяем: packed.numel() * element_size * 4 == W.numel() * element_size (1024x1024),
    а для матрицы 17x33 (строки с паддингом) — точный ожидаемый размер.
    Зачем: гарантия выигрыша по памяти."""
    W = make_weights(*ACCURACY_SHAPE)
    packed, _, _ = quantize_fp16_to_int4(W, pack_bits=pack_bits)
    packed_bytes, fp16_bytes = packed.numel() * packed.element_size(), W.numel() * W.element_size()
    assert packed_bytes * 4 == fp16_bytes, f"{packed_bytes} Б * 4 != {fp16_bytes} Б"
    W_odd = make_weights(17, 33)
    packed_odd, _, _ = quantize_fp16_to_int4(W_odd, pack_bits=pack_bits)
    check_packed_memory(W_odd, packed_odd, pack_bits)
    return f"{fp16_bytes} Б (fp16) -> {packed_bytes} Б (int4)"


def test_matmul_matches_reference(M: int, N: int, K: int, pack_bits: int, per_row: bool = False) -> float:
    """Что проверяем: Triton-матмул X @ W4^T совпадает с X.float() @ dequant(W).T
    (atol = rtol = 1e-2). Возвращает max abs error. Зачем: основной тест матмула,
    включая M, N, K не кратные размерам блоков."""
    X = torch.randn(M, K, device=DEVICE, dtype=torch.float32).to(torch.bfloat16)
    W = make_weights(N, K)
    packed, scale, shape = quantize_fp16_to_int4(W, pack_bits=pack_bits, per_row=per_row)
    out = matmul_bf16_x_int4(X, packed, scale, shape, pack_bits=pack_bits)
    assert out.dtype == torch.bfloat16 and out.shape == (M, N), f"выход {out.dtype} {tuple(out.shape)}"
    ref = X.float() @ dequantize_int4_to_fp16(packed, scale, shape, pack_bits).float().T
    max_err = (out.float() - ref).abs().max().item()
    assert torch.allclose(out.float(), ref, atol=MATMUL_ATOL, rtol=MATMUL_RTOL), \
        f"M={M} N={N} K={K} pack={pack_bits} per_row={per_row}: max_abs_err={max_err:.5f}"
    return max_err


def test_rejects_bad_inputs() -> str:
    """Что проверяем: неверные dtype и pack_bits отвергаются ValueError, а не падением ядра."""
    W = make_weights(8, 16)
    packed, scale, shape = quantize_fp16_to_int4(W)
    X_fp32 = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
    bad_calls = {
        "x не bf16": lambda: matmul_bf16_x_int4(X_fp32, packed, scale, shape),
        "W не fp16": lambda: quantize_fp16_to_int4(W.float()),
        "pack_bits=16": lambda: quantize_fp16_to_int4(W, pack_bits=16),
    }
    for label, call in bad_calls.items():
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"«{label}» не вызвал ValueError")
    return "3 некорректных вызова отвергнуты"


# ==========================================================================
# Запуск и отчёт
# ==========================================================================
def run_group(title: str, cases) -> bool:
    """Запускает набор тестов и печатает одну строку-итог (+ строки по упавшим).

    cases — список (функция, аргументы). Если тест вернул число, оно считается ошибкой
    (берётся максимум); если строку — печатается как деталь (для одиночных тестов)."""
    failures, details, worst = [], [], None
    for fn, args in cases:
        try:
            result = fn(*args)
            if isinstance(result, float):
                worst = result if worst is None else max(worst, result)
            elif isinstance(result, str):
                details.append(result)
        except Exception as exc:                              # собираем все падения, не останавливаясь
            failures.append(f"{fn.__name__}{args}: {type(exc).__name__}: {exc}")
    passed = len(cases) - len(failures)
    mark = "OK  " if not failures else "FAIL"
    extra = f"  worst max_abs_err = {worst:.5f}" if worst is not None else ""
    print(f"[{mark}] {title}: {passed}/{len(cases)}{extra}")
    if len(cases) <= 4:                                       # для маленьких групп показываем детали
        for d in details:
            print(f"        {d}")
    elif details and not failures:
        print(f"        например: {details[0]}")
    for f in failures:
        print(f"        ✗ {f}")
    return not failures


def run_all_tests() -> None:
    """Запускает все группы тестов; при любом падении бросает AssertionError."""
    torch.manual_seed(TEST_SEED)
    torch.cuda.manual_seed_all(TEST_SEED)
    packs = SUPPORTED_PACK_BITS
    ok = [
        run_group("Эталон: pack -> unpack возвращает исходные коды",
                  [(test_pack_roundtrip, (*s, pb)) for pb in packs for s in ((1, 1), (3, 5), (4, 8), (7, 17))]),
        run_group("Эталон: формат упаковки на известном векторе", [(test_pack_layout_known_vector, ())]),
        run_group("Квантизация: Triton == torch-эталон (бит-в-бит), 5 форм x 2 pack x 2 режима scale",
                  [(test_quant_kernel_equals_torch, (s, pb, pr)) for s in QUANT_SHAPES for pb in packs for pr in (False, True)]),
    ]
    for pb, pr in itertools.product(packs, (False, True)):
        ok.append(run_group(f"Квантизация: ошибка и нули (1024x1024), pack={pb}, per_row={pr}",
                            [(test_quant_error_is_within_one_step, (pb, pr))]))
    for pb in packs:
        ok.append(run_group(f"Память: ровно 4x, pack={pb}", [(test_memory_is_4x_smaller, (pb,))]))
    for pb in packs:
        ok.append(run_group(f"Матмул: M, N, K ∈ {set(EDGE_SIZES)}, pack={pb} (64 комбинации)",
                            [(test_matmul_matches_reference, (m, n, k, pb)) for m, n, k in itertools.product(EDGE_SIZES, repeat=3)]))
    for pb in packs:
        ok.append(run_group(f"Матмул: per-row scale, pack={pb}",
                            [(test_matmul_matches_reference, (*mnk, pb, True)) for mnk in PER_ROW_MATMUL_SHAPES]))
    ok.append(run_group("Валидация входов", [(test_rejects_bad_inputs, ())]))

    if not all(ok):
        raise AssertionError("Есть упавшие тесты — см. строки FAIL выше")
    print("\nВСЕ ТЕСТЫ ПРОЙДЕНЫ")


if __name__ == "__main__":
    run_all_tests()
