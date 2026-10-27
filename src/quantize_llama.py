"""Квантизация линейных слоёв Llama-3.2-1B-Instruct через QuantizedLinear.

Пример в ноутбуке / скрипте:
  model, tokenizer = load_quantized_llama()
  logits = run_dummy_forward(model)
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_shapes import MODEL_ID
from linear import QuantizedLinear, replace_linear_with_quantized
from utils import PACK_BITS_INT8


def count_linears(module: torch.nn.Module) -> tuple[int, int]:
    """(число nn.Linear, число QuantizedLinear)."""
    n_fp, n_q = 0, 0
    for m in module.modules():
        if isinstance(m, QuantizedLinear):
            n_q += 1
        elif isinstance(m, torch.nn.Linear):
            n_fp += 1
    return n_fp, n_q


def bytes_of_params(module: torch.nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def load_quantized_llama(
    model_id: str = MODEL_ID,
    *,
    pack_bits: int = PACK_BITS_INT8,
    per_row: bool = True,
    quantize_lm_head: bool = False,
    device: str = "cuda",
    dtype=torch.bfloat16,
):
    """Загружает Llama, заменяет nn.Linear на QuantizedLinear, возвращает (model, tokenizer).

    lm_head по умолчанию не квантуется (tied embeddings / большая матрица).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("нужна CUDA")

    skip = set() if quantize_lm_head else {"lm_head"}
    print(f"Загрузка {model_id} ({dtype})…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
    )
    model.eval()

    bytes_before = bytes_of_params(model)
    n_fp0, _ = count_linears(model)
    print(f"  Linear до квантизации: {n_fp0}, память ≈ {bytes_before / 1e9:.2f} ГБ")

    print(f"Квантизация Linear → QuantizedLinear (pack={pack_bits}, per_row={per_row}, skip={skip})…")
    n_replaced = replace_linear_with_quantized(
        model, pack_bits=pack_bits, per_row=per_row, skip_names=skip
    )
    torch.cuda.empty_cache()

    bytes_after = bytes_of_params(model)
    n_fp, n_q = count_linears(model)
    print(f"  заменено слоёв: {n_replaced}")
    print(f"  осталось nn.Linear: {n_fp}, QuantizedLinear: {n_q}")
    print(
        f"  память ≈ {bytes_after / 1e9:.2f} ГБ  "
        f"(×{bytes_after / max(bytes_before, 1):.2f} от исходной)"
    )
    return model, tokenizer


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 32, device: str = "cuda") -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"generate: {max_new_tokens} new tokens за {dt:.2f} с")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Int4 QuantizedLinear → Llama-3.2-1B")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--pack-bits", type=int, default=PACK_BITS_INT8, choices=(8, 32))
    parser.add_argument("--per-tensor", action="store_true")
    parser.add_argument("--quantize-lm-head", action="store_true")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_quantized_llama(
        args.model,
        pack_bits=args.pack_bits,
        per_row=not args.per_tensor,
        quantize_lm_head=args.quantize_lm_head,
    )
    run_dummy_forward(model)

    if not args.skip_generate:
        prompt = args.prompt or "What is the capital of France? Answer briefly."
        print(generate_text(model, tokenizer, prompt, args.max_new_tokens))

    print("\nГотово.")


if __name__ == "__main__":
    main()
