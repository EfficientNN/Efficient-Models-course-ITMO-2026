import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalConfig:
    model: str = "unsloth/Llama-3.2-1B-Instruct"
    revision: str = "main"
    context_length: int = 2048
    stride: int = 512
    max_tokens: int | None = None
    pack_bits: int = 8
    skip_lm_head: bool = False
    warmup: int = 3
    repeats: int = 10
    output: str = "results/wikitext2"

    def validate(self) -> None:
        if self.context_length < 2 or not 0 < self.stride < self.context_length:
            raise ValueError("Нужны context_length >= 2 и 0 < stride < context_length")
        if self.max_tokens is not None and self.max_tokens < 2:
            raise ValueError("max_tokens должен быть >= 2")
        if self.pack_bits not in (8, 32) or self.warmup < 1 or self.repeats < 1:
            raise ValueError("pack_bits: 8/32, warmup и repeats: >= 1")


def iter_windows(
    n_tokens: int, context_length: int, stride: int
) -> Iterator[tuple[int, int, int]]:
    """Индексы (начало, конец, первая новая цель) с однократным покрытием целей [1, n_tokens)."""
    if n_tokens < 2 or context_length < 2 or not 0 < stride < context_length:
        raise ValueError("Нужны >= 2 токенов и 0 < stride < context_length")
    previous_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + context_length, n_tokens)
        yield begin, end, max(previous_end, begin + 1)
        previous_end = end
        if end == n_tokens:
            break


def save_results(report: dict, output: str | Path) -> None:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "results.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(directory / "results.json")
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "variant",
            "ppl",
            "ppl_delta",
            "ppl_relative_change",
            "length",
            "median_ms",
            "tokens_per_second",
            "speedup_bf16_over_int4",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        baseline = report.get("bf16", {})
        base_ppl = baseline.get("perplexity", {}).get("ppl")
        base_speed = {
            row["length"]: row["median_ms"] for row in baseline.get("forward", [])
        }
        for variant in ("bf16", "int4"):
            result = report.get(variant, {})
            ppl = result.get("perplexity", {}).get("ppl")
            for row in result.get("forward", [{}]):
                length = row.get("length")
                writer.writerow(
                    {
                        "variant": variant,
                        "ppl": ppl,
                        "length": length,
                        "ppl_delta": (
                            ppl - base_ppl
                            if ppl is not None and base_ppl is not None
                            else None
                        ),
                        "ppl_relative_change": (
                            ppl / base_ppl - 1 if ppl is not None and base_ppl else None
                        ),
                        "median_ms": row.get("median_ms"),
                        "tokens_per_second": row.get("tokens_per_second"),
                        "speedup_bf16_over_int4": (
                            base_speed[length] / row["median_ms"]
                            if variant == "int4" and length in base_speed
                            else None
                        ),
                    }
                )
