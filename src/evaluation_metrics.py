import math
import statistics
import time

import torch
import torch.nn.functional as F

from evaluation_utils import iter_windows


def chunked_nll(
    logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 128
) -> tuple[float, int]:
    """Сумма NLL и число немаскированных целей после causal shift без полной fp32-копии logits."""

    total = torch.zeros((), dtype=torch.float64, device=logits.device)
    targets = labels[:, 1:]
    count = int((targets != -100).sum().item())
    for begin in range(0, targets.shape[1], chunk_size):
        part = targets[:, begin : begin + chunk_size]
        if not bool((part != -100).any()):
            continue
        # Последний logit не имеет целевой метки после causal shift.
        scores = logits[:, begin : begin + part.shape[1], :].float()
        loss = F.cross_entropy(
            scores.reshape(-1, scores.shape[-1]),
            part.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total += loss.double()
    return total.item(), count


def evaluate_perplexity(
    model: torch.nn.Module, tokens: torch.Tensor, context_length: int, stride: int
) -> dict:
    """PPL = exp(NLL / число целей). Время в секундах включает перенос окон и расчет loss."""

    total_nll, total_targets = 0.0, 0
    windows = list(iter_windows(tokens.shape[1], context_length, stride))
    started = time.perf_counter()
    with torch.inference_mode():
        for index, (begin, end, first_target) in enumerate(windows):
            inputs = tokens[:, begin:end].to("cuda")
            labels = inputs.clone()
            # Перекрытие дает контекст, но его цели уже учтены в предыдущих окнах.
            labels[:, : first_target - begin] = -100
            outputs = model(input_ids=inputs, use_cache=False, return_dict=True)
            nll, count = chunked_nll(outputs.logits, labels)
            if not math.isfinite(nll):
                raise ValueError(f"Неконечный NLL в окне {index}")
            total_nll += nll
            total_targets += count
            del outputs, inputs, labels
            if index % 10 == 0 or index + 1 == len(windows):
                print(
                    f"PPL: окно {index + 1}/{len(windows)}, целей {total_targets}",
                    flush=True,
                )
    if total_targets != tokens.shape[1] - 1:
        raise AssertionError("Целевые токены пропущены или посчитаны повторно")
    ppl = math.exp(total_nll / total_targets)
    if not math.isfinite(ppl):
        raise ValueError("Получена неконечная перплексия")
    return {
        "ppl": ppl,
        "total_nll": total_nll,
        "targets": total_targets,
        "windows": len(windows),
        "evaluation_seconds": time.perf_counter() - started,
    }


def benchmark_forward(
    model: torch.nn.Module, tokens: torch.Tensor, warmup: int, repeats: int
) -> list[dict]:
    """Медиана и отдельные времена forward в мс, пропускная способность во входных токенах/с."""

    rows = []
    with torch.inference_mode():
        for length in (128, 512, 2048):
            inputs = [
                tokens[:, i * length : (i + 1) * length].to("cuda")
                for i in range(repeats)
            ]
            for _ in range(warmup):
                outputs = model(input_ids=inputs[0], use_cache=False, return_dict=True)
                del outputs
            torch.cuda.synchronize()
            samples = []
            for item in inputs:
                # CUDA асинхронна: без синхронизации таймер не ограничивает время полного forward.
                torch.cuda.synchronize()
                started = time.perf_counter()
                outputs = model(input_ids=item, use_cache=False, return_dict=True)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
                del outputs
            if not all(math.isfinite(ms) and ms > 0 for ms in samples):
                raise ValueError("Некорректное время forward")
            row = {
                "length": length,
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
                "tokens_per_second": length * repeats * 1000 / sum(samples),
            }
            rows.append(row)
            print(
                f"Forward {length}: {row['median_ms']:.3f} мс, {row['tokens_per_second']:.1f} токенов/с",
                flush=True,
            )
            del inputs
    return rows
