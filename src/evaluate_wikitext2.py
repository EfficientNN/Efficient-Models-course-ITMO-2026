"""Оценка скорости и перплексии Llama bf16/int4 на WikiText-2."""

import argparse
import hashlib
import importlib.metadata
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation_metrics import benchmark_forward, evaluate_perplexity
from evaluation_utils import EvalConfig, save_results
from linear import replace_linear_with_quantized
from quantize_llama import bytes_of_params, count_linears


def create_report(config: EvalConfig) -> dict:
    major, minor = torch.cuda.get_device_capability()
    return {
        "config": asdict(config),
        "status": "running",
        "stage": "load",
        "metadata": {
            "gpu": torch.cuda.get_device_name(0),
            "capability": f"sm_{major}{minor}",
            "cuda": torch.version.cuda,
            "int4_dot_path": "fp32" if major < 8 else "bf16",
            "attention": "sdpa",
            "seed": 0,
            "allow_tf32": False,
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            },
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
        },
    }


def load_tokens(config: EvalConfig, tokenizer) -> tuple[torch.Tensor, dict]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    tokens = tokenizer(
        "\n\n".join(dataset["text"]),
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    ).input_ids
    full_tokens = tokens.shape[1]
    if config.max_tokens is not None:
        tokens = tokens[:, : config.max_tokens].contiguous()
    if tokens.shape[1] < 2048 * config.repeats:
        raise ValueError(
            "Для замеров нужно минимум 2048 * repeats токенов. Уменьшите repeats"
        )
    metadata = {
        "dataset": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": "test",
        "fingerprint": dataset._fingerprint,
        "separator": "\n\n",
        "add_special_tokens": False,
        "full_tokens": full_tokens,
        "evaluated_tokens": tokens.shape[1],
        "partial": tokens.shape[1] < full_tokens,
        "token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
    }
    return tokens, metadata


def prepare_variant(model: torch.nn.Module, config: EvalConfig, variant: str) -> dict:
    replaced = 0
    if variant == "int4":
        before = count_linears(model)[0]
        replaced = replace_linear_with_quantized(
            model,
            pack_bits=config.pack_bits,
            per_row=True,
            skip_names={"lm_head"} if config.skip_lm_head else set(),
        )
        remaining, quantized = count_linears(model)
        expected = before - int(config.skip_lm_head)
        if replaced != expected or quantized != expected:
            raise AssertionError(
                f"Ожидалось {expected} замен, получено {replaced}/{quantized}"
            )
        torch.cuda.empty_cache()
    else:
        remaining, quantized = count_linears(model)
    return {
        "replaced": replaced,
        "remaining_linear": remaining,
        "quantized_linear": quantized,
        "parameter_buffer_bytes": bytes_of_params(model),
    }


def evaluate_variant(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    config: EvalConfig,
    report: dict,
    variant: str,
) -> None:
    report["stage"] = variant + ":prepare"
    report[variant] = prepare_variant(model, config, variant)
    save_results(report, config.output)
    print(
        f"\nОценка {variant}, Linear int4: {report[variant]['quantized_linear']}",
        flush=True,
    )
    report["stage"] = variant + ":forward"
    report[variant]["forward"] = benchmark_forward(
        model, tokens, config.warmup, config.repeats
    )
    save_results(report, config.output)
    report["stage"] = variant + ":perplexity"
    report[variant]["perplexity"] = evaluate_perplexity(
        model, tokens, config.context_length, config.stride
    )
    print(f"{variant}: PPL={report[variant]['perplexity']['ppl']:.4f}", flush=True)
    save_results(report, config.output)


def run_evaluation(config: EvalConfig) -> dict:
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("Для оценки нужна CUDA. Включите GPU в настройках ноутбука")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = create_report(config)
    save_results(report, config.output)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model, revision=config.revision
        )
        tokens, report["data"] = load_tokens(config, tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            revision=config.revision,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )
        model.eval()
        model.requires_grad_(False)
        report["metadata"]["resolved_model_revision"] = getattr(
            model.config, "_commit_hash", None
        )
        if config.context_length > model.config.max_position_embeddings:
            raise ValueError("context_length превышает контекст модели")
        for variant in ("bf16", "int4"):
            evaluate_variant(model, tokens, config, report, variant)
        report["status"], report["stage"] = "complete", "complete"
        save_results(report, config.output)
        print(f"Результаты: {Path(config.output).resolve()}")
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_results(report, config.output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=EvalConfig.model)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--pack-bits", type=int, choices=(8, 32), default=8)
    parser.add_argument("--skip-lm-head", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="results/wikitext2")
    run_evaluation(EvalConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
