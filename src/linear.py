"""Квантизованный линейный слой на Triton-ядрах: y = X @ W4^T (+ bias)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from matmul import matmul_bf16_x_int4
from quantize import quantize_fp16_to_int4
from utils import PACK_BITS_INT8


class QuantizedLinear(nn.Module):
    """Замена nn.Linear: веса хранятся как упакованный int4 + scale.

    forward: вход приводится к bf16, считается matmul_bf16_x_int4, выход
    возвращается в исходном dtype (с опциональным bias).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        w_packed: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        pack_bits: int = PACK_BITS_INT8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pack_bits = pack_bits
        # Буферы (не обучаемые): упакованные веса и scale
        self.register_buffer("w_packed", w_packed)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        pack_bits: int = PACK_BITS_INT8,
        per_row: bool = True,
    ) -> "QuantizedLinear":
        """Строит QuantizedLinear из nn.Linear (квантизация weight на CUDA)."""
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"ожидался nn.Linear, получено {type(linear)}")
        W = linear.weight.detach()
        if W.device.type != "cuda":
            W = W.cuda()
        # Ядро квантизации принимает fp16
        W_fp16 = W.to(torch.float16).contiguous()
        packed, scale, shape = quantize_fp16_to_int4(W_fp16, pack_bits=pack_bits, per_row=per_row)
        bias = None if linear.bias is None else linear.bias.detach().to(packed.device).contiguous()
        out_features, in_features = shape
        return cls(
            in_features,
            out_features,
            w_packed=packed,
            scale=scale,
            bias=bias,
            pack_bits=pack_bits,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_shape = x.shape
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"последняя ось входа {x.shape[-1]} != in_features={self.in_features}"
            )
        x2d = x.reshape(-1, self.in_features)
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        if not x2d.is_cuda:
            raise ValueError("QuantizedLinear требует CUDA-тензор")

        out = matmul_bf16_x_int4(
            x2d.contiguous(),
            self.w_packed,
            self.scale,
            self.weight_shape,
            pack_bits=self.pack_bits,
        )
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        out = out.to(orig_dtype).reshape(*orig_shape[:-1], self.out_features)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"pack_bits={self.pack_bits}, bias={self.bias is not None}"
        )


def replace_linear_with_quantized(
    module: nn.Module,
    *,
    pack_bits: int = PACK_BITS_INT8,
    per_row: bool = True,
    skip_names: Optional[set[str]] = None,
) -> int:
    """Рекурсивно заменяет дочерние nn.Linear на QuantizedLinear.

    Возвращает число заменённых слоёв. Имена в skip_names (суффикс/полное)
    пропускаются (например {'lm_head'}).
    """
    skip_names = skip_names or set()
    replaced = 0
    for name, child in list(module.named_children()):
        full_skip = name in skip_names
        if isinstance(child, nn.Linear) and not full_skip:
            setattr(
                module,
                name,
                QuantizedLinear.from_linear(child, pack_bits=pack_bits, per_row=per_row),
            )
            replaced += 1
        else:
            replaced += replace_linear_with_quantized(
                child, pack_bits=pack_bits, per_row=per_row, skip_names=skip_names
            )
    return replaced


@torch.no_grad()
def compare_linear_outputs(
    fp_linear: nn.Linear,
    q_linear: QuantizedLinear,
    x: torch.Tensor,
) -> dict:
    """Сравнивает выход fp-слоя и квантизованного (для смоук-теста)."""
    y_fp = F.linear(x.to(fp_linear.weight.dtype), fp_linear.weight, fp_linear.bias)
    y_q = q_linear(x)
    diff = (y_fp.float() - y_q.float()).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_fro": (diff.norm() / y_fp.float().norm().clamp_min(1e-8)).item(),
    }
