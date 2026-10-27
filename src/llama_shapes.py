"""Размеры линейных слоёв Llama-3.2-1B-Instruct (unsloth).

Источник: https://huggingface.co/unsloth/Llama-3.2-1B-Instruct
  hidden_size=2048, intermediate_size=8192,
  num_attention_heads=32, num_key_value_heads=8, head_dim=64,
  vocab_size=128256, tie_word_embeddings=True.
"""

from __future__ import annotations

# Конфиг модели
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 8192
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 64
VOCAB_SIZE = 128256
NUM_HIDDEN_LAYERS = 16

Q_OUT = NUM_ATTENTION_HEADS * HEAD_DIM   # 2048
KV_OUT = NUM_KEY_VALUE_HEADS * HEAD_DIM  # 512

# (имя, out_features N, in_features K) — форма weight nn.Linear: (N, K)
# forward: X (M, K) @ W^T -> (M, N)
LLAMA_LINEAR_SHAPES: tuple[tuple[str, int, int], ...] = (
    ("q_proj", Q_OUT, HIDDEN_SIZE),                 # (2048, 2048)
    ("k_proj", KV_OUT, HIDDEN_SIZE),                # (512, 2048)
    ("v_proj", KV_OUT, HIDDEN_SIZE),                # (512, 2048)
    ("o_proj", HIDDEN_SIZE, Q_OUT),                 # (2048, 2048)
    ("gate_proj", INTERMEDIATE_SIZE, HIDDEN_SIZE),  # (8192, 2048)
    ("up_proj", INTERMEDIATE_SIZE, HIDDEN_SIZE),    # (8192, 2048)
    ("down_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE),  # (2048, 8192)
)

# lm_head tied с embed — большой; для бенча матмула полезен отдельно
LM_HEAD_SHAPE = ("lm_head", VOCAB_SIZE, HIDDEN_SIZE)  # (128256, 2048)

# Число токенов (строк X) по заданию
ACTIVATION_ROWS = (128, 512, 2048)

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
