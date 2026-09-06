from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiTaskConfig:
    """Configuration for a shared-trunk multitask classifier."""

    input_dim: int
    task_sizes: dict[str, int]
    hidden_dims: tuple[int, ...] = (1024, 512, 256)
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not self.task_sizes or any(size <= 1 for size in self.task_sizes.values()):
            raise ValueError("task_sizes must contain at least one task with >=2 classes")
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class CVAEConfig:
    """Configuration for the conditional protein-sequence VAE."""

    esm_dim: int = 640
    latent_dim: int = 128
    time_embedding_dim: int = 64
    num_time_tokens: int = 64
    vocab_size: int = 32
    decoder_dim: int = 256
    decoder_layers: int = 4
    decoder_heads: int = 8
    max_sequence_length: int = 4096
    encoder_hidden_dim: int = 512
    dropout: float = 0.1
    padding_index: int = 0

    def __post_init__(self) -> None:
        integer_fields = {
            "esm_dim": self.esm_dim,
            "latent_dim": self.latent_dim,
            "time_embedding_dim": self.time_embedding_dim,
            "num_time_tokens": self.num_time_tokens,
            "vocab_size": self.vocab_size,
            "decoder_dim": self.decoder_dim,
            "decoder_layers": self.decoder_layers,
            "decoder_heads": self.decoder_heads,
            "max_sequence_length": self.max_sequence_length,
            "encoder_hidden_dim": self.encoder_hidden_dim,
        }
        if any(value <= 0 for value in integer_fields.values()):
            raise ValueError("all dimensional configuration values must be positive")
        if self.decoder_dim % self.decoder_heads:
            raise ValueError("decoder_dim must be divisible by decoder_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.padding_index < self.vocab_size:
            raise ValueError("padding_index must be a valid token id")
