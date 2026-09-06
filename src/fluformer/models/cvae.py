from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import CVAEConfig


class ConditionalEncoder(nn.Module):
    """Fuse token-level, global, and temporal embeddings into a latent posterior."""

    def __init__(self, config: CVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.time_embedding = nn.Embedding(
            config.num_time_tokens,
            config.time_embedding_dim,
        )

        input_dim = (
            2 * config.esm_dim
            + config.time_embedding_dim
        )

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                config.encoder_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.encoder_hidden_dim,
                2 * config.latent_dim,
            ),
        )

    def forward(
        self,
        token_embeddings: Tensor,
        global_embedding: Tensor,
        time_index: Tensor,
        token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if (
            token_embeddings.ndim != 3
            or token_embeddings.shape[-1]
            != self.config.esm_dim
        ):
            raise ValueError(
                "token_embeddings must have shape "
                "[batch, tokens, esm_dim]"
            )

        if global_embedding.shape != (
            token_embeddings.shape[0],
            self.config.esm_dim,
        ):
            raise ValueError(
                "global_embedding must have shape "
                "[batch, esm_dim]"
            )

        if time_index.shape != (
            token_embeddings.shape[0],
        ):
            raise ValueError(
                "time_index must have shape [batch]"
            )

        if token_mask is None:
            pooled = token_embeddings.mean(dim=1)

        else:
            if (
                token_mask.shape
                != token_embeddings.shape[:2]
            ):
                raise ValueError(
                    "token_mask must have shape "
                    "[batch, tokens]"
                )

            weights = (
                token_mask
                .to(token_embeddings.dtype)
                .unsqueeze(-1)
            )

            denom = (
                weights
                .sum(dim=1)
                .clamp_min(1.0)
            )

            pooled = (
                (token_embeddings * weights)
                .sum(dim=1)
                / denom
            )

        time_embedding = self.time_embedding(
            time_index
        )

        posterior = self.network(
            torch.cat(
                (
                    pooled,
                    global_embedding,
                    time_embedding,
                ),
                dim=-1,
            )
        )

        return posterior.chunk(2, dim=-1)


class ConditionalDecoder(nn.Module):
    """Causal Transformer decoder conditioned on latent state and collection time.

    Conditioning is supplied as a separate one-token memory sequence. Target
    tokens are never exposed through decoder memory.
    """

    def __init__(self, config: CVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.decoder_dim,
            padding_idx=config.padding_index,
        )

        self.position_embedding = nn.Embedding(
            config.max_sequence_length,
            config.decoder_dim,
        )

        self.time_embedding = nn.Embedding(
            config.num_time_tokens,
            config.time_embedding_dim,
        )

        self.latent_projection = nn.Linear(
            config.latent_dim,
            config.decoder_dim,
        )

        self.time_projection = nn.Linear(
            config.time_embedding_dim,
            config.decoder_dim,
        )

        layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.decoder_heads,
            dim_feedforward=4 * config.decoder_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=config.decoder_layers,
        )

        self.output = nn.Linear(
            config.decoder_dim,
            config.vocab_size,
        )

    def _memory(
        self,
        latent: Tensor,
        time_index: Tensor,
    ) -> Tensor:
        condition = (
            self.latent_projection(latent)
            + self.time_projection(
                self.time_embedding(time_index)
            )
        )

        return condition.unsqueeze(1)

    def forward(
        self,
        tokens: Tensor,
        latent: Tensor,
        time_index: Tensor,
    ) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError(
                "tokens must have shape "
                "[batch, sequence]"
            )

        batch, length = tokens.shape

        if (
            length
            > self.config.max_sequence_length
        ):
            raise ValueError(
                "sequence exceeds configured "
                "max_sequence_length"
            )

        if latent.shape != (
            batch,
            self.config.latent_dim,
        ):
            raise ValueError(
                "latent must have shape "
                "[batch, latent_dim]"
            )

        if time_index.shape != (batch,):
            raise ValueError(
                "time_index must have shape [batch]"
            )

        positions = torch.arange(
            length,
            device=tokens.device,
        )

        target = (
            self.token_embedding(tokens)
            + self.position_embedding(
                positions
            ).unsqueeze(0)
        )

        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )

        padding_mask = tokens.eq(
            self.config.padding_index
        )

        decoded = self.decoder(
            tgt=target,
            memory=self._memory(
                latent,
                time_index,
            ),
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        return self.output(decoded)


class ConditionalProteinVAE(nn.Module):
    """Conditional VAE over protein tokens using protein-LM embeddings."""

    def __init__(self, config: CVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ConditionalEncoder(config)
        self.decoder = ConditionalDecoder(config)

    @staticmethod
    def reparameterize(
        mu: Tensor,
        log_variance: Tensor,
    ) -> Tensor:
        std = torch.exp(
            0.5 * log_variance
        )

        return (
            mu
            + torch.randn_like(std) * std
        )

    def encode(
        self,
        token_embeddings: Tensor,
        global_embedding: Tensor,
        time_index: Tensor,
        token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.encoder(
            token_embeddings,
            global_embedding,
            time_index,
            token_mask,
        )

    def forward(
        self,
        token_embeddings: Tensor,
        global_embedding: Tensor,
        time_index: Tensor,
        decoder_input: Tensor,
        token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mu, log_variance = self.encode(
            token_embeddings,
            global_embedding,
            time_index,
            token_mask,
        )

        latent = self.reparameterize(
            mu,
            log_variance,
        )

        return (
            self.decoder(
                decoder_input,
                latent,
                time_index,
            ),
            mu,
            log_variance,
        )


def cvae_loss(
    logits: Tensor,
    target: Tensor,
    mu: Tensor,
    log_variance: Tensor,
    *,
    padding_index: int = 0,
    beta: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Token reconstruction loss plus beta-weighted KL divergence."""

    if logits.shape[:2] != target.shape:
        raise ValueError(
            "logits and target sequence "
            "dimensions do not match"
        )

    reconstruction = F.cross_entropy(
        logits.reshape(
            -1,
            logits.shape[-1],
        ),
        target.reshape(-1),
        ignore_index=padding_index,
    )

    kl = -0.5 * torch.mean(
        1
        + log_variance
        - mu.square()
        - log_variance.exp()
    )

    total = reconstruction + beta * kl

    return total, {
        "reconstruction": reconstruction.detach(),
        "kl": kl.detach(),
    }


@torch.no_grad()
def sample_sequence(
    model: ConditionalProteinVAE,
    *,
    time_index: Tensor,
    start_tokens: Tensor,
    max_length: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    latent: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Autoregressively sample tokens from latent and temporal conditioning."""

    if temperature <= 0:
        raise ValueError(
            "temperature must be positive"
        )

    if (
        start_tokens.ndim != 2
        or start_tokens.shape[1] < 1
    ):
        raise ValueError(
            "start_tokens must have shape "
            "[batch, >=1]"
        )

    if (
        max_length < start_tokens.shape[1]
        or max_length
        > model.config.max_sequence_length
    ):
        raise ValueError(
            "max_length is outside the "
            "valid configured range"
        )

    device = start_tokens.device
    batch = start_tokens.shape[0]

    if latent is None:
        latent = torch.randn(
            batch,
            model.config.latent_dim,
            device=device,
            generator=generator,
        )

    generated = start_tokens

    while generated.shape[1] < max_length:
        next_logits = (
            model.decoder(
                generated,
                latent,
                time_index,
            )[:, -1, :]
            / temperature
        )

        if top_k is not None:
            if (
                not 1
                <= top_k
                <= model.config.vocab_size
            ):
                raise ValueError(
                    "top_k must be between "
                    "1 and vocab_size"
                )

            values, indices = torch.topk(
                next_logits,
                k=top_k,
                dim=-1,
            )

            probabilities = torch.softmax(
                values,
                dim=-1,
            )

            sampled_local = torch.multinomial(
                probabilities,
                1,
                generator=generator,
            )

            next_token = indices.gather(
                1,
                sampled_local,
            )

        else:
            probabilities = torch.softmax(
                next_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                1,
                generator=generator,
            )

        generated = torch.cat(
            (
                generated,
                next_token,
            ),
            dim=1,
        )

    return generated
