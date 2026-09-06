"""Tiny data-free smoke example for the public API."""

import torch

from fluformer import ConditionalProteinVAE, CVAEConfig, MultiTaskConfig, MultiTaskMLP
from fluformer.models.cvae import cvae_loss
from fluformer.models.multitask import multitask_loss


def main() -> None:
    torch.manual_seed(7)

    baseline = MultiTaskMLP(
        MultiTaskConfig(input_dim=32, task_sizes={"host": 3, "subtype": 4}, hidden_dims=(64, 32))
    )
    features = torch.randn(8, 32)
    labels = {"host": torch.randint(0, 3, (8,)), "subtype": torch.randint(0, 4, (8,))}
    baseline_loss, _ = multitask_loss(baseline(features), labels)
    print(f"multitask loss: {baseline_loss.item():.3f}")

    config = CVAEConfig(
        esm_dim=16,
        latent_dim=8,
        time_embedding_dim=8,
        num_time_tokens=10,
        vocab_size=24,
        decoder_dim=32,
        decoder_layers=1,
        decoder_heads=4,
        max_sequence_length=64,
        encoder_hidden_dim=32,
    )
    cvae = ConditionalProteinVAE(config)
    token_embeddings = torch.randn(4, 12, 16)
    global_embedding = torch.randn(4, 16)
    time = torch.randint(0, 10, (4,))
    decoder_input = torch.randint(1, 24, (4, 10))
    target = torch.randint(1, 24, (4, 10))
    logits, mu, logvar = cvae(token_embeddings, global_embedding, time, decoder_input)
    loss, _ = cvae_loss(logits, target, mu, logvar)
    print(f"CVAE loss: {loss.item():.3f}")


if __name__ == "__main__":
    main()
