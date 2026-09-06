import torch

from fluformer import CVAEConfig
from fluformer.models.cvae import ConditionalEncoder


def tiny_encoder() -> ConditionalEncoder:
    config = CVAEConfig(
        esm_dim=2,
        latent_dim=2,
        time_embedding_dim=2,
        num_time_tokens=2,
        vocab_size=5,
        decoder_dim=4,
        decoder_layers=1,
        decoder_heads=2,
        max_sequence_length=8,
        encoder_hidden_dim=4,
        dropout=0.0,
    )
    return ConditionalEncoder(config)


def capture_encoder_input(
    encoder: ConditionalEncoder,
    token_embeddings: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    captured = {}

    def capture(module, args):
        captured["x"] = args[0].detach().clone()

    handle = encoder.network[0].register_forward_pre_hook(capture)

    try:
        encoder(
            token_embeddings,
            torch.zeros(token_embeddings.shape[0], 2),
            torch.zeros(token_embeddings.shape[0], dtype=torch.long),
            mask,
        )
    finally:
        handle.remove()

    return captured["x"]


def test_masked_pooling_ignores_padding_tokens():
    encoder = tiny_encoder()

    token_embeddings = torch.tensor(
        [[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]]
    )
    mask = torch.tensor([[True, True, False]])

    network_input = capture_encoder_input(
        encoder,
        token_embeddings,
        mask,
    )

    assert torch.allclose(
        network_input[0, :2],
        torch.tensor([2.0, 4.0]),
    )


def test_unmasked_pooling_uses_mean():
    encoder = tiny_encoder()

    token_embeddings = torch.tensor(
        [[[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]]
    )

    network_input = capture_encoder_input(
        encoder,
        token_embeddings,
        None,
    )

    assert torch.allclose(
        network_input[0, :2],
        torch.tensor([3.0, 5.0]),
    )


def test_all_false_mask_produces_finite_zero_pool():
    encoder = tiny_encoder()

    token_embeddings = torch.tensor(
        [[[10.0, 20.0], [30.0, 40.0]]]
    )
    mask = torch.tensor([[False, False]])

    network_input = capture_encoder_input(
        encoder,
        token_embeddings,
        mask,
    )

    assert torch.allclose(
        network_input[0, :2],
        torch.zeros(2),
    )
    assert torch.isfinite(network_input).all()
