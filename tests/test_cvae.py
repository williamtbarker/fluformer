import pytest
import torch

from fluformer import ConditionalProteinVAE, CVAEConfig
from fluformer.models.cvae import ConditionalEncoder, cvae_loss, sample_sequence


def tiny_config() -> CVAEConfig:
    return CVAEConfig(
        esm_dim=8,
        latent_dim=4,
        time_embedding_dim=4,
        num_time_tokens=6,
        vocab_size=12,
        decoder_dim=16,
        decoder_layers=1,
        decoder_heads=4,
        max_sequence_length=20,
        encoder_hidden_dim=16,
        dropout=0.0,
    )


def tiny_model() -> ConditionalProteinVAE:
    return ConditionalProteinVAE(tiny_config())


def test_cvae_forward_loss_and_gradients():
    torch.manual_seed(2)
    model = tiny_model()
    batch = 3

    token_embeddings = torch.randn(batch, 5, 8)
    global_embedding = torch.randn(batch, 8)
    time_index = torch.tensor([0, 1, 2])
    decoder_input = torch.randint(1, 12, (batch, 7))
    target = torch.randint(1, 12, (batch, 7))

    logits, mu, logvar = model(
        token_embeddings,
        global_embedding,
        time_index,
        decoder_input,
    )

    assert logits.shape == (batch, 7, 12)
    assert mu.shape == logvar.shape == (batch, 4)

    loss, parts = cvae_loss(logits, target, mu, logvar)
    loss.backward()

    assert torch.isfinite(loss)
    assert parts["reconstruction"] > 0
    assert torch.isfinite(parts["kl"])

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None for gradient in gradients)


def test_encoder_accepts_valid_mask():
    encoder = ConditionalEncoder(tiny_config())

    token_embeddings = torch.randn(2, 5, 8)
    global_embedding = torch.randn(2, 8)
    time_index = torch.tensor([0, 1])
    mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
        ]
    )

    mu, logvar = encoder(
        token_embeddings,
        global_embedding,
        time_index,
        mask,
    )

    assert mu.shape == (2, 4)
    assert logvar.shape == (2, 4)


@pytest.mark.parametrize(
    ("token_embeddings", "global_embedding", "time_index", "mask", "message"),
    [
        (
            torch.randn(2, 5),
            torch.randn(2, 8),
            torch.tensor([0, 1]),
            None,
            "token_embeddings",
        ),
        (
            torch.randn(2, 5, 7),
            torch.randn(2, 8),
            torch.tensor([0, 1]),
            None,
            "token_embeddings",
        ),
        (
            torch.randn(2, 5, 8),
            torch.randn(2, 7),
            torch.tensor([0, 1]),
            None,
            "global_embedding",
        ),
        (
            torch.randn(2, 5, 8),
            torch.randn(2, 8),
            torch.tensor([[0], [1]]),
            None,
            "time_index",
        ),
        (
            torch.randn(2, 5, 8),
            torch.randn(2, 8),
            torch.tensor([0, 1]),
            torch.ones(2, 4, dtype=torch.bool),
            "token_mask",
        ),
    ],
)
def test_encoder_rejects_invalid_shapes(
    token_embeddings,
    global_embedding,
    time_index,
    mask,
    message,
):
    encoder = ConditionalEncoder(tiny_config())

    with pytest.raises(ValueError, match=message):
        encoder(
            token_embeddings,
            global_embedding,
            time_index,
            mask,
        )


def test_decoder_rejects_non_matrix_tokens():
    model = tiny_model()

    with pytest.raises(ValueError, match="tokens"):
        model.decoder(
            torch.tensor([1, 2, 3]),
            torch.randn(1, 4),
            torch.tensor([0]),
        )


def test_decoder_rejects_sequence_longer_than_configuration():
    model = tiny_model()

    with pytest.raises(ValueError, match="max_sequence_length"):
        model.decoder(
            torch.ones((1, 21), dtype=torch.long),
            torch.randn(1, 4),
            torch.tensor([0]),
        )


def test_decoder_rejects_invalid_latent_shape():
    model = tiny_model()

    with pytest.raises(ValueError, match="latent"):
        model.decoder(
            torch.ones((2, 3), dtype=torch.long),
            torch.randn(2, 3),
            torch.tensor([0, 1]),
        )


def test_decoder_rejects_invalid_time_shape():
    model = tiny_model()

    with pytest.raises(ValueError, match="time_index"):
        model.decoder(
            torch.ones((2, 3), dtype=torch.long),
            torch.randn(2, 4),
            torch.tensor([[0], [1]]),
        )


def test_cvae_loss_rejects_mismatched_sequence_dimensions():
    logits = torch.randn(2, 4, 12)
    target = torch.randint(0, 12, (2, 3))
    mu = torch.zeros(2, 4)
    logvar = torch.zeros(2, 4)

    with pytest.raises(ValueError, match="sequence dimensions"):
        cvae_loss(logits, target, mu, logvar)


def test_cvae_loss_ignores_padding_tokens():
    torch.manual_seed(5)

    logits = torch.randn(2, 4, 12)
    target = torch.tensor(
        [
            [1, 2, 0, 0],
            [3, 4, 5, 0],
        ]
    )
    mu = torch.zeros(2, 4)
    logvar = torch.zeros(2, 4)

    loss, parts = cvae_loss(
        logits,
        target,
        mu,
        logvar,
        padding_index=0,
        beta=0.25,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(parts["reconstruction"])
    assert torch.isfinite(parts["kl"])
    assert parts["kl"].item() == pytest.approx(0.0)


def test_sampling_is_shape_safe_with_top_k():
    torch.manual_seed(3)

    model = tiny_model().eval()
    start = torch.ones((2, 1), dtype=torch.long)
    time_index = torch.tensor([1, 2])

    sampled = sample_sequence(
        model,
        time_index=time_index,
        start_tokens=start,
        max_length=6,
        top_k=4,
    )

    assert sampled.shape == (2, 6)
    assert sampled.min() >= 0
    assert sampled.max() < model.config.vocab_size


def test_sampling_without_top_k_uses_full_vocabulary():
    model = tiny_model().eval()
    start = torch.ones((1, 1), dtype=torch.long)
    time_index = torch.tensor([1])
    latent = torch.zeros((1, model.config.latent_dim))
    generator = torch.Generator().manual_seed(11)

    sampled = sample_sequence(
        model,
        time_index=time_index,
        start_tokens=start,
        max_length=4,
        latent=latent,
        generator=generator,
    )

    assert sampled.shape == (1, 4)


def test_sampling_is_reproducible_with_fixed_latent_and_generator():
    torch.manual_seed(7)

    model = tiny_model().eval()
    start = torch.ones((2, 1), dtype=torch.long)
    time_index = torch.tensor([1, 2])
    latent = torch.zeros((2, model.config.latent_dim))

    first = sample_sequence(
        model,
        time_index=time_index,
        start_tokens=start,
        max_length=7,
        top_k=4,
        latent=latent,
        generator=torch.Generator().manual_seed(123),
    )

    second = sample_sequence(
        model,
        time_index=time_index,
        start_tokens=start,
        max_length=7,
        top_k=4,
        latent=latent,
        generator=torch.Generator().manual_seed(123),
    )

    assert torch.equal(first, second)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_sampling_requires_positive_temperature(temperature):
    model = tiny_model().eval()

    with pytest.raises(ValueError, match="temperature"):
        sample_sequence(
            model,
            time_index=torch.tensor([0]),
            start_tokens=torch.ones((1, 1), dtype=torch.long),
            max_length=2,
            temperature=temperature,
        )


@pytest.mark.parametrize(
    "start_tokens",
    [
        torch.tensor([1, 2]),
        torch.empty((1, 0), dtype=torch.long),
    ],
)
def test_sampling_validates_start_tokens(start_tokens):
    model = tiny_model().eval()

    with pytest.raises(ValueError, match="start_tokens"):
        sample_sequence(
            model,
            time_index=torch.tensor([0]),
            start_tokens=start_tokens,
            max_length=2,
        )


@pytest.mark.parametrize("max_length", [0, 21])
def test_sampling_validates_max_length(max_length):
    model = tiny_model().eval()

    with pytest.raises(ValueError, match="max_length"):
        sample_sequence(
            model,
            time_index=torch.tensor([0]),
            start_tokens=torch.ones((1, 1), dtype=torch.long),
            max_length=max_length,
        )


@pytest.mark.parametrize("top_k", [0, 13])
def test_sampling_validates_top_k(top_k):
    model = tiny_model().eval()

    with pytest.raises(ValueError, match="top_k"):
        sample_sequence(
            model,
            time_index=torch.tensor([0]),
            start_tokens=torch.ones((1, 1), dtype=torch.long),
            max_length=2,
            top_k=top_k,
        )


def test_decoder_is_causal_with_respect_to_future_tokens():
    torch.manual_seed(4)

    model = tiny_model().eval()
    latent = torch.randn(1, model.config.latent_dim)
    time_index = torch.tensor([1])

    first = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 9, 10, 11]])

    with torch.no_grad():
        logits_a = model.decoder(first, latent, time_index)
        logits_b = model.decoder(changed_future, latent, time_index)

    assert torch.allclose(
        logits_a[:, 0, :],
        logits_b[:, 0, :],
        atol=1e-6,
    )


def test_state_dict_roundtrip_preserves_decoder_output():
    torch.manual_seed(9)

    original = tiny_model().eval()
    restored = tiny_model().eval()
    restored.load_state_dict(original.state_dict())

    tokens = torch.tensor([[1, 2, 3]])
    latent = torch.randn(1, original.config.latent_dim)
    time_index = torch.tensor([2])

    with torch.no_grad():
        first = original.decoder(tokens, latent, time_index)
        second = restored.decoder(tokens, latent, time_index)

    assert torch.allclose(first, second)
