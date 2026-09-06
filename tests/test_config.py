import pytest

from fluformer import CVAEConfig, MultiTaskConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 0, "task_sizes": {"x": 2}},
        {"input_dim": -1, "task_sizes": {"x": 2}},
        {"input_dim": 4, "task_sizes": {}},
        {"input_dim": 4, "task_sizes": {"x": 1}},
        {"input_dim": 4, "task_sizes": {"x": 0}},
        {"input_dim": 4, "task_sizes": {"x": 2}, "hidden_dims": ()},
        {"input_dim": 4, "task_sizes": {"x": 2}, "hidden_dims": (8, 0)},
        {"input_dim": 4, "task_sizes": {"x": 2}, "dropout": -0.1},
        {"input_dim": 4, "task_sizes": {"x": 2}, "dropout": 1.0},
    ],
)
def test_invalid_multitask_configs_fail_early(kwargs):
    with pytest.raises(ValueError):
        MultiTaskConfig(**kwargs)


def test_valid_multitask_config_preserves_values():
    config = MultiTaskConfig(
        input_dim=12,
        task_sizes={"host": 3, "subtype": 4},
        hidden_dims=(16, 8),
        dropout=0.25,
    )

    assert config.input_dim == 12
    assert config.task_sizes == {"host": 3, "subtype": 4}
    assert config.hidden_dims == (16, 8)
    assert config.dropout == 0.25


@pytest.mark.parametrize(
    "field",
    [
        "esm_dim",
        "latent_dim",
        "time_embedding_dim",
        "num_time_tokens",
        "vocab_size",
        "decoder_dim",
        "decoder_layers",
        "decoder_heads",
        "max_sequence_length",
        "encoder_hidden_dim",
    ],
)
def test_cvae_dimensional_fields_must_be_positive(field):
    with pytest.raises(ValueError, match="positive"):
        CVAEConfig(**{field: 0})


def test_cvae_decoder_dimension_must_be_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        CVAEConfig(decoder_dim=10, decoder_heads=4)


@pytest.mark.parametrize("dropout", [-0.01, 1.0, 1.5])
def test_cvae_dropout_range_is_validated(dropout):
    with pytest.raises(ValueError, match="dropout"):
        CVAEConfig(dropout=dropout)


@pytest.mark.parametrize("padding_index", [-1, 32, 100])
def test_cvae_padding_index_is_validated(padding_index):
    with pytest.raises(ValueError, match="padding_index"):
        CVAEConfig(vocab_size=32, padding_index=padding_index)


def test_valid_cvae_config_preserves_values():
    config = CVAEConfig(
        esm_dim=16,
        latent_dim=8,
        decoder_dim=32,
        decoder_heads=4,
        dropout=0.0,
        padding_index=1,
        vocab_size=20,
    )

    assert config.esm_dim == 16
    assert config.latent_dim == 8
    assert config.decoder_dim == 32
    assert config.padding_index == 1
