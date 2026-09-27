from importlib.metadata import version

import fluformer


def test_public_api_exports_expected_symbols():
    assert fluformer.__version__ == version("fluformer")

    assert set(fluformer.__all__) == {
        "CVAEConfig",
        "MultiTaskConfig",
        "ConditionalProteinVAE",
        "MultiTaskMLP",
    }

    for name in fluformer.__all__:
        assert hasattr(fluformer, name)


def test_public_classes_are_importable():
    from fluformer import (
        ConditionalProteinVAE,
        CVAEConfig,
        MultiTaskConfig,
        MultiTaskMLP,
    )

    assert ConditionalProteinVAE is not None
    assert CVAEConfig is not None
    assert MultiTaskConfig is not None
    assert MultiTaskMLP is not None
