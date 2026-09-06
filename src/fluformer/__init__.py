"""Fluformer: reusable temporal protein-modeling components."""

from .config import CVAEConfig, MultiTaskConfig
from .models.cvae import ConditionalProteinVAE
from .models.multitask import MultiTaskMLP

__all__ = ["CVAEConfig", "MultiTaskConfig", "ConditionalProteinVAE", "MultiTaskMLP"]
__version__ = "0.1.0"
