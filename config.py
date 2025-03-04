"""
Configuration module for the UMAF Capability Extractor.

This module defines the configuration classes and options for the UMAF framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Type

from umaf.metrics import SimilarityMetric, CosineSimilarity


@dataclass
class CapabilityExtractorConfig:
    """
    Configuration class for the Capability Extractor.
    
    Attributes:
        input_dim (int): Input dimension size (default: 768).
        output_dim (int): Output dimension size for the capability fingerprint (default: 512).
        num_layers (int): Number of transformer encoder layers (default: 4).
        num_heads (int): Number of attention heads in transformer (default: 8).
        hidden_dim (int): Hidden dimension size in transformer (default: 2048).
        dropout (float): Dropout rate (default: 0.1).
        similarity_metric (SimilarityMetric): Metric for computing similarity (default: CosineSimilarity).
        positive_pair_threshold (float): Threshold for positive pair definition (default: 0.85).
        temperature (float): Temperature parameter for contrastive learning (default: 0.07).
        max_length (int): Maximum sequence length (default: 128).
        normalization (str): Type of normalization to apply ('mean', 'l2', 'layer', or None) (default: 'layer').
        adaptive_pooling (bool): Whether to use adaptive pooling (default: True).
        task_weight (float): Weight for task performance in composite similarity score (default: 0.6).
        rep_weight (float): Weight for representational similarity in composite score (default: 0.4).
        additional_params (Dict[str, Any]): Additional parameters for extensibility.
    """
    input_dim: int = 768
    output_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 2048
    dropout: float = 0.1
    similarity_metric: SimilarityMetric = field(default_factory=CosineSimilarity)
    positive_pair_threshold: float = 0.85
    temperature: float = 0.07
    max_length: int = 128
    normalization: Optional[str] = 'layer'  # 'mean', 'l2', 'layer', or None
    adaptive_pooling: bool = True
    task_weight: float = 0.6
    rep_weight: float = 0.4
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        if not 0 <= self.positive_pair_threshold <= 1:
            raise ValueError("positive_pair_threshold must be between 0 and 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.normalization not in [None, 'mean', 'l2', 'layer']:
            raise ValueError("normalization must be None, 'mean', 'l2', or 'layer'")
        if not 0 <= self.task_weight <= 1:
            raise ValueError("task_weight must be between 0 and 1")
        if not 0 <= self.rep_weight <= 1:
            raise ValueError("rep_weight must be between 0 and 1")
        if abs(self.task_weight + self.rep_weight - 1.0) > 1e-6:
            raise ValueError("task_weight and rep_weight must sum to 1.0")