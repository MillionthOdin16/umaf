"""
Capability Extractor module for the UMAF framework.

This module implements the core CapabilityExtractor class that extracts capability fingerprints
from model activations.
"""

from typing import Dict, Any, Optional, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from umaf.config import CapabilityExtractorConfig
from umaf.processor import InputProcessor
from umaf.metrics import SimilarityMetric


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for contextual feature extraction.
    
    Learns complex activation representations through self-attention mechanisms.
    """
    
    def __init__(self, config: CapabilityExtractorConfig):
        """
        Initialize transformer encoder.
        
        Args:
            config (CapabilityExtractorConfig): Configuration for the encoder
        """
        super().__init__()
        
        # Create transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Create transformer encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, sequence_length, input_dim]
            mask (Optional[torch.Tensor]): Attention mask
        
        Returns:
            torch.Tensor: Encoded tensor [batch_size, sequence_length, input_dim]
        """
        return self.encoder(x, src_key_padding_mask=mask)


class ProjectionHead(nn.Module):
    """
    Projection head for mapping extracted features to a fixed-dimensional latent space.
    
    Produces the final capability fingerprint.
    """
    
    def __init__(self, config: CapabilityExtractorConfig):
        """
        Initialize projection head.
        
        Args:
            config (CapabilityExtractorConfig): Configuration for the projection head
        """
        super().__init__()
        
        # Create projection layers
        self.projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(config.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection head.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
        
        Returns:
            torch.Tensor: Projected tensor [batch_size, output_dim]
        """
        x = self.projection(x)
        x = self.layer_norm(x)
        return x


class CapabilityExtractor(nn.Module):
    """
    Main capability extractor module.
    
    Extracts semantically rich, architecture-agnostic model capability fingerprints.
    """
    
    def __init__(
        self,
        config: CapabilityExtractorConfig = CapabilityExtractorConfig(),
        device: Optional[torch.device] = None
    ):
        """
        Initialize capability extractor.
        
        Args:
            config (CapabilityExtractorConfig): Configuration for the extractor
            device (Optional[torch.device]): Device to use for computation
        """
        super().__init__()
        
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input processor
        self.input_processor = InputProcessor(
            max_length=config.max_length,
            normalization=config.normalization,
            device=self.device
        )
        
        # Transformer encoder
        self.encoder = TransformerEncoder(config)
        
        # Projection head
        self.projection_head = ProjectionHead(config)
        
        # Adaptive pooling flag
        self.adaptive_pooling = config.adaptive_pooling
        
        # Move to device
        self.to(self.device)
    
    def forward(self, activations: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the capability extractor.
        
        Args:
            activations (torch.Tensor): Activation tensor [batch_size, sequence_length, input_dim]
            attention_mask (Optional[torch.Tensor]): Attention mask [batch_size, sequence_length]
        
        Returns:
            torch.Tensor: Capability fingerprint [batch_size, output_dim]
        """
        # Validate input dimensions
        if activations.size(-1) != self.config.input_dim:
            raise ValueError(
                f"Expected input dimension {self.config.input_dim}, "
                f"got {activations.size(-1)}"
            )
        
        # Process input
        processed_activations = self.input_processor.process(activations)
        
        # Encode activations
        encoded = self.encoder(processed_activations, mask=attention_mask)
        
        # Pool encoded activations
        if self.adaptive_pooling:
            # Mean pooling over sequence dimension
            pooled = encoded.mean(dim=1)
        else:
            # Use [CLS] token (first token)
            pooled = encoded[:, 0]
        
        # Project to fingerprint space
        fingerprint = self.projection_head(pooled)
        
        return fingerprint
    
    def extract_fingerprint(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract capability fingerprint from a model.
        
        Args:
            model (Any): Pre-trained model
            inputs (Dict[str, torch.Tensor]): Model inputs
            layer_idx (int): Index of the layer to extract activations from (-1 for last layer)
        
        Returns:
            torch.Tensor: Capability fingerprint
        """
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract activations
        with torch.no_grad():
            # Get model outputs
            outputs = model(**inputs)
            
            # Extract activations
            if hasattr(outputs, 'last_hidden_state'):
                activations = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                activations = outputs.hidden_states[layer_idx]
            else:
                raise ValueError(
                    "Model output must have 'last_hidden_state' or 'hidden_states' attribute"
                )
        
        # Extract attention mask if available
        attention_mask = inputs.get('attention_mask', None)
        
        # Extract fingerprint
        return self.forward(activations, attention_mask)
    
    def compute_similarity(
        self,
        fingerprint1: torch.Tensor,
        fingerprint2: torch.Tensor,
        metric: Optional[SimilarityMetric] = None
    ) -> float:
        """
        Compute similarity between two fingerprints.
        
        Args:
            fingerprint1 (torch.Tensor): First fingerprint
            fingerprint2 (torch.Tensor): Second fingerprint
            metric (Optional[SimilarityMetric]): Similarity metric to use
        
        Returns:
            float: Similarity score
        """
        # Use default metric if none provided
        metric = metric or self.config.similarity_metric
        
        # Compute similarity
        return metric.compute(fingerprint1, fingerprint2)
    
    def is_positive_pair(
        self,
        task_performance1: float,
        task_performance2: float,
        fingerprint1: torch.Tensor,
        fingerprint2: torch.Tensor
    ) -> bool:
        """
        Determine if two models form a positive pair based on task performance and fingerprint similarity.
        
        Args:
            task_performance1 (float): Task performance of first model
            task_performance2 (float): Task performance of second model
            fingerprint1 (torch.Tensor): Capability fingerprint of first model
            fingerprint2 (torch.Tensor): Capability fingerprint of second model
        
        Returns:
            bool: True if models form a positive pair, False otherwise
        """
        # Compute task performance similarity
        task_diff = abs(task_performance1 - task_performance2)
        task_sim = 1.0 if task_diff <= 0.05 else 0.0
        
        # Compute representational similarity
        rep_sim = self.compute_similarity(fingerprint1, fingerprint2)
        
        # Compute composite similarity score
        composite_score = (
            self.config.task_weight * task_sim +
            self.config.rep_weight * rep_sim
        )
        
        # Check if score exceeds threshold
        return composite_score > self.config.positive_pair_threshold
    
    def save(self, path: str) -> None:
        """
        Save the capability extractor to a file.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'CapabilityExtractor':
        """
        Load a capability extractor from a file.
        
        Args:
            path (str): Path to load the model from
            device (Optional[torch.device]): Device to load the model to
        
        Returns:
            CapabilityExtractor: Loaded capability extractor
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        extractor = cls(config, device)
        extractor.load_state_dict(checkpoint['state_dict'])
        
        return extractor