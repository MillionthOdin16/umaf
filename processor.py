"""
Input processor module for the UMAF Capability Extractor.

This module handles preprocessing of model activation inputs.
"""

from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputProcessor:
    """
    Processes and normalizes model activation inputs.
    
    Handles diverse model activation inputs, normalizes and preprocesses activation tensors.
    """
    
    def __init__(
        self,
        max_length: int = 128,
        normalization: Optional[str] = 'layer',
        pad_value: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize input processor.
        
        Args:
            max_length (int): Maximum sequence length
            normalization (Optional[str]): Normalization method ('mean', 'l2', 'layer', or None)
            pad_value (float): Value to use for padding
            device (Optional[torch.device]): Device to use for processing
        """
        self.max_length = max_length
        self.normalization = normalization
        self.pad_value = pad_value
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize layer normalization if needed
        if normalization == 'layer':
            self.layer_norm = None  # Will be initialized dynamically based on input dimension
    
    def process(self, activations: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        """
        Process activation tensors.
        
        Args:
            activations (Union[torch.Tensor, Dict[str, Any]]): Model activations
                - If torch.Tensor: Expected shape [batch_size, sequence_length, hidden_size]
                - If Dict: Expected to contain 'last_hidden_state' or 'hidden_states'
        
        Returns:
            torch.Tensor: Processed activations
        """
        # Extract tensor from dictionary if needed
        if isinstance(activations, dict):
            if 'last_hidden_state' in activations:
                activations = activations['last_hidden_state']
            elif 'hidden_states' in activations:
                # Use the last layer's hidden states
                activations = activations['hidden_states'][-1]
            else:
                raise ValueError("Input dictionary must contain 'last_hidden_state' or 'hidden_states'")
        
        # Convert numpy arrays to torch tensors if needed
        if not isinstance(activations, torch.Tensor):
            activations = torch.tensor(activations, device=self.device)
        
        # Move to device
        activations = activations.to(self.device)
        
        # Handle sequence length
        activations = self._handle_sequence_length(activations)
        
        # Apply normalization
        activations = self._normalize(activations)
        
        return activations
    
    def _handle_sequence_length(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Handle sequence length by padding or truncating.
        
        Args:
            activations (torch.Tensor): Activation tensor [batch_size, sequence_length, hidden_size]
        
        Returns:
            torch.Tensor: Tensor with adjusted sequence length
        """
        batch_size, seq_len, hidden_size = activations.shape
        
        if seq_len > self.max_length:
            # Truncate
            return activations[:, :self.max_length, :]
        elif seq_len < self.max_length:
            # Pad
            padding = torch.full(
                (batch_size, self.max_length - seq_len, hidden_size),
                self.pad_value,
                dtype=activations.dtype,
                device=activations.device
            )
            return torch.cat([activations, padding], dim=1)
        else:
            # No adjustment needed
            return activations
    
    def _normalize(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to activations.
        
        Args:
            activations (torch.Tensor): Activation tensor [batch_size, sequence_length, hidden_size]
        
        Returns:
            torch.Tensor: Normalized activations
        """
        if self.normalization is None:
            return activations
        
        if self.normalization == 'mean':
            # Mean subtraction
            mean = activations.mean(dim=-1, keepdim=True)
            return activations - mean
        
        elif self.normalization == 'l2':
            # L2 normalization
            norm = torch.norm(activations, p=2, dim=-1, keepdim=True)
            # Avoid division by zero
            norm = torch.clamp(norm, min=1e-12)
            return activations / norm
        
        elif self.normalization == 'layer':
            # Layer normalization
            if self.layer_norm is None or self.layer_norm.normalized_shape[0] != activations.size(-1):
                self.layer_norm = nn.LayerNorm(
                    normalized_shape=activations.size(-1),
                    device=activations.device
                )
            return self.layer_norm(activations)
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")