"""
Similarity metrics module for the UMAF Capability Extractor.

This module defines various similarity metrics used for comparing capability fingerprints.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr


class SimilarityMetric(ABC):
    """
    Abstract base class for similarity metrics.
    
    All similarity metrics should inherit from this class and implement the compute method.
    """
    
    @abstractmethod
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute similarity between two tensors.
        
        Args:
            x (torch.Tensor): First tensor
            y (torch.Tensor): Second tensor
        
        Returns:
            float: Similarity score
        """
        pass
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Make the metric callable.
        
        Args:
            x (torch.Tensor): First tensor
            y (torch.Tensor): Second tensor
        
        Returns:
            float: Similarity score
        """
        return self.compute(x, y)


class CosineSimilarity(SimilarityMetric):
    """
    Cosine similarity metric.
    
    Computes the cosine similarity between two tensors.
    """
    
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute cosine similarity between two tensors.
        
        Args:
            x (torch.Tensor): First tensor
            y (torch.Tensor): Second tensor
        
        Returns:
            float: Cosine similarity score
        """
        # Ensure tensors are properly shaped
        if x.dim() > 1 and y.dim() > 1:
            # For batched inputs, compute pairwise similarities and take mean
            return F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2).mean().item()
        else:
            # For single vectors
            return F.cosine_similarity(x, y, dim=0).mean().item()


class KLDivergenceSimilarity(SimilarityMetric):
    """
    KL Divergence similarity metric.
    
    Computes the negative KL divergence between two tensors (negated to represent similarity).
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize KL Divergence similarity metric.
        
        Args:
            epsilon (float): Small value to avoid numerical instability
        """
        self.epsilon = epsilon
    
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute negative KL divergence between two tensors.
        
        Args:
            x (torch.Tensor): First tensor
            y (torch.Tensor): Second tensor
        
        Returns:
            float: Negative KL divergence (higher is more similar)
        """
        # Normalize to ensure valid distributions
        x = F.softmax(x, dim=-1)
        y = F.softmax(y, dim=-1)
        
        # Add epsilon to avoid log(0)
        x = x + self.epsilon
        y = y + self.epsilon
        
        # Renormalize
        x = x / x.sum(dim=-1, keepdim=True)
        y = y / y.sum(dim=-1, keepdim=True)
        
        # Compute KL divergence
        kl_div = F.kl_div(x.log(), y, reduction='batchmean').item()
        
        # Return negative KL divergence (higher is more similar)
        return -kl_div


class PearsonCorrelation(SimilarityMetric):
    """
    Pearson correlation similarity metric.
    
    Computes the Pearson correlation coefficient between two tensors.
    """
    
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute Pearson correlation between two tensors.
        
        Args:
            x (torch.Tensor): First tensor
            y (torch.Tensor): Second tensor
        
        Returns:
            float: Pearson correlation coefficient
        """
        # Convert to numpy for scipy's pearsonr
        x_np = x.detach().cpu().numpy().flatten()
        y_np = y.detach().cpu().numpy().flatten()
        
        # Compute Pearson correlation
        corr, _ = pearsonr(x_np, y_np)
        
        return corr


class CompositeSimilarity(SimilarityMetric):
    """
    Composite similarity metric.
    
    Combines multiple similarity metrics with weighted averaging.
    """
    
    def __init__(self, metrics: list):
        """
        Initialize composite similarity metric.
        
        Args:
            metrics (list[Tuple[SimilarityMetric, float]]): List of (metric, weight) tuples
        """
        self.metrics = metrics
        
        # Normalize weights
        total_weight = sum(weight for _, weight in metrics)
        self.metrics = [(metric, weight / total_weight) for metric, weight in metrics]
    
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute weighted average of multiple similarity metrics.
        
        Args:
            x (torch.Tensor): First tensor
            y (torch.Tensor): Second tensor
        
        Returns:
            float: Composite similarity score
        """
        return sum(weight * metric.compute(x, y) for metric, weight in self.metrics)


class TaskPerformanceSimilarity(SimilarityMetric):
    """
    Task performance similarity metric.
    
    Computes similarity based on task performance metrics.
    """
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize task performance similarity metric.
        
        Args:
            threshold (float): Threshold for considering performances similar (default: 0.05 or 5%)
        """
        self.threshold = threshold
    
    def compute(self, x: Union[torch.Tensor, float], y: Union[torch.Tensor, float]) -> float:
        """
        Compute task performance similarity.
        
        Args:
            x (Union[torch.Tensor, float]): First performance metric
            y (Union[torch.Tensor, float]): Second performance metric
        
        Returns:
            float: Similarity score based on performance difference
        """
        # Convert to float if tensors
        if isinstance(x, torch.Tensor):
            x = x.item()
        if isinstance(y, torch.Tensor):
            y = y.item()
        
        # Compute absolute difference
        diff = abs(x - y)
        
        # If difference is below threshold, high similarity
        if diff <= self.threshold:
            # Scale from 0.8 to 1.0 based on how close they are
            return 1.0 - (diff / self.threshold) * 0.2
        else:
            # Scale from 0.0 to 0.8 based on difference
            max_diff = 1.0  # Maximum expected difference
            return max(0.0, 0.8 * (1.0 - (diff - self.threshold) / (max_diff - self.threshold)))