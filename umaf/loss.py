"""
Loss functions module for the UMAF Capability Extractor.

This module provides loss functions for training the capability extractor.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(
    fingerprints: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    epsilon: float = 1e-8,
    max_margin: float = 10.0
) -> torch.Tensor:
    """
    InfoNCE loss for contrastive learning.
    
    Args:
        fingerprints (torch.Tensor): Capability fingerprints [batch_size, output_dim]
        labels (torch.Tensor): Labels [batch_size]
        temperature (float): Temperature parameter
        epsilon (float): Small value to avoid numerical instability
        max_margin (float): Maximum margin for similarity values
    
    Returns:
        torch.Tensor: InfoNCE loss
    """
    # Normalize fingerprints
    fingerprints = F.normalize(fingerprints, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(fingerprints, fingerprints.T) / temperature
    
    # Clamp similarity values
    similarity_matrix = torch.clamp(similarity_matrix, -max_margin, max_margin)
    
    # Create positive mask
    labels = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float()
    
    # Remove self-similarity
    pos_mask.fill_diagonal_(0)
    
    # Compute positive and negative similarities
    exp_sim = torch.exp(similarity_matrix)
    pos_sim = torch.sum(exp_sim * pos_mask, dim=1)
    
    # Create negative mask
    neg_mask = 1 - pos_mask
    neg_mask.fill_diagonal_(0)
    neg_sim = torch.sum(exp_sim * neg_mask, dim=1)
    
    # Compute loss
    loss = -torch.log(pos_sim / (pos_sim + neg_sim + epsilon))
    
    # Average over non-zero elements
    non_zero = (pos_sim > 0).float()
    if non_zero.sum() > 0:
        loss = (loss * non_zero).sum() / non_zero.sum()
    else:
        loss = loss.mean()
    
    return loss


class AdvancedInfoNCELoss(nn.Module):
    """
    Advanced InfoNCE loss with additional features.
    
    Extends the basic InfoNCE loss with additional features like hard negative mining,
    temperature annealing, and class balancing.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        epsilon: float = 1e-8,
        max_margin: float = 10.0,
        hard_negative_weight: float = 1.0,
        class_balance: bool = True
    ):
        """
        Initialize advanced InfoNCE loss.
        
        Args:
            temperature (float): Temperature parameter
            epsilon (float): Small value to avoid numerical instability
            max_margin (float): Maximum margin for similarity values
            hard_negative_weight (float): Weight for hard negatives
            class_balance (bool): Whether to balance classes
        """
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        self.max_margin = max_margin
        self.hard_negative_weight = hard_negative_weight
        self.class_balance = class_balance
    
    def forward(
        self,
        fingerprints: torch.Tensor,
        labels: torch.Tensor,
        task_performances: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            fingerprints (torch.Tensor): Capability fingerprints [batch_size, output_dim]
            labels (torch.Tensor): Labels [batch_size]
            task_performances (Optional[torch.Tensor]): Task performances [batch_size]
        
        Returns:
            torch.Tensor: Loss value
        """
        # Normalize fingerprints
        fingerprints = F.normalize(fingerprints, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(fingerprints, fingerprints.T) / self.temperature
        
        # Clamp similarity values
        similarity_matrix = torch.clamp(similarity_matrix, -self.max_margin, self.max_margin)
        
        # Create positive mask
        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float()
        
        # Remove self-similarity
        pos_mask.fill_diagonal_(0)
        
        # Create negative mask
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)
        
        # Compute positive and negative similarities
        exp_sim = torch.exp(similarity_matrix)
        pos_sim = torch.sum(exp_sim * pos_mask, dim=1)
        
        # Hard negative mining
        if self.hard_negative_weight > 1.0:
            # Identify hard negatives (high similarity but different labels)
            hard_negatives = similarity_matrix * neg_mask
            hard_negative_values, _ = torch.topk(hard_negatives, k=5, dim=1)
            hard_negative_weight = torch.exp(hard_negative_values.mean(dim=1) / self.temperature)
            
            # Weight negative similarities
            neg_sim = torch.sum(exp_sim * neg_mask, dim=1)
            neg_sim = neg_sim + self.hard_negative_weight * hard_negative_weight * neg_sim
        else:
            neg_sim = torch.sum(exp_sim * neg_mask, dim=1)
        
        # Compute loss
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + self.epsilon))
        
        # Class balancing
        if self.class_balance:
            # Compute class weights
            unique_labels = torch.unique(labels)
            class_weights = torch.zeros_like(labels, dtype=torch.float)
            
            for label in unique_labels:
                mask = (labels == label).float()
                count = mask.sum()
                weight = 1.0 / count if count > 0 else 0.0
                class_weights[mask.bool()] = weight
            
            # Normalize weights
            class_weights = class_weights / class_weights.sum() * len(labels)
            
            # Apply weights
            loss = loss * class_weights.view(-1)
        
        # Average over non-zero elements
        non_zero = (pos_sim > 0).float()
        if non_zero.sum() > 0:
            loss = (loss * non_zero).sum() / non_zero.sum()
        else:
            loss = loss.mean()
        
        return loss


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple loss functions.
    
    Combines multiple loss functions with weighted averaging.
    """
    
    def __init__(self, losses, weights=None):
        """
        Initialize composite loss.
        
        Args:
            losses (list): List of loss functions
            weights (list, optional): List of weights for each loss function
        """
        super().__init__()
        self.losses = losses
        
        # Set equal weights if not provided
        if weights is None:
            weights = [1.0] * len(losses)
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
    
    def forward(self, *args, **kwargs):
        """
        Forward pass.
        
        Args:
            *args: Arguments to pass to loss functions
            **kwargs: Keyword arguments to pass to loss functions
        
        Returns:
            torch.Tensor: Weighted average of loss values
        """
        total_loss = 0.0
        
        for i, loss_fn in enumerate(self.losses):
            loss_value = loss_fn(*args, **kwargs)
            total_loss += self.weights[i] * loss_value
        
        return total_loss