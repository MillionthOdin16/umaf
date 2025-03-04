"""
Dataset module for the UMAF Capability Extractor.

This module provides dataset classes for training and evaluating the capability extractor.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable, Union

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class CapabilityDataset(Dataset):
    """
    Dataset for training and evaluating the capability extractor.
    
    Prepares data from multiple models and datasets for capability extraction.
    """
    
    def __init__(
        self,
        models: List[Any],
        datasets: List[Any],
        tokenizer: Any,
        max_length: int = 128,
        text_field: str = "text",
        label_field: str = "label",
        transform: Optional[Callable] = None
    ):
        """
        Initialize capability dataset.
        
        Args:
            models (List[Any]): List of pre-trained models
            datasets (List[Any]): List of datasets (one per model)
            tokenizer (Any): Tokenizer for processing text
            max_length (int): Maximum sequence length
            text_field (str): Field name for text in dataset samples
            label_field (str): Field name for labels in dataset samples
            transform (Optional[Callable]): Optional transform to apply to samples
        """
        self.models = models
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.label_field = label_field
        self.transform = transform
        
        # Prepare data
        self.data = self._prepare_data()
    
    def _prepare_data(self) -> List[Tuple[Dict[str, torch.Tensor], int, Any]]:
        """
        Prepare data for capability extraction.
        
        Returns:
            List[Tuple[Dict[str, torch.Tensor], int, Any]]: List of (inputs, model_idx, label) tuples
        """
        data = []
        
        for model_idx, (model, dataset) in enumerate(zip(self.models, self.datasets)):
            for sample in dataset:
                # Extract text and label
                text = sample[self.text_field]
                label = sample[self.label_field]
                
                # Tokenize text
                inputs = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                
                # Add to data
                data.append((inputs, model_idx, label))
        
        return data
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            int: Number of samples in dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int, Any]:
        """
        Get dataset item.
        
        Args:
            idx (int): Index of item to get
        
        Returns:
            Tuple[Dict[str, torch.Tensor], int, Any]: (inputs, model_idx, label) tuple
        """
        inputs, model_idx, label = self.data[idx]
        
        # Apply transform if provided
        if self.transform is not None:
            inputs, model_idx, label = self.transform(inputs, model_idx, label)
        
        return inputs, model_idx, label


class CapabilityPairDataset(Dataset):
    """
    Dataset for training with pairs of samples.
    
    Creates positive and negative pairs for contrastive learning.
    """
    
    def __init__(
        self,
        fingerprints: torch.Tensor,
        model_indices: torch.Tensor,
        labels: torch.Tensor,
        task_performances: torch.Tensor,
        positive_threshold: float = 0.85,
        task_weight: float = 0.6,
        rep_weight: float = 0.4,
        similarity_fn: Callable = F.cosine_similarity
    ):
        """
        Initialize capability pair dataset.
        
        Args:
            fingerprints (torch.Tensor): Capability fingerprints [n_samples, output_dim]
            model_indices (torch.Tensor): Model indices [n_samples]
            labels (torch.Tensor): Labels [n_samples]
            task_performances (torch.Tensor): Task performances [n_models]
            positive_threshold (float): Threshold for positive pair definition
            task_weight (float): Weight for task performance in composite similarity score
            rep_weight (float): Weight for representational similarity in composite score
            similarity_fn (Callable): Function for computing similarity between fingerprints
        """
        self.fingerprints = fingerprints
        self.model_indices = model_indices
        self.labels = labels
        self.task_performances = task_performances
        self.positive_threshold = positive_threshold
        self.task_weight = task_weight
        self.rep_weight = rep_weight
        self.similarity_fn = similarity_fn
        
        # Create pairs
        self.pairs = self._create_pairs()
    
    def _create_pairs(self) -> List[Tuple[int, int, bool]]:
        """
        Create positive and negative pairs for contrastive learning.
        
        Returns:
            List[Tuple[int, int, bool]]: List of (idx1, idx2, is_positive) tuples
        """
        n_samples = len(self.fingerprints)
        pairs = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Get model indices
                model_i = self.model_indices[i].item()
                model_j = self.model_indices[j].item()
                
                # Get task performances
                perf_i = self.task_performances[model_i].item()
                perf_j = self.task_performances[model_j].item()
                
                # Compute task performance similarity
                task_diff = abs(perf_i - perf_j)
                task_sim = 1.0 if task_diff <= 0.05 else 0.0
                
                # Compute representational similarity
                rep_sim = self.similarity_fn(
                    self.fingerprints[i].unsqueeze(0),
                    self.fingerprints[j].unsqueeze(0),
                    dim=1
                ).item()
                
                # Compute composite similarity score
                composite_score = (
                    self.task_weight * task_sim +
                    self.rep_weight * rep_sim
                )
                
                # Determine if positive pair
                is_positive = composite_score > self.positive_threshold
                
                # Add pair
                pairs.append((i, j, is_positive))
        
        return pairs
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            int: Number of pairs in dataset
        """
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Get dataset item.
        
        Args:
            idx (int): Index of item to get
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, bool]: (fingerprint1, fingerprint2, is_positive) tuple
        """
        i, j, is_positive = self.pairs[idx]
        return self.fingerprints[i], self.fingerprints[j], is_positive