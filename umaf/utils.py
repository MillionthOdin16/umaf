"""
Utility functions for the UMAF Capability Extractor.

This module provides utility functions for the UMAF framework.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import os
import json
import torch
import numpy as np


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config: Any, path: str):
    """
    Save configuration to a file.
    
    Args:
        config (Any): Configuration object
        path (str): Path to save the configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert config to dictionary
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config
    
    # Handle non-serializable objects
    for key, value in config_dict.items():
        if isinstance(value, torch.Tensor):
            config_dict[key] = value.tolist()
        elif hasattr(value, '__dict__'):
            config_dict[key] = value.__class__.__name__
    
    # Save to file
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        path (str): Path to load the configuration from
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(path, 'r') as f:
        config = json.load(f)
    
    return config


def visualize_fingerprints(
    fingerprints: List[torch.Tensor],
    labels: List[Any],
    method: str = 'tsne',
    save_path: Optional[str] = None
):
    """
    Visualize capability fingerprints.
    
    Args:
        fingerprints (List[torch.Tensor]): List of capability fingerprints
        labels (List[Any]): List of labels
        method (str): Dimensionality reduction method ('tsne', 'pca', or 'umap')
        save_path (Optional[str]): Path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
        
        # Convert fingerprints to numpy arrays
        fingerprints_np = [f.detach().cpu().numpy() for f in fingerprints]
        fingerprints_np = np.stack(fingerprints_np)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                print("UMAP not available, falling back to t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                method = 'tsne'
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Reduce dimensionality
        reduced = reducer.fit_transform(fingerprints_np)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Convert labels to colors if they are categorical
        unique_labels = list(set(labels))
        if len(unique_labels) <= 20:  # Arbitrary threshold for categorical labels
            label_to_color = {label: i for i, label in enumerate(unique_labels)}
            colors = [label_to_color[label] for label in labels]
            plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, cmap='tab20', s=100)
            
            # Add legend
            for label, color in label_to_color.items():
                plt.scatter([], [], c=[color], cmap='tab20', s=100, label=label)
            plt.legend()
        else:
            # Continuous labels
            plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', s=100)
            plt.colorbar(label='Label')
        
        plt.title(f'Capability Fingerprints Visualization ({method.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True)
        
        # Save or show
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
    
    except ImportError:
        print("Visualization libraries not available, skipping visualization")


def compute_model_similarity_matrix(
    models: List[Any],
    extractor: 'CapabilityExtractor',
    inputs: Dict[str, torch.Tensor],
    similarity_metric: Optional['SimilarityMetric'] = None
) -> Tuple[np.ndarray, List[torch.Tensor]]:
    """
    Compute similarity matrix between models.
    
    Args:
        models (List[Any]): List of models
        extractor (CapabilityExtractor): Capability extractor
        inputs (Dict[str, torch.Tensor]): Inputs for fingerprint extraction
        similarity_metric (Optional[SimilarityMetric]): Similarity metric
    
    Returns:
        Tuple[np.ndarray, List[torch.Tensor]]: Similarity matrix and fingerprints
    """
    # Extract fingerprints
    fingerprints = []
    for model in models:
        fingerprint = extractor.extract_fingerprint(model, inputs)
        fingerprints.append(fingerprint)
    
    # Compute similarity matrix
    n_models = len(models)
    similarity_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if similarity_metric is not None:
                similarity_matrix[i, j] = similarity_metric(fingerprints[i], fingerprints[j])
            else:
                # Default to cosine similarity
                similarity_matrix[i, j] = extractor.compute_similarity(fingerprints[i], fingerprints[j])
    
    return similarity_matrix, fingerprints


def find_similar_models(
    query_model: Any,
    candidate_models: List[Any],
    extractor: 'CapabilityExtractor',
    inputs: Dict[str, torch.Tensor],
    top_k: int = 5,
    similarity_metric: Optional['SimilarityMetric'] = None
) -> List[Tuple[int, float]]:
    """
    Find models similar to a query model.
    
    Args:
        query_model (Any): Query model
        candidate_models (List[Any]): List of candidate models
        extractor (CapabilityExtractor): Capability extractor
        inputs (Dict[str, torch.Tensor]): Inputs for fingerprint extraction
        top_k (int): Number of top similar models to return
        similarity_metric (Optional[SimilarityMetric]): Similarity metric
    
    Returns:
        List[Tuple[int, float]]: List of (model_idx, similarity) tuples
    """
    # Extract query fingerprint
    query_fingerprint = extractor.extract_fingerprint(query_model, inputs)
    
    # Extract candidate fingerprints
    candidate_fingerprints = []
    for model in candidate_models:
        fingerprint = extractor.extract_fingerprint(model, inputs)
        candidate_fingerprints.append(fingerprint)
    
    # Compute similarities
    similarities = []
    for i, fingerprint in enumerate(candidate_fingerprints):
        if similarity_metric is not None:
            similarity = similarity_metric(query_fingerprint, fingerprint)
        else:
            similarity = extractor.compute_similarity(query_fingerprint, fingerprint)
        similarities.append((i, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k
    return similarities[:top_k]