"""
Example usage of the UMAF Capability Extractor.

This module demonstrates how to use the UMAF framework for capability extraction.
"""

import torch
from transformers import AutoModel, AutoTokenizer

from umaf.config import CapabilityExtractorConfig
from umaf.extractor import CapabilityExtractor
from umaf.metrics import CosineSimilarity, KLDivergenceSimilarity, PearsonCorrelation
from umaf.utils import set_seed, visualize_fingerprints, compute_model_similarity_matrix


def main():
    """Example usage of the UMAF Capability Extractor."""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize configuration
    config = CapabilityExtractorConfig(
        input_dim=768,  # BERT hidden size
        output_dim=512,
        num_layers=4,
        num_heads=8,
        hidden_dim=2048,
        dropout=0.1,
        similarity_metric=CosineSimilarity(),
        normalization='layer'
    )
    
    # Initialize capability extractor
    extractor = CapabilityExtractor(config)
    
    # Load models
    model_names = [
        'bert-base-uncased',
        'bert-base-cased',
        'distilbert-base-uncased'
    ]
    
    models = []
    for name in model_names:
        model = AutoModel.from_pretrained(name)
        models.append(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare input
    text = "The Universal Model Adapter Fusion framework enables cross-model capability transfer."
    inputs = tokenizer(text, return_tensors="pt")
    
    # Extract fingerprints
    fingerprints = []
    for model in models:
        fingerprint = extractor.extract_fingerprint(model, inputs)
        fingerprints.append(fingerprint)
    
    # Compute similarities
    print("Similarity Matrix:")
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            similarity = extractor.compute_similarity(fingerprints[i], fingerprints[j])
            print(f"  {name_i} vs {name_j}: {similarity:.4f}")
    
    # Try different similarity metrics
    metrics = [
        ('Cosine', CosineSimilarity()),
        ('KL Divergence', KLDivergenceSimilarity()),
        ('Pearson', PearsonCorrelation())
    ]
    
    print("\nSimilarity with Different Metrics:")
    for name_i, fingerprint_i in zip(model_names, fingerprints):
        for name_j, fingerprint_j in zip(model_names, fingerprints):
            if name_i != name_j:
                print(f"  {name_i} vs {name_j}:")
                for metric_name, metric in metrics:
                    similarity = metric.compute(fingerprint_i, fingerprint_j)
                    print(f"    {metric_name}: {similarity:.4f}")
    
    # Compute model similarity matrix
    similarity_matrix, _ = compute_model_similarity_matrix(
        models, extractor, inputs, CosineSimilarity()
    )
    
    print("\nSimilarity Matrix:")
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            print(f"  {name_i} vs {name_j}: {similarity_matrix[i, j]:.4f}")
    
    # Visualize fingerprints
    # Note: This requires matplotlib and scikit-learn
    try:
        # Create dummy labels for visualization
        labels = list(range(len(fingerprints)))
        
        # Visualize fingerprints
        visualize_fingerprints(fingerprints, labels, method='tsne')
    except ImportError:
        print("Visualization libraries not available, skipping visualization")
    
    print("\nCapability extraction complete!")


if __name__ == "__main__":
    main()