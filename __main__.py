"""
Main entry point for the UMAF Capability Extractor.

This module provides a command-line interface for the UMAF framework.
"""

import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer

from umaf.config import CapabilityExtractorConfig
from umaf.extractor import CapabilityExtractor
from umaf.metrics import CosineSimilarity, KLDivergenceSimilarity, PearsonCorrelation
from umaf.utils import set_seed, visualize_fingerprints, compute_model_similarity_matrix, save_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="UMAF Capability Extractor")
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="extract",
        choices=["extract", "compare", "visualize"],
        help="Operation mode"
    )
    
    # Model arguments
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["bert-base-uncased"],
        help="Model names or paths"
    )
    
    # Input arguments
    parser.add_argument(
        "--input_text",
        type=str,
        default="The Universal Model Adapter Fusion framework enables cross-model capability transfer.",
        help="Input text for capability extraction"
    )
    
    # Configuration arguments
    parser.add_argument("--input_dim", type=int, default=768, help="Input dimension")
    parser.add_argument("--output_dim", type=int, default=512, help="Output dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--normalization",
        type=str,
        default="layer",
        choices=["mean", "l2", "layer", "none"],
        help="Normalization method"
    )
    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="cosine",
        choices=["cosine", "kl", "pearson"],
        help="Similarity metric"
    )
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--save_config", action="store_true", help="Save configuration")
    parser.add_argument("--save_fingerprints", action="store_true", help="Save fingerprints")
    parser.add_argument("--save_visualization", action="store_true", help="Save visualization")
    
    # Visualization arguments
    parser.add_argument(
        "--visualization_method",
        type=str,
        default="tsne",
        choices=["tsne", "pca", "umap"],
        help="Visualization method"
    )
    
    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize similarity metric
    if args.similarity_metric == "cosine":
        similarity_metric = CosineSimilarity()
    elif args.similarity_metric == "kl":
        similarity_metric = KLDivergenceSimilarity()
    elif args.similarity_metric == "pearson":
        similarity_metric = PearsonCorrelation()
    else:
        raise ValueError(f"Unknown similarity metric: {args.similarity_metric}")
    
    # Initialize configuration
    config = CapabilityExtractorConfig(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        similarity_metric=similarity_metric,
        normalization=None if args.normalization == "none" else args.normalization
    )
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, os.path.join(args.output_dir, "config.json"))
    
    # Initialize capability extractor
    extractor = CapabilityExtractor(config, device=device)
    
    # Load models
    models = []
    for model_name in args.models:
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        models.append(model)
    
    # Load tokenizer (using the first model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.models[0])
    
    # Prepare input
    inputs = tokenizer(args.input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract fingerprints
    fingerprints = []
    for model in models:
        fingerprint = extractor.extract_fingerprint(model, inputs)
        fingerprints.append(fingerprint)
    
    # Save fingerprints if requested
    if args.save_fingerprints:
        fingerprints_dir = os.path.join(args.output_dir, "fingerprints")
        os.makedirs(fingerprints_dir, exist_ok=True)
        
        for i, (model_name, fingerprint) in enumerate(zip(args.models, fingerprints)):
            # Create a safe filename
            safe_name = model_name.replace("/", "_").replace("\\", "_")
            torch.save(fingerprint, os.path.join(fingerprints_dir, f"{safe_name}.pt"))
    
    # Process based on mode
    if args.mode == "extract":
        # Print fingerprints
        print("Extracted Fingerprints:")
        for model_name, fingerprint in zip(args.models, fingerprints):
            print(f"  {model_name}: Shape {fingerprint.shape}")
    
    elif args.mode == "compare":
        # Compute similarity matrix
        similarity_matrix, _ = compute_model_similarity_matrix(
            models, extractor, inputs, similarity_metric
        )
        
        # Print similarity matrix
        print("Similarity Matrix:")
        for i, name_i in enumerate(args.models):
            for j, name_j in enumerate(args.models):
                print(f"  {name_i} vs {name_j}: {similarity_matrix[i, j]:.4f}")
        
        # Save similarity matrix if requested
        if args.save_fingerprints:
            import numpy as np
            np.save(os.path.join(args.output_dir, "similarity_matrix.npy"), similarity_matrix)
    
    elif args.mode == "visualize":
        # Create dummy labels for visualization
        labels = list(range(len(fingerprints)))
        
        # Visualize fingerprints
        if args.save_visualization:
            visualization_path = os.path.join(args.output_dir, f"visualization_{args.visualization_method}.png")
        else:
            visualization_path = None
        
        visualize_fingerprints(
            fingerprints,
            labels,
            method=args.visualization_method,
            save_path=visualization_path
        )
    
    print("\nCapability extraction complete!")


if __name__ == "__main__":
    main()