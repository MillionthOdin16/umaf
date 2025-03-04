"""
Monitoring module for the UMAF Capability Extractor.

This module provides monitoring and logging functionality for the capability extractor.
"""

from typing import Dict, Any, List, Optional, Union
import time
import json
import os
from pathlib import Path
import numpy as np


class CapabilityExtractionMonitor:
    """
    Monitor for capability extraction.
    
    Tracks and logs metrics during training and evaluation.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize monitor.
        
        Args:
            log_dir (Optional[str]): Directory to save logs
        """
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'learning_rate': [],
            'epoch_time': [],
            'fingerprint_clustering': [],
            'capability_transfer_performance': [],
            'representational_similarity': []
        }
        
        self.log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        
        self.start_time = time.time()
    
    def log_metrics(self, **kwargs):
        """
        Log metrics.
        
        Args:
            **kwargs: Metrics to log
        """
        for key, value in kwargs.items():
            if key in self.metrics:
                if value is not None:
                    self.metrics[key].append(value)
        
        # Save metrics to file if log_dir is provided
        if self.log_dir is not None:
            self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to file."""
        metrics_file = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def generate_report(self) -> Dict[str, Dict[str, float]]:
        """
        Generate a report of metrics.
        
        Returns:
            Dict[str, Dict[str, float]]: Report of metrics
        """
        report = {}
        
        for key, values in self.metrics.items():
            if len(values) > 0:
                report[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'last': float(values[-1])
                }
        
        # Add total training time
        total_time = time.time() - self.start_time
        report['total_time'] = {
            'mean': total_time,
            'std': 0.0,
            'min': total_time,
            'max': total_time,
            'last': total_time
        }
        
        # Save report to file if log_dir is provided
        if self.log_dir is not None:
            report_file = os.path.join(self.log_dir, 'report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def plot_metrics(self, save_dir: Optional[str] = None):
        """
        Plot metrics.
        
        Args:
            save_dir (Optional[str]): Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create save directory if provided
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
            
            # Plot training and validation loss
            if len(self.metrics['train_loss']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train Loss')
                if len(self.metrics['val_loss']) > 0:
                    plt.plot(self.metrics['epoch'], self.metrics['val_loss'], label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True)
                
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, 'loss.png'))
                else:
                    plt.show()
            
            # Plot learning rate
            if len(self.metrics['learning_rate']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics['epoch'], self.metrics['learning_rate'])
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.grid(True)
                
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, 'learning_rate.png'))
                else:
                    plt.show()
            
            # Plot epoch time
            if len(self.metrics['epoch_time']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics['epoch'], self.metrics['epoch_time'])
                plt.xlabel('Epoch')
                plt.ylabel('Time (s)')
                plt.title('Epoch Time')
                plt.grid(True)
                
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, 'epoch_time.png'))
                else:
                    plt.show()
            
            # Plot fingerprint clustering quality
            if len(self.metrics['fingerprint_clustering']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics['fingerprint_clustering'])
                plt.xlabel('Evaluation')
                plt.ylabel('Silhouette Score')
                plt.title('Fingerprint Clustering Quality')
                plt.grid(True)
                
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, 'fingerprint_clustering.png'))
                else:
                    plt.show()
            
            # Plot capability transfer performance
            if len(self.metrics['capability_transfer_performance']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics['capability_transfer_performance'])
                plt.xlabel('Evaluation')
                plt.ylabel('Performance Improvement (%)')
                plt.title('Capability Transfer Performance')
                plt.grid(True)
                
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, 'capability_transfer_performance.png'))
                else:
                    plt.show()
            
            # Plot representational similarity
            if len(self.metrics['representational_similarity']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics['representational_similarity'])
                plt.xlabel('Evaluation')
                plt.ylabel('Correlation')
                plt.title('Representational Similarity')
                plt.grid(True)
                
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, 'representational_similarity.png'))
                else:
                    plt.show()
        
        except ImportError:
            print("matplotlib not available, skipping plots")
    
    def log_model_comparison(
        self,
        model_names: List[str],
        fingerprints: List[torch.Tensor],
        task_performances: List[float],
        similarity_metric: Optional[Callable] = None
    ):
        """
        Log model comparison.
        
        Args:
            model_names (List[str]): Names of models
            fingerprints (List[torch.Tensor]): Capability fingerprints
            task_performances (List[float]): Task performances
            similarity_metric (Optional[Callable]): Similarity metric
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            # Compute similarity matrix
            n_models = len(model_names)
            similarity_matrix = np.zeros((n_models, n_models))
            
            for i in range(n_models):
                for j in range(n_models):
                    if similarity_metric is not None:
                        similarity_matrix[i, j] = similarity_metric(fingerprints[i], fingerprints[j])
                    else:
                        # Default to cosine similarity
                        similarity_matrix[i, j] = F.cosine_similarity(
                            fingerprints[i].unsqueeze(0),
                            fingerprints[j].unsqueeze(0),
                            dim=1
                        ).item()
            
            # Create DataFrame
            df = pd.DataFrame(similarity_matrix, index=model_names, columns=model_names)
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1)
            plt.title('Model Similarity Matrix')
            
            if self.log_dir is not None:
                plt.savefig(os.path.join(self.log_dir, 'model_similarity.png'))
            else:
                plt.show()
            
            # Plot task performance vs. similarity
            performance_diff = []
            similarities = []
            model_pairs = []
            
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    performance_diff.append(abs(task_performances[i] - task_performances[j]))
                    similarities.append(similarity_matrix[i, j])
                    model_pairs.append(f"{model_names[i]} vs {model_names[j]}")
            
            plt.figure(figsize=(12, 8))
            plt.scatter(similarities, performance_diff)
            
            for i, pair in enumerate(model_pairs):
                plt.annotate(pair, (similarities[i], performance_diff[i]))
            
            plt.xlabel('Fingerprint Similarity')
            plt.ylabel('Task Performance Difference')
            plt.title('Fingerprint Similarity vs. Task Performance Difference')
            plt.grid(True)
            
            if self.log_dir is not None:
                plt.savefig(os.path.join(self.log_dir, 'similarity_vs_performance.png'))
            else:
                plt.show()
            
            # Compute correlation between similarity and performance difference
            correlation = np.corrcoef(similarities, performance_diff)[0, 1]
            
            # Log correlation
            self.log_metrics(representational_similarity=correlation)
            
            print(f"Correlation between fingerprint similarity and task performance difference: {correlation:.4f}")
        
        except ImportError:
            print("matplotlib, seaborn, or pandas not available, skipping model comparison plots")