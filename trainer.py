"""
Trainer module for the UMAF Capability Extractor.

This module provides training functionality for the capability extractor.
"""

import time
from typing import Dict, Any, Optional, List, Tuple, Callable, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from umaf.extractor import CapabilityExtractor
from umaf.dataset import CapabilityDataset
from umaf.loss import info_nce_loss, AdvancedInfoNCELoss
from umaf.monitor import CapabilityExtractionMonitor


class CapabilityExtractorTrainer:
    """
    Trainer for the capability extractor.
    
    Handles training, validation, and evaluation of the capability extractor.
    """
    
    def __init__(
        self,
        extractor: CapabilityExtractor,
        train_dataset: CapabilityDataset,
        val_dataset: Optional[CapabilityDataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        monitor: Optional[CapabilityExtractionMonitor] = None
    ):
        """
        Initialize trainer.
        
        Args:
            extractor (CapabilityExtractor): Capability extractor to train
            train_dataset (CapabilityDataset): Training dataset
            val_dataset (Optional[CapabilityDataset]): Validation dataset
            optimizer (Optional[torch.optim.Optimizer]): Optimizer
            loss_fn (Optional[Callable]): Loss function
            device (Optional[torch.device]): Device to use for training
            monitor (Optional[CapabilityExtractionMonitor]): Monitor for logging metrics
        """
        self.extractor = extractor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(extractor.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer
        
        # Set up loss function
        if loss_fn is None:
            self.loss_fn = AdvancedInfoNCELoss(
                temperature=extractor.config.temperature,
                hard_negative_weight=1.2,
                class_balance=True
            )
        else:
            self.loss_fn = loss_fn
        
        # Set up monitor
        if monitor is None:
            self.monitor = CapabilityExtractionMonitor()
        else:
            self.monitor = monitor
        
        # Move extractor to device
        self.extractor.to(self.device)
    
    def train(
        self,
        num_epochs: int,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: Optional[str] = 'cosine',
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 5,
        checkpoint_path: Optional[str] = None,
        log_interval: int = 10,
        use_mixed_precision: bool = False
    ) -> Dict[str, List[float]]:
        """
        Train the capability extractor.
        
        Args:
            num_epochs (int): Number of epochs to train for
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay
            lr_scheduler (Optional[str]): Learning rate scheduler ('cosine', 'linear', 'step', or None)
            gradient_accumulation_steps (int): Number of steps to accumulate gradients
            max_grad_norm (float): Maximum gradient norm for gradient clipping
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping
            checkpoint_path (Optional[str]): Path to save checkpoints
            log_interval (int): Interval for logging metrics
            use_mixed_precision (bool): Whether to use mixed precision training
        
        Returns:
            Dict[str, List[float]]: Training metrics
        """
        # Set up data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Set up validation loader if validation dataset is provided
        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.extractor.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Set up learning rate scheduler
        scheduler = None
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs
            )
        elif lr_scheduler == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=num_epochs
            )
        elif lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=num_epochs // 3,
                gamma=0.1
            )
        
        # Set up mixed precision training
        scaler = None
        if use_mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        
        # Initialize metrics
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_time_per_epoch': []
        }
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Start timer
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self._train_epoch(
                train_loader,
                gradient_accumulation_steps,
                max_grad_norm,
                log_interval,
                scaler
            )
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Validate if validation dataset is provided
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                metrics['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save checkpoint
                    if checkpoint_path is not None:
                        self.extractor.save(f"{checkpoint_path}_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Update metrics
            metrics['train_loss'].append(train_loss)
            metrics['train_time_per_epoch'].append(epoch_time)
            
            # Log metrics
            self.monitor.log_metrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                epoch_time=epoch_time
            )
            
            # Print progress
            val_str = f", val_loss: {val_loss:.4f}" if val_loss is not None else ""
            print(f"Epoch {epoch + 1}/{num_epochs}, train_loss: {train_loss:.4f}{val_str}, time: {epoch_time:.2f}s")
            
            # Save checkpoint
            if checkpoint_path is not None:
                self.extractor.save(f"{checkpoint_path}_last.pt")
        
        # Generate final report
        report = self.monitor.generate_report()
        print("\nTraining complete. Final metrics:")
        for key, value in report.items():
            print(f"  {key}: mean={value['mean']:.4f}, std={value['std']:.4f}, best={value['best']:.4f}")
        
        return metrics
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
        log_interval: int,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            gradient_accumulation_steps (int): Number of steps to accumulate gradients
            max_grad_norm (float): Maximum gradient norm for gradient clipping
            log_interval (int): Interval for logging metrics
            scaler (Optional[torch.cuda.amp.GradScaler]): Gradient scaler for mixed precision training
        
        Returns:
            float: Average training loss
        """
        self.extractor.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, model_indices, labels) in enumerate(train_loader):
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            model_indices = model_indices.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Extract fingerprints
                    fingerprints = []
                    for model_idx in torch.unique(model_indices):
                        # Get model
                        model = self.train_dataset.models[model_idx.item()]
                        
                        # Get inputs for this model
                        model_mask = (model_indices == model_idx)
                        model_inputs = {k: v[model_mask] for k, v in inputs.items()}
                        
                        # Extract fingerprints
                        model_fingerprints = self.extractor.extract_fingerprint(model, model_inputs)
                        fingerprints.append(model_fingerprints)
                    
                    # Concatenate fingerprints
                    fingerprints = torch.cat(fingerprints, dim=0)
                    
                    # Compute loss
                    loss = self.loss_fn(fingerprints, labels)
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.extractor.parameters(), max_grad_norm)
                    
                    # Optimizer step with gradient scaling
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Extract fingerprints
                fingerprints = []
                for model_idx in torch.unique(model_indices):
                    # Get model
                    model = self.train_dataset.models[model_idx.item()]
                    
                    # Get inputs for this model
                    model_mask = (model_indices == model_idx)
                    model_inputs = {k: v[model_mask] for k, v in inputs.items()}
                    
                    # Extract fingerprints
                    model_fingerprints = self.extractor.extract_fingerprint(model, model_inputs)
                    fingerprints.append(model_fingerprints)
                
                # Concatenate fingerprints
                fingerprints = torch.cat(fingerprints, dim=0)
                
                # Compute loss
                loss = self.loss_fn(fingerprints, labels)
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.extractor.parameters(), max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update total loss
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
        
        # Return average loss
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            float: Average validation loss
        """
        self.extractor.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, model_indices, labels in val_loader:
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                model_indices = model_indices.to(self.device)
                labels = labels.to(self.device)
                
                # Extract fingerprints
                fingerprints = []
                for model_idx in torch.unique(model_indices):
                    # Get model
                    model = self.val_dataset.models[model_idx.item()]
                    
                    # Get inputs for this model
                    model_mask = (model_indices == model_idx)
                    model_inputs = {k: v[model_mask] for k, v in inputs.items()}
                    
                    # Extract fingerprints
                    model_fingerprints = self.extractor.extract_fingerprint(model, model_inputs)
                    fingerprints.append(model_fingerprints)
                
                # Concatenate fingerprints
                fingerprints = torch.cat(fingerprints, dim=0)
                
                # Compute loss
                loss = self.loss_fn(fingerprints, labels)
                
                # Update total loss
                total_loss += loss.item()
        
        # Return average loss
        return total_loss / len(val_loader)
    
    def evaluate(
        self,
        test_dataset: CapabilityDataset,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_dataset (CapabilityDataset): Test dataset
            batch_size (int): Batch size
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Set up data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Evaluate
        self.extractor.eval()
        total_loss = 0.0
        all_fingerprints = []
        all_labels = []
        all_model_indices = []
        
        with torch.no_grad():
            for inputs, model_indices, labels in test_loader:
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                model_indices = model_indices.to(self.device)
                labels = labels.to(self.device)
                
                # Extract fingerprints
                fingerprints = []
                for model_idx in torch.unique(model_indices):
                    # Get model
                    model = test_dataset.models[model_idx.item()]
                    
                    # Get inputs for this model
                    model_mask = (model_indices == model_idx)
                    model_inputs = {k: v[model_mask] for k, v in inputs.items()}
                    
                    # Extract fingerprints
                    model_fingerprints = self.extractor.extract_fingerprint(model, model_inputs)
                    fingerprints.append(model_fingerprints)
                
                # Concatenate fingerprints
                fingerprints = torch.cat(fingerprints, dim=0)
                
                # Compute loss
                loss = self.loss_fn(fingerprints, labels)
                
                # Update total loss
                total_loss += loss.item()
                
                # Store fingerprints and labels
                all_fingerprints.append(fingerprints.cpu())
                all_labels.append(labels.cpu())
                all_model_indices.append(model_indices.cpu())
        
        # Concatenate all fingerprints and labels
        all_fingerprints = torch.cat(all_fingerprints, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_model_indices = torch.cat(all_model_indices, dim=0)
        
        # Compute metrics
        metrics = {
            'test_loss': total_loss / len(test_loader)
        }
        
        # Compute additional metrics if scikit-learn is available
        try:
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import KMeans
            
            # Compute fingerprint clustering quality
            fingerprints_np = all_fingerprints.numpy()
            labels_np = all_labels.numpy()
            
            # K-means clustering
            kmeans = KMeans(n_clusters=len(torch.unique(all_labels)))
            cluster_labels = kmeans.fit_predict(fingerprints_np)
            
            # Silhouette score
            silhouette = silhouette_score(fingerprints_np, cluster_labels)
            metrics['fingerprint_clustering'] = silhouette
            
            # Log metrics
            self.monitor.log_metrics(fingerprint_clustering=silhouette)
        except ImportError:
            print("scikit-learn not available, skipping clustering metrics")
        
        # Log metrics
        self.monitor.log_metrics(test_loss=metrics['test_loss'])
        
        return metrics