"""
Tests for the loss functions.

This module contains tests for the loss functions.
"""

import unittest
import torch

from umaf.loss import info_nce_loss, AdvancedInfoNCELoss, CompositeLoss


class TestLossFunctions(unittest.TestCase):
    """Tests for the loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test tensors
        self.fingerprints = torch.randn(8, 512)
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    
    def test_info_nce_loss(self):
        """Test InfoNCE loss."""
        # Compute loss
        loss = info_nce_loss(self.fingerprints, self.labels)
        
        # Check that loss is a scalar
        self.assertEqual(loss.shape, ())
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0.0)
        
        # Check that loss requires gradients
        self.assertTrue(loss.requires_grad)
    
    def test_advanced_info_nce_loss(self):
        """Test advanced InfoNCE loss."""
        # Create loss function
        loss_fn = AdvancedInfoNCELoss(
            temperature=0.07,
            hard_negative_weight=1.2,
            class_balance=True
        )
        
        # Compute loss
        loss = loss_fn(self.fingerprints, self.labels)
        
        # Check that loss is a scalar
        self.assertEqual(loss.shape, ())
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0.0)
        
        # Check that loss requires gradients
        self.assertTrue(loss.requires_grad)
        
        # Test with task performances
        task_performances = torch.tensor([0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6])
        loss = loss_fn(self.fingerprints, self.labels, task_performances)
        
        # Check that loss is a scalar
        self.assertEqual(loss.shape, ())
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0.0)
        
        # Check that loss requires gradients
        self.assertTrue(loss.requires_grad)
    
    def test_composite_loss(self):
        """Test composite loss."""
        # Create component loss functions
        loss_fn1 = AdvancedInfoNCELoss(temperature=0.07)
        loss_fn2 = AdvancedInfoNCELoss(temperature=0.1)
        
        # Create composite loss function
        loss_fn = CompositeLoss([loss_fn1, loss_fn2], weights=[0.7, 0.3])
        
        # Compute loss
        loss = loss_fn(self.fingerprints, self.labels)
        
        # Check that loss is a scalar
        self.assertEqual(loss.shape, ())
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0.0)
        
        # Check that loss requires gradients
        self.assertTrue(loss.requires_grad)
        
        # Check that loss is a weighted average of component losses
        loss1 = loss_fn1(self.fingerprints, self.labels)
        loss2 = loss_fn2(self.fingerprints, self.labels)
        expected_loss = 0.7 * loss1 + 0.3 * loss2
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)


if __name__ == '__main__':
    unittest.main()