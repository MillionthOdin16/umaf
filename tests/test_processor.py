"""
Tests for the InputProcessor class.

This module contains tests for the InputProcessor class.
"""

import unittest
import torch

from umaf.processor import InputProcessor


class TestInputProcessor(unittest.TestCase):
    """Tests for the InputProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = InputProcessor(
            max_length=5,
            normalization='layer',
            pad_value=0.0
        )
        
        # Create test tensors
        self.activations = torch.randn(2, 3, 4)  # [batch_size, seq_len, hidden_size]
        self.activations_dict = {'last_hidden_state': self.activations}
        self.activations_dict2 = {'hidden_states': [self.activations, self.activations]}
    
    def test_process_tensor(self):
        """Test processing a tensor."""
        # Process tensor
        processed = self.processor.process(self.activations)
        
        # Check output shape
        self.assertEqual(processed.shape, (2, 5, 4))
        
        # Check that padding was added
        self.assertEqual(processed[:, 3:, :].sum().item(), 0.0)
    
    def test_process_dict(self):
        """Test processing a dictionary."""
        # Process dictionary with last_hidden_state
        processed = self.processor.process(self.activations_dict)
        
        # Check output shape
        self.assertEqual(processed.shape, (2, 5, 4))
        
        # Process dictionary with hidden_states
        processed = self.processor.process(self.activations_dict2)
        
        # Check output shape
        self.assertEqual(processed.shape, (2, 5, 4))
    
    def test_handle_sequence_length(self):
        """Test handling sequence length."""
        # Test padding
        activations = torch.randn(2, 3, 4)
        processed = self.processor._handle_sequence_length(activations)
        self.assertEqual(processed.shape, (2, 5, 4))
        
        # Test truncation
        activations = torch.randn(2, 7, 4)
        processed = self.processor._handle_sequence_length(activations)
        self.assertEqual(processed.shape, (2, 5, 4))
        
        # Test no change
        activations = torch.randn(2, 5, 4)
        processed = self.processor._handle_sequence_length(activations)
        self.assertEqual(processed.shape, (2, 5, 4))
    
    def test_normalize(self):
        """Test normalization."""
        # Test mean normalization
        processor = InputProcessor(normalization='mean')
        activations = torch.randn(2, 3, 4)
        processed = processor._normalize(activations)
        self.assertEqual(processed.shape, (2, 3, 4))
        self.assertAlmostEqual(processed.mean().item(), 0.0, places=5)
        
        # Test L2 normalization
        processor = InputProcessor(normalization='l2')
        activations = torch.randn(2, 3, 4)
        processed = processor._normalize(activations)
        self.assertEqual(processed.shape, (2, 3, 4))
        norms = torch.norm(processed, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
        
        # Test layer normalization
        processor = InputProcessor(normalization='layer')
        activations = torch.randn(2, 3, 4)
        processed = processor._normalize(activations)
        self.assertEqual(processed.shape, (2, 3, 4))
        
        # Test no normalization
        processor = InputProcessor(normalization=None)
        activations = torch.randn(2, 3, 4)
        processed = processor._normalize(activations)
        self.assertEqual(processed.shape, (2, 3, 4))
        self.assertTrue(torch.allclose(processed, activations))


if __name__ == '__main__':
    unittest.main()