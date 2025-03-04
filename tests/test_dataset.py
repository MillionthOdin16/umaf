"""
Tests for the dataset classes.

This module contains tests for the dataset classes.
"""

import unittest
import torch
import torch.nn as nn

from umaf.dataset import CapabilityDataset, CapabilityPairDataset


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
    
    def forward(self, **kwargs):
        """Forward pass."""
        batch_size = kwargs.get('input_ids', torch.tensor([[0]])).shape[0]
        seq_len = kwargs.get('input_ids', torch.tensor([[0]])).shape[1]
        
        # Create mock hidden states
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_size)
        
        # Create mock output
        class MockOutput:
            def __init__(self, hidden_states):
                self.last_hidden_state = hidden_states
                self.hidden_states = [hidden_states]
        
        return MockOutput(hidden_states)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __call__(self, text, **kwargs):
        """Tokenize text."""
        # Create mock tokens
        if isinstance(text, list):
            batch_size = len(text)
        else:
            batch_size = 1
        
        max_length = kwargs.get('max_length', 10)
        
        # Create mock input_ids and attention_mask
        input_ids = torch.randint(0, 1000, (batch_size, max_length))
        attention_mask = torch.ones(batch_size, max_length)
        
        # Return as dictionary
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size=10):
        """Initialize mock dataset."""
        self.size = size
        self.data = [
            {'text': f'Sample {i}', 'label': i % 3}
            for i in range(size)
        ]
    
    def __len__(self):
        """Get dataset length."""
        return self.size
    
    def __getitem__(self, idx):
        """Get dataset item."""
        return self.data[idx]


class TestDatasetClasses(unittest.TestCase):
    """Tests for the dataset classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock models
        self.models = [MockModel(), MockModel()]
        
        # Create mock datasets
        self.datasets = [MockDataset(5), MockDataset(5)]
        
        # Create mock tokenizer
        self.tokenizer = MockTokenizer()
        
        # Create capability dataset
        self.capability_dataset = CapabilityDataset(
            models=self.models,
            datasets=self.datasets,
            tokenizer=self.tokenizer,
            max_length=10
        )
        
        # Create capability pair dataset
        self.fingerprints = torch.randn(10, 512)
        self.model_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.labels = torch.tensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 2])
        self.task_performances = torch.tensor([0.9, 0.8])
        
        self.pair_dataset = CapabilityPairDataset(
            fingerprints=self.fingerprints,
            model_indices=self.model_indices,
            labels=self.labels,
            task_performances=self.task_performances
        )
    
    def test_capability_dataset(self):
        """Test CapabilityDataset."""
        # Check length
        self.assertEqual(len(self.capability_dataset), 10)
        
        # Check getitem
        inputs, model_idx, label = self.capability_dataset[0]
        
        # Check that inputs is a dictionary
        self.assertIsInstance(inputs, dict)
        
        # Check that inputs contains input_ids and attention_mask
        self.assertIn('input_ids', inputs)
        self.assertIn('attention_mask', inputs)
        
        # Check that model_idx is an integer
        self.assertIsInstance(model_idx, int)
        
        # Check that label is an integer
        self.assertIsInstance(label, int)
    
    def test_capability_pair_dataset(self):
        """Test CapabilityPairDataset."""
        # Check length
        self.assertGreater(len(self.pair_dataset), 0)
        
        # Check getitem
        fingerprint1, fingerprint2, is_positive = self.pair_dataset[0]
        
        # Check that fingerprints are tensors
        self.assertIsInstance(fingerprint1, torch.Tensor)
        self.assertIsInstance(fingerprint2, torch.Tensor)
        
        # Check that is_positive is a boolean
        self.assertIsInstance(is_positive, bool)
        
        # Check that fingerprints have the right shape
        self.assertEqual(fingerprint1.shape, (512,))
        self.assertEqual(fingerprint2.shape, (512,))


if __name__ == '__main__':
    unittest.main()