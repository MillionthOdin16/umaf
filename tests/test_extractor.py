"""
Tests for the CapabilityExtractor class.

This module contains tests for the CapabilityExtractor class.
"""

import unittest
import torch
import torch.nn as nn

from umaf.config import CapabilityExtractorConfig
from umaf.extractor import CapabilityExtractor
from umaf.metrics import CosineSimilarity, KLDivergenceSimilarity


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


class TestCapabilityExtractor(unittest.TestCase):
    """Tests for the CapabilityExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CapabilityExtractorConfig(
            input_dim=768,
            output_dim=512,
            num_layers=2,
            num_heads=4,
            hidden_dim=1024,
            dropout=0.1
        )
        self.extractor = CapabilityExtractor(self.config)
        self.model = MockModel()
        self.inputs = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.extractor.config.input_dim, 768)
        self.assertEqual(self.extractor.config.output_dim, 512)
        self.assertEqual(self.extractor.config.num_layers, 2)
        self.assertEqual(self.extractor.config.num_heads, 4)
        self.assertEqual(self.extractor.config.hidden_dim, 1024)
        self.assertEqual(self.extractor.config.dropout, 0.1)
    
    def test_forward(self):
        """Test forward pass."""
        # Create mock activations
        activations = torch.randn(2, 10, 768)
        
        # Forward pass
        fingerprint = self.extractor(activations)
        
        # Check output shape
        self.assertEqual(fingerprint.shape, (2, 512))
    
    def test_extract_fingerprint(self):
        """Test extract_fingerprint method."""
        # Extract fingerprint
        fingerprint = self.extractor.extract_fingerprint(self.model, self.inputs)
        
        # Check output shape
        self.assertEqual(fingerprint.shape, (2, 512))
    
    def test_compute_similarity(self):
        """Test compute_similarity method."""
        # Create mock fingerprints
        fingerprint1 = torch.randn(512)
        fingerprint2 = torch.randn(512)
        
        # Compute similarity with default metric
        similarity = self.extractor.compute_similarity(fingerprint1, fingerprint2)
        
        # Check that similarity is a float
        self.assertIsInstance(similarity, float)
        
        # Compute similarity with custom metric
        kl_metric = KLDivergenceSimilarity()
        similarity = self.extractor.compute_similarity(fingerprint1, fingerprint2, kl_metric)
        
        # Check that similarity is a float
        self.assertIsInstance(similarity, float)
    
    def test_is_positive_pair(self):
        """Test is_positive_pair method."""
        # Create mock fingerprints
        fingerprint1 = torch.randn(512)
        fingerprint2 = torch.randn(512)
        
        # Test with similar task performance and fingerprints
        is_positive = self.extractor.is_positive_pair(0.9, 0.91, fingerprint1, fingerprint1)
        
        # Should be positive
        self.assertTrue(is_positive)
        
        # Test with different task performance
        is_positive = self.extractor.is_positive_pair(0.9, 0.7, fingerprint1, fingerprint1)
        
        # Should be negative
        self.assertFalse(is_positive)
    
    def test_save_load(self):
        """Test save and load methods."""
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        
        try:
            # Save extractor
            self.extractor.save(path)
            
            # Load extractor
            loaded_extractor = CapabilityExtractor.load(path)
            
            # Check that config is the same
            self.assertEqual(loaded_extractor.config.input_dim, self.config.input_dim)
            self.assertEqual(loaded_extractor.config.output_dim, self.config.output_dim)
            self.assertEqual(loaded_extractor.config.num_layers, self.config.num_layers)
            self.assertEqual(loaded_extractor.config.num_heads, self.config.num_heads)
            self.assertEqual(loaded_extractor.config.hidden_dim, self.config.hidden_dim)
            self.assertEqual(loaded_extractor.config.dropout, self.config.dropout)
            
            # Check that parameters are the same
            for p1, p2 in zip(self.extractor.parameters(), loaded_extractor.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
        
        finally:
            # Clean up
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()