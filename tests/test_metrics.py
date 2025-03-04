"""
Tests for the similarity metrics.

This module contains tests for the similarity metrics.
"""

import unittest
import torch

from umaf.metrics import (
    CosineSimilarity,
    KLDivergenceSimilarity,
    PearsonCorrelation,
    CompositeSimilarity,
    TaskPerformanceSimilarity
)


class TestSimilarityMetrics(unittest.TestCase):
    """Tests for the similarity metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test tensors
        self.x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        self.z = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Same as x
    
    def test_cosine_similarity(self):
        """Test cosine similarity metric."""
        metric = CosineSimilarity()
        
        # Compute similarity between x and y
        similarity = metric.compute(self.x, self.y)
        
        # Check that similarity is a float
        self.assertIsInstance(similarity, float)
        
        # Check that similarity is between -1 and 1
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)
        
        # Check that similarity between x and z is 1.0
        similarity = metric.compute(self.x, self.z)
        self.assertAlmostEqual(similarity, 1.0)
        
        # Check that similarity between x and y is negative
        similarity = metric.compute(self.x, self.y)
        self.assertLess(similarity, 0.0)
        
        # Check that metric is callable
        similarity = metric(self.x, self.y)
        self.assertIsInstance(similarity, float)
    
    def test_kl_divergence_similarity(self):
        """Test KL divergence similarity metric."""
        metric = KLDivergenceSimilarity()
        
        # Compute similarity between x and y
        similarity = metric.compute(self.x, self.y)
        
        # Check that similarity is a float
        self.assertIsInstance(similarity, float)
        
        # Check that similarity is negative (KL divergence is always non-negative)
        self.assertLessEqual(similarity, 0.0)
        
        # Check that similarity between x and z is 0.0 (KL divergence is 0 for identical distributions)
        similarity = metric.compute(self.x, self.z)
        self.assertAlmostEqual(similarity, 0.0, places=5)
    
    def test_pearson_correlation(self):
        """Test Pearson correlation metric."""
        metric = PearsonCorrelation()
        
        # Compute similarity between x and y
        similarity = metric.compute(self.x, self.y)
        
        # Check that similarity is a float
        self.assertIsInstance(similarity, float)
        
        # Check that similarity is between -1 and 1
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)
        
        # Check that similarity between x and z is 1.0
        similarity = metric.compute(self.x, self.z)
        self.assertAlmostEqual(similarity, 1.0)
        
        # Check that similarity between x and y is -1.0
        similarity = metric.compute(self.x, self.y)
        self.assertAlmostEqual(similarity, -1.0)
    
    def test_composite_similarity(self):
        """Test composite similarity metric."""
        # Create component metrics
        cosine = CosineSimilarity()
        pearson = PearsonCorrelation()
        
        # Create composite metric
        metric = CompositeSimilarity([
            (cosine, 0.7),
            (pearson, 0.3)
        ])
        
        # Compute similarity between x and y
        similarity = metric.compute(self.x, self.y)
        
        # Check that similarity is a float
        self.assertIsInstance(similarity, float)
        
        # Check that similarity is between -1 and 1
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)
        
        # Check that similarity is a weighted average of component similarities
        cosine_sim = cosine.compute(self.x, self.y)
        pearson_sim = pearson.compute(self.x, self.y)
        expected_sim = 0.7 * cosine_sim + 0.3 * pearson_sim
        self.assertAlmostEqual(similarity, expected_sim)
    
    def test_task_performance_similarity(self):
        """Test task performance similarity metric."""
        metric = TaskPerformanceSimilarity(threshold=0.05)
        
        # Compute similarity between similar performances
        similarity = metric.compute(0.9, 0.91)
        
        # Check that similarity is a float
        self.assertIsInstance(similarity, float)
        
        # Check that similarity is between 0 and 1
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        
        # Check that similarity is high for similar performances
        self.assertGreaterEqual(similarity, 0.8)
        
        # Compute similarity between different performances
        similarity = metric.compute(0.9, 0.7)
        
        # Check that similarity is low for different performances
        self.assertLessEqual(similarity, 0.8)


if __name__ == '__main__':
    unittest.main()