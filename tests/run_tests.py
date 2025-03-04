"""
Run all tests for the UMAF Capability Extractor.

This module runs all tests for the UMAF framework.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from test_extractor import TestCapabilityExtractor
from test_metrics import TestSimilarityMetrics
from test_processor import TestInputProcessor
from test_loss import TestLossFunctions
from test_dataset import TestDatasetClasses
from test_utils import TestUtilityFunctions
from test_monitor import TestCapabilityExtractionMonitor


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestCapabilityExtractor))
    test_suite.addTest(unittest.makeSuite(TestSimilarityMetrics))
    test_suite.addTest(unittest.makeSuite(TestInputProcessor))
    test_suite.addTest(unittest.makeSuite(TestLossFunctions))
    test_suite.addTest(unittest.makeSuite(TestDatasetClasses))
    test_suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    test_suite.addTest(unittest.makeSuite(TestCapabilityExtractionMonitor))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())