"""
Tests for the utility functions.

This module contains tests for the utility functions.
"""

import unittest
import torch
import os
import tempfile

from umaf.utils import set_seed, save_config, load_config


class TestUtilityFunctions(unittest.TestCase):
    """Tests for the utility functions."""
    
    def test_set_seed(self):
        """Test set_seed function."""
        # Set seed
        set_seed(42)
        
        # Generate random tensors
        tensor1 = torch.randn(5, 5)
        
        # Set seed again
        set_seed(42)
        
        # Generate random tensors again
        tensor2 = torch.randn(5, 5)
        
        # Check that tensors are the same
        self.assertTrue(torch.allclose(tensor1, tensor2))
    
    def test_save_load_config(self):
        """Test save_config and load_config functions."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config
            config = {
                'input_dim': 768,
                'output_dim': 512,
                'num_layers': 4,
                'num_heads': 8,
                'hidden_dim': 2048,
                'dropout': 0.1
            }
            
            # Save config
            config_path = os.path.join(temp_dir, 'config.json')
            save_config(config, config_path)
            
            # Check that file exists
            self.assertTrue(os.path.exists(config_path))
            
            # Load config
            loaded_config = load_config(config_path)
            
            # Check that loaded config is the same as original
            self.assertEqual(loaded_config, config)
    
    def test_save_load_config_with_tensor(self):
        """Test save_config and load_config functions with tensor."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config with tensor
            config = {
                'input_dim': 768,
                'output_dim': 512,
                'tensor': torch.tensor([1.0, 2.0, 3.0])
            }
            
            # Save config
            config_path = os.path.join(temp_dir, 'config.json')
            save_config(config, config_path)
            
            # Check that file exists
            self.assertTrue(os.path.exists(config_path))
            
            # Load config
            loaded_config = load_config(config_path)
            
            # Check that loaded config has the same keys
            self.assertEqual(set(loaded_config.keys()), set(config.keys()))
            
            # Check that tensor was converted to list
            self.assertIsInstance(loaded_config['tensor'], list)
            self.assertEqual(loaded_config['tensor'], [1.0, 2.0, 3.0])


if __name__ == '__main__':
    unittest.main()