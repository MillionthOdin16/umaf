"""
Tests for the CapabilityExtractionMonitor class.

This module contains tests for the CapabilityExtractionMonitor class.
"""

import unittest
import os
import tempfile
import json

from umaf.monitor import CapabilityExtractionMonitor


class TestCapabilityExtractionMonitor(unittest.TestCase):
    """Tests for the CapabilityExtractionMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = CapabilityExtractionMonitor()
    
    def test_log_metrics(self):
        """Test log_metrics method."""
        # Log metrics
        self.monitor.log_metrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            learning_rate=0.001
        )
        
        # Check that metrics were logged
        self.assertEqual(self.monitor.metrics['epoch'], [1])
        self.assertEqual(self.monitor.metrics['train_loss'], [0.5])
        self.assertEqual(self.monitor.metrics['val_loss'], [0.6])
        self.assertEqual(self.monitor.metrics['learning_rate'], [0.001])
        
        # Log more metrics
        self.monitor.log_metrics(
            epoch=2,
            train_loss=0.4,
            val_loss=0.5,
            learning_rate=0.0005
        )
        
        # Check that metrics were appended
        self.assertEqual(self.monitor.metrics['epoch'], [1, 2])
        self.assertEqual(self.monitor.metrics['train_loss'], [0.5, 0.4])
        self.assertEqual(self.monitor.metrics['val_loss'], [0.6, 0.5])
        self.assertEqual(self.monitor.metrics['learning_rate'], [0.001, 0.0005])
    
    def test_generate_report(self):
        """Test generate_report method."""
        # Log metrics
        self.monitor.log_metrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            learning_rate=0.001
        )
        
        self.monitor.log_metrics(
            epoch=2,
            train_loss=0.4,
            val_loss=0.5,
            learning_rate=0.0005
        )
        
        # Generate report
        report = self.monitor.generate_report()
        
        # Check that report contains metrics
        self.assertIn('epoch', report)
        self.assertIn('train_loss', report)
        self.assertIn('val_loss', report)
        self.assertIn('learning_rate', report)
        
        # Check that report contains statistics
        self.assertIn('mean', report['train_loss'])
        self.assertIn('std', report['train_loss'])
        self.assertIn('min', report['train_loss'])
        self.assertIn('max', report['train_loss'])
        self.assertIn('last', report['train_loss'])
        
        # Check that statistics are correct
        self.assertEqual(report['train_loss']['mean'], 0.45)
        self.assertEqual(report['train_loss']['min'], 0.4)
        self.assertEqual(report['train_loss']['max'], 0.5)
        self.assertEqual(report['train_loss']['last'], 0.4)
    
    def test_log_dir(self):
        """Test log_dir functionality."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create monitor with log_dir
            monitor = CapabilityExtractionMonitor(log_dir=temp_dir)
            
            # Log metrics
            monitor.log_metrics(
                epoch=1,
                train_loss=0.5,
                val_loss=0.6,
                learning_rate=0.001
            )
            
            # Check that metrics file exists
            metrics_file = os.path.join(temp_dir, 'metrics.json')
            self.assertTrue(os.path.exists(metrics_file))
            
            # Check that metrics file contains correct data
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            self.assertEqual(metrics['epoch'], [1])
            self.assertEqual(metrics['train_loss'], [0.5])
            self.assertEqual(metrics['val_loss'], [0.6])
            self.assertEqual(metrics['learning_rate'], [0.001])
            
            # Generate report
            report = monitor.generate_report()
            
            # Check that report file exists
            report_file = os.path.join(temp_dir, 'report.json')
            self.assertTrue(os.path.exists(report_file))
            
            # Check that report file contains correct data
            with open(report_file, 'r') as f:
                loaded_report = json.load(f)
            
            self.assertEqual(loaded_report['train_loss']['mean'], 0.5)
            self.assertEqual(loaded_report['val_loss']['mean'], 0.6)
            self.assertEqual(loaded_report['learning_rate']['mean'], 0.001)


if __name__ == '__main__':
    unittest.main()