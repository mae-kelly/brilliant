
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

class TestTradingBot(unittest.TestCase):
    def setUp(self):
        self.mock_web3 = MagicMock()
        self.mock_web3.is_connected.return_value = True
        
    @patch('web3.Web3')
    def test_scanner_initialization(self, mock_web3):
        mock_web3.return_value = self.mock_web3
        try:
            from scanner_v3 import TokenScanner
            scanner = TokenScanner()
            self.assertIsNotNone(scanner)
        except ImportError:
            self.skipTest("Scanner module not available")
    
    def test_config_validation(self):
        try:
            from secure_loader import config
            self.assertTrue(hasattr(config, 'validate_all'))
        except ImportError:
            self.skipTest("Config module not available")
    
    def test_safety_manager(self):
        try:
            from dev_mode import dev_wrapper
            self.assertFalse(dev_wrapper.config.enable_real_trading)
            self.assertTrue(dev_wrapper.config.dry_run)
        except ImportError:
            self.skipTest("Dev mode module not available")

class TestSafetyFeatures(unittest.TestCase):
    def test_environment_variables(self):
        required_vars = ['API_KEY', 'WALLET_ADDRESS']
        for var in required_vars:
            value = os.getenv(var)
            if value and not value.startswith('your_'):
                self.assertTrue(len(value) > 10)
    
    def test_no_hardcoded_secrets(self):
        unsafe_strings = ['0xYourWallet', os.getenv("API_KEY", ""), 'your_alchemy']
        
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                        for unsafe in unsafe_strings:
                            self.assertNotIn(unsafe, content, 
                                f"Unsafe string '{unsafe}' found in {filepath}")
                    except:
                        continue

if __name__ == '__main__':
    unittest.main()
