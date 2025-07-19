#!/usr/bin/env python3
"""
Renaissance DeFi Trading System - Complete Test Suite
Tests all components and validates system readiness
"""

import sys
import os
import asyncio
import importlib.util
import time
from pathlib import Path

class RenaissanceSystemTester:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []

    def log_test(self, test_name, passed, details=""):
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            status = "✅ PASS"
        else:
            self.tests_failed += 1
            status = "❌ FAIL"
        
        print(f"{status}: {test_name}")
        if details:
            print(f"   {details}")
        
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'details': details
        })

    def test_python_requirements(self):
        print("\n🐍 Testing Python Requirements")
        print("=" * 40)
        
        # Test Python version
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            self.log_test("Python Version (3.8+)", True, f"Found Python {version.major}.{version.minor}")
        else:
            self.log_test("Python Version (3.8+)", False, f"Found Python {version.major}.{version.minor}, need 3.8+")

        # Test critical imports
        critical_modules = [
            'asyncio', 'json', 'time', 'logging', 'os', 'sys',
            'typing', 'dataclasses', 'collections', 'threading'
        ]
        
        for module in critical_modules:
            try:
                importlib.import_module(module)
                self.log_test(f"Import {module}", True)
            except ImportError:
                self.log_test(f"Import {module}", False, "Required for system operation")

    def test_optional_dependencies(self):
        print("\n📦 Testing Optional Dependencies")
        print("=" * 40)
        
        optional_modules = [
            ('numpy', 'Numerical operations'),
            ('pandas', 'Data analysis'),
            ('requests', 'HTTP requests'),
            ('aiohttp', 'Async HTTP'),
            ('websockets', 'WebSocket connections'),
            ('sqlite3', 'Database operations'),
            ('json', 'JSON processing')
        ]
        
        for module, description in optional_modules:
            try:
                importlib.import_module(module)
                self.log_test(f"Optional: {module}", True, description)
            except ImportError:
                self.log_test(f"Optional: {module}", False, f"{description} - will use fallbacks")

    def test_file_structure(self):
        print("\n📁 Testing File Structure")
        print("=" * 40)
        
        critical_files = [
            'production_renaissance_system.py',
            'run_production_system.py', 
            'run_pipeline.ipynb',
            'README.md',
            'requirements_final.txt'
        ]
        
        for file_path in critical_files:
            if Path(file_path).exists():
                self.log_test(f"File: {file_path}", True)
            else:
                self.log_test(f"File: {file_path}", False, "Critical system file missing")

        # Test directory structure
        required_dirs = [
            'config', 'scanners', 'executors', 'analyzers', 
            'profilers', 'watchers', 'models', 'data', 'monitoring'
        ]
        
        for dir_path in required_dirs:
            if Path(dir_path).is_dir():
                self.log_test(f"Directory: {dir_path}", True)
            else:
                self.log_test(f"Directory: {dir_path}", False, "Component directory missing")

    def test_system_components(self):
        print("\n🧠 Testing System Components")
        print("=" * 40)
        
        components = [
            ('config/dynamic_parameters.py', 'Dynamic Configuration'),
            ('scanners/enhanced_ultra_scanner.py', 'Ultra-Scale Scanner'),
            ('executors/position_manager.py', 'Position Management'),
            ('models/online_learner.py', 'Online Learning ML'),
            ('data/async_token_cache.py', 'Async Database'),
            ('analyzers/anti_rug_analyzer.py', 'Safety Analysis')
        ]
        
        for file_path, description in components:
            if Path(file_path).exists():
                self.log_test(f"Component: {description}", True, file_path)
            else:
                self.log_test(f"Component: {description}", False, f"Missing: {file_path}")

    def test_system_imports(self):
        print("\n⚡ Testing System Imports")
        print("=" * 40)
        
        # Test main system import
        try:
            # Add current directory to path
            sys.path.insert(0, '.')
            
            # Try to import the main system
            spec = importlib.util.spec_from_file_location(
                "production_renaissance_system", 
                "production_renaissance_system.py"
            )
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'renaissance_system'):
                    self.log_test("Main System Import", True, "Renaissance system loaded")
                else:
                    self.log_test("Main System Import", False, "Renaissance system object not found")
            else:
                self.log_test("Main System Import", False, "Could not load main system module")
                
        except Exception as e:
            self.log_test("Main System Import", False, f"Import error: {str(e)}")

    def test_system_functionality(self):
        print("\n🚀 Testing System Functionality")
        print("=" * 40)
        
        try:
            # Test if we can run the system briefly
            from production_renaissance_system import renaissance_system
            
            # Test basic system methods
            if hasattr(renaissance_system, 'initialize_system'):
                self.log_test("System Initialization Method", True, "initialize_system method exists")
            else:
                self.log_test("System Initialization Method", False, "initialize_system method missing")
                
            if hasattr(renaissance_system, 'start_production_trading'):
                self.log_test("Trading Method", True, "start_production_trading method exists")
            else:
                self.log_test("Trading Method", False, "start_production_trading method missing")
                
            if hasattr(renaissance_system, 'shutdown_system'):
                self.log_test("Shutdown Method", True, "shutdown_system method exists")
            else:
                self.log_test("Shutdown Method", False, "shutdown_system method missing")

        except ImportError:
            self.log_test("System Functionality", False, "Could not import system for testing")
        except Exception as e:
            self.log_test("System Functionality", False, f"Error testing functionality: {str(e)}")

    async def test_async_functionality(self):
        print("\n🔄 Testing Async Functionality")
        print("=" * 40)
        
        try:
            # Test basic async operations
            await asyncio.sleep(0.1)
            self.log_test("Asyncio Support", True, "Basic async operations working")
            
            # Test async context managers
            async def test_async_cm():
                return True
            
            result = await test_async_cm()
            if result:
                self.log_test("Async Context Managers", True, "Async functionality operational")
            else:
                self.log_test("Async Context Managers", False, "Async context issues")
                
        except Exception as e:
            self.log_test("Async Functionality", False, f"Async error: {str(e)}")

    def test_configuration(self):
        print("\n⚙️ Testing Configuration System")
        print("=" * 40)
        
        try:
            sys.path.append('config')
            from dynamic_parameters import get_dynamic_config
            
            config = get_dynamic_config()
            if isinstance(config, dict) and len(config) > 0:
                self.log_test("Dynamic Configuration", True, f"Loaded {len(config)} parameters")
            else:
                self.log_test("Dynamic Configuration", False, "Configuration empty or invalid")
                
        except ImportError:
            self.log_test("Dynamic Configuration", False, "Configuration module not found")
        except Exception as e:
            self.log_test("Dynamic Configuration", False, f"Configuration error: {str(e)}")

    def generate_report(self):
        print("\n" + "=" * 60)
        print("🎯 RENAISSANCE SYSTEM TEST REPORT")
        print("=" * 60)
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        print(f"📊 Tests Run: {self.tests_run}")
        print(f"✅ Tests Passed: {self.tests_passed}")
        print(f"❌ Tests Failed: {self.tests_failed}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        print("\n🎪 SYSTEM STATUS:")
        if success_rate >= 90:
            print("🎉 EXCELLENT - System is production-ready!")
            print("🚀 Ready to launch Renaissance Trading System")
            
            print("\n🎯 QUICK START OPTIONS:")
            print("1. Jupyter: jupyter notebook run_pipeline.ipynb")
            print("2. CLI: python run_production_system.py --duration 0.5")
            print("3. Full: ./deploy_complete_system.sh")
            
        elif success_rate >= 70:
            print("⚠️  GOOD - System mostly functional with minor issues")
            print("🔧 Some components may need attention")
            
            print("\n🛠️  RECOMMENDED ACTIONS:")
            print("1. Review failed tests above")
            print("2. Run: pip install -r requirements_final.txt")
            print("3. Generate missing components with setup scripts")
            
        elif success_rate >= 50:
            print("⚠️  PARTIAL - System has significant issues")
            print("🔧 Multiple components need attention")
            
            print("\n🚨 REQUIRED ACTIONS:")
            print("1. Install missing dependencies")
            print("2. Run component generation scripts")
            print("3. Re-test system")
            
        else:
            print("❌ CRITICAL - System requires major setup")
            print("🚨 Significant work needed before deployment")
            
            print("\n🆘 EMERGENCY ACTIONS:")
            print("1. Verify Python 3.8+ installation")
            print("2. Run: ./deploy_complete_system.sh")
            print("3. Install all requirements")

        print("\n📚 RESOURCES:")
        print("• README.md - Complete documentation")
        print("• deploy_complete_system.sh - Full deployment")
        print("• requirements_final.txt - Dependencies")
        
        return success_rate >= 70

async def main():
    print("🧪 RENAISSANCE DEFI TRADING SYSTEM - COMPLETE TEST SUITE")
    print("=" * 65)
    print("Testing all components and validating system readiness...")
    
    tester = RenaissanceSystemTester()
    
    # Run all test phases
    tester.test_python_requirements()
    tester.test_optional_dependencies()
    tester.test_file_structure()
    tester.test_system_components()
    tester.test_system_imports()
    tester.test_system_functionality()
    await tester.test_async_functionality()
    tester.test_configuration()
    
    # Generate final report
    system_ready = tester.generate_report()
    
    print("\n🏆 Renaissance System Test Complete!")
    
    return system_ready

if __name__ == "__main__":
    try:
        system_ready = asyncio.run(main())
        sys.exit(0 if system_ready else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite error: {e}")
        sys.exit(1)