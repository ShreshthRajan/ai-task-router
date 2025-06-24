#!/usr/bin/env python3
"""
Simple test runner for AI Task Router
"""

import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run the test suite."""
    print("🧪 Running AI Task Router Tests")
    print("=" * 40)
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_core.py", 
            "-v",
            "--tb=short"
        ], capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Some tests failed (exit code: {result.returncode})")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 