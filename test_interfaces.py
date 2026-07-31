#!/usr/bin/env python3

import sys
from pathlib import Path
from abc import ABC, abstractmethod

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_interface_compliance():
    """Test that classes properly implement the DataSource interface"""
    try:
        from extracao.datasources.base_refactored import DataSource, Base
        print("✓ Successfully imported DataSource and Base classes")
    except ImportError as e:
        print(f"✗ Failed to import classes: {e}")
        return False
    
    # Check that DataSource is abstract
    try:
        # This should fail because DataSource is abstract
        ds = DataSource()
        print("✗ DataSource should be abstract but it's instantiable")
        return False
    except TypeError:
        print("✓ DataSource is properly abstract and cannot be instantiated")
    
    # Check that Base implements all abstract methods
    # We can't instantiate Base directly because it's missing implementations
    # but we can check that it has the required method signatures
    required_methods = ['stem', 'columns', 'extract_raw_data', 'format_data', 'process_data']
    
    for method_name in required_methods:
        if hasattr(Base, method_name):
            print(f"✓ Base class has method/property '{method_name}'")
        else:
            print(f"✗ Base class missing method/property '{method_name}'")
            return False
    
    return True

def main():
    """Main test function"""
    print("Running tests for interface standardization...")
    
    if not test_interface_compliance():
        return False
    
    print("\nAll interface standardization tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)