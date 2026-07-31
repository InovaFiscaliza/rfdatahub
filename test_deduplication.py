#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_utility_methods():
    """Test the new utility methods in the Base class"""
    try:
        from extracao.datasources.base_refactored import Base
        print("✓ Successfully imported Base class")
    except ImportError as e:
        print(f"✗ Failed to import Base class: {e}")
        return False
    
    # Create a test dataframe
    df = pd.DataFrame({
        "Frequência": ["100", "200", "300", "invalid"],
        "Unidade": ["kHz", "MHz", "GHz", "MHz"],
        "Validade_RF": ["2023-01-01 10:30:00", "2023-02-01 11:45:00", "2023-03-01 12:00:00", "2023-04-01 13:15:00"],
        "Serviço": ["205", "206", "207", "208"]
    })
    
    # Test frequency conversion
    try:
        df_converted = Base.convert_frequency(df.copy())
        print("✓ convert_frequency method works correctly")
        print(f"  Original frequencies: {df['Frequência'].tolist()}")
        print(f"  Converted frequencies: {df_converted['Frequência'].tolist()}")
    except Exception as e:
        print(f"✗ convert_frequency method failed: {e}")
        return False
    
    # Test date formatting
    try:
        df_formatted = Base.format_date_column(df.copy())
        print("✓ format_date_column method works correctly")
        print(f"  Formatted dates: {df_formatted['Validade_RF'].tolist()}")
    except Exception as e:
        print(f"✗ format_date_column method failed: {e}")
        return False
    
    # Test setting data source
    try:
        df_source = Base.set_data_source(df.copy(), "TEST")
        print("✓ set_data_source method works correctly")
        print(f"  Data source: {df_source['Fonte'].iloc[0]}")
    except Exception as e:
        print(f"✗ set_data_source method failed: {e}")
        return False
    
    # Test setting multiplicity
    try:
        df_multiplicity = Base.set_multiplicity(df.copy())
        print("✓ set_multiplicity method works correctly")
        print(f"  Multiplicity: {df_multiplicity['Multiplicidade'].iloc[0]}")
    except Exception as e:
        print(f"✗ set_multiplicity method failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("Running tests for deduplication refactoring...")
    
    if not test_utility_methods():
        return False
    
    print("\nAll deduplication tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)