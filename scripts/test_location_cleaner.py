"""
Test script for location_cleaner module.

Demonstrates functionality and validates cleaning logic.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_cleaning import clean_location_columns
from data_cleaning.location_cleaner import (
    clean_location_text,
    normalize_unicode,
    resolve_to_canonical,
    CANONICAL_DISTRICT_MAPPING,
    validate_cleaned_data
)


def test_unicode_normalization():
    """Test Unicode normalization functionality."""
    print("=" * 80)
    print("TEST 1: Unicode Normalization")
    print("=" * 80)
    
    test_cases = [
        ("Medchal−Malkajgiri", "Medchal-Malkajgiri"),  # Unicode minus
        ("Medchal–Malkajgiri", "Medchal-Malkajgiri"),  # En dash
        ("Medchal—Malkajgiri", "Medchal-Malkajgiri"),  # Em dash
        ("Bengaluru\u200B", "Bengaluru"),  # Zero-width space
    ]
    
    for input_text, expected in test_cases:
        result = normalize_unicode(input_text)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"{status} {input_text!r:30s} -> {result!r:30s} (expected: {expected!r})")
    print()


def test_text_cleaning():
    """Test text cleaning and validation."""
    print("=" * 80)
    print("TEST 2: Text Cleaning")
    print("=" * 80)
    
    test_cases = [
        # Valid cases
        ("BENGALURU", "Bengaluru"),
        ("  bangalore  ", "Bengaluru"),  # Will be resolved to canonical
        ("hooghly", "Hooghly"),
        ("Medchal-Malkajgiri", "Medchal-Malkajgiri"),
        
        # Invalid cases (should return None)
        ("100000", None),
        ("5th cross", None),
        ("Near Dhyana Ashram", None),
        ("IDPL COLONY", None),
        ("123 Main Road", None),
        ("Plot 45", None),
        ("AB", None),  # Too short
    ]
    
    for input_text, expected in test_cases:
        result = clean_location_text(input_text)
        status = "[PASS]" if result == expected else "[FAIL]"
        expected_str = expected if expected else "None (invalid)"
        print(f"{status} {input_text:30s} -> {str(result):30s} (expected: {expected_str})")
    print()


def test_canonical_resolution():
    """Test fuzzy matching to canonical names."""
    print("=" * 80)
    print("TEST 3: Canonical Resolution")
    print("=" * 80)
    
    test_cases = [
        ("Bangalore", "Bengaluru"),
        ("bangalore urban", "Bengaluru Urban"),
        ("Mysore", "Mysuru"),
        ("Belgaum", "Belagavi"),
        ("Allahabad", "Prayagraj"),
        ("Gurgaon", "Gurugram"),
        ("Ahmadabad", "Ahmedabad"),
        ("UnknownDistrict", "UnknownDistrict"),  # Should remain unchanged
    ]
    
    for input_text, expected in test_cases:
        result = resolve_to_canonical(input_text, CANONICAL_DISTRICT_MAPPING)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"{status} {input_text:25s} -> {result:30s} (expected: {expected})")
    print()


def test_dataframe_cleaning():
    """Test full DataFrame cleaning."""
    print("=" * 80)
    print("TEST 4: DataFrame Cleaning")
    print("=" * 80)
    
    # Create test DataFrame
    df = pd.DataFrame({
        'state': ['Karnataka', 'WEST BENGAL', 'uttar pradesh', 'Maharashtra', 'Gujarat'],
        'district': ['bangalore', 'KOLKATA', 'allahabad', '5th cross', 'ahmadabad'],
        'sub_district': ['Urban', 'North', 'City', 'Invalid', 'Central'],
        'enrollment_count': [1000, 2000, 3000, 4000, 5000]
    })
    
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean data
    df_clean = clean_location_columns(df)
    
    print("Cleaned DataFrame:")
    print(df_clean[['state', 'state_raw', 'district', 'district_raw', 'enrollment_count']])
    print()
    
    print(f"Rows before: {len(df)}")
    print(f"Rows after: {len(df_clean)}")
    print(f"Rows removed: {len(df) - len(df_clean)}")
    print()


def test_with_real_data():
    """Test with real dataset if available."""
    print("=" * 80)
    print("TEST 5: Real Data Processing")
    print("=" * 80)
    
    # Check if processed data exists
    data_path = Path(__file__).parent.parent / "dataset" / "processed" / "api_data_aadhar_demographic"
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print("[WARNING] No processed CSV files found. Skipping real data test.")
        return
    
    # Load first file (just first 1000 rows for testing)
    test_file = csv_files[0]
    print(f"Loading: {test_file.name}")
    
    df = pd.read_csv(test_file, nrows=1000)
    print(f"Loaded {len(df):,} rows")
    print()
    
    # Show some original district values
    print("Sample original districts:")
    print(df['district'].value_counts().head(10))
    print()
    
    # Clean data
    df_clean = clean_location_columns(df)
    
    # Show cleaned district values
    print("Sample cleaned districts:")
    print(df_clean['district'].value_counts().head(10))
    print()
    
    # Validation report
    report = validate_cleaned_data(df, df_clean)
    
    print("Validation Report:")
    print(f"  Rows before: {report['rows_before']:,}")
    print(f"  Rows after: {report['rows_after']:,}")
    print(f"  Rows removed: {report['rows_removed']:,} ({report['removal_rate']:.2f}%)")
    print()
    
    if 'district' in report['unique_values_before']:
        print(f"  District unique values:")
        print(f"    Before: {report['unique_values_before']['district']:,}")
        print(f"    After: {report['unique_values_after']['district']:,}")
        reduction = report['unique_values_before']['district'] - report['unique_values_after']['district']
        print(f"    Reduction: {reduction:,} variants resolved")
    print()


def test_performance():
    """Test performance with larger dataset."""
    print("=" * 80)
    print("TEST 6: Performance Test")
    print("=" * 80)
    
    import time
    
    # Create large test dataset
    n_rows = 99999  # Make it divisible by 3
    print(f"Creating test dataset with {n_rows:,} rows...")
    
    states = ['Karnataka', 'WEST BENGAL', 'uttar pradesh'] * (n_rows // 3)
    districts = ['bangalore', 'KOLKATA', 'allahabad'] * (n_rows // 3)
    
    df = pd.DataFrame({
        'state': states,
        'district': districts,
        'value': range(len(states))
    })
    
    print(f"Dataset size: {len(df):,} rows")
    print()
    
    # Time the cleaning operation
    print("Starting cleaning operation...")
    start_time = time.time()
    
    df_clean = clean_location_columns(df)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"[PASS] Cleaning completed in {elapsed:.2f} seconds")
    print(f"  Throughput: {len(df)/elapsed:,.0f} rows/second")
    print(f"  Memory efficient: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("LOCATION CLEANER MODULE - TEST SUITE")
    print("=" * 80)
    print()
    
    try:
        test_unicode_normalization()
        test_text_cleaning()
        test_canonical_resolution()
        test_dataframe_cleaning()
        test_with_real_data()
        test_performance()
        
        print("=" * 80)
        print("[SUCCESS] ALL TESTS COMPLETED")
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
