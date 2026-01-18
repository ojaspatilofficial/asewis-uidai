# Location Cleaner Module - Documentation

## Overview

The `location_cleaner` module provides robust, deterministic data cleaning for Aadhaar location data (state, district, sub_district) without modifying existing pipeline logic.

## Key Features

✅ **Rule-Based** - No ML models, fully explainable  
✅ **Deterministic** - Same input always produces same output  
✅ **Auditable** - Preserves original values in `*_raw` columns  
✅ **Performant** - Handles 1M+ rows efficiently (~20K rows/second)  
✅ **Unicode-Safe** - Handles various dash characters and invisible characters  
✅ **Fuzzy Matching** - Resolves spelling variants using rapidfuzz (90%+ similarity)  
✅ **Non-Invasive** - Adds preprocessing layer without changing downstream code  

## Installation

```bash
# Install required dependency
pip install rapidfuzz==3.6.1
```

## Quick Start

### Basic Usage

```python
from src.data_cleaning import clean_location_columns
import pandas as pd

# Load your data
df = pd.read_csv('dataset/api_data_aadhar_demographic_0_500000.csv')

# Clean location columns
df_clean = clean_location_columns(df)

# Original values preserved in *_raw columns
print(df_clean[['district', 'district_raw']].head())

# Save cleaned data
df_clean.to_csv('dataset/cleaned/demographic_clean.csv', index=False)
```

### Integration with Existing Pipeline

```python
# Add at the START of your pipeline, before any other processing

from src.data_loading import load_dataset
from src.data_cleaning import clean_location_columns

for chunk in load_dataset('dataset/api_data_aadhar_demographic', 'demographic'):
    # Clean location data (all other columns unchanged)
    chunk_clean = clean_location_columns(chunk)
    
    # Continue with your existing pipeline
    aggregated = aggregate_data(chunk_clean)
    features = engineer_features(chunk_clean)
    # etc.
```

## What Gets Fixed

### 1. Duplicate Spelling Variants

| Before | After |
|--------|-------|
| Bangalore | Bengaluru |
| bangalore urban | Bengaluru Urban |
| Mysore | Mysuru |
| Belgaum | Belagavi |
| Allahabad | Prayagraj |
| Gurgaon | Gurugram |
| Ahmadabad | Ahmedabad |
| KOLKATA | Kolkata |
| hooghly | Hooghly |

**Result**: 200+ canonical mappings applied

### 2. Invalid Location Entries (Removed)

| Invalid Entry | Reason |
|---------------|--------|
| 100000 | Numeric code |
| 5th cross | Street address |
| Near Dhyana Ashram | Landmark |
| IDPL COLONY | Colony name |
| Plot 45 | Address component |
| 123 Main Road | Street address |

**Result**: Invalid rows removed entirely

### 3. Unicode Issues

| Before | After |
|--------|-------|
| Medchal−Malkajgiri | Medchal-Malkajgiri |
| Medchal–Malkajgiri | Medchal-Malkajgiri |
| Medchal—Malkajgiri | Medchal-Malkajgiri |
| Bengaluru\u200B | Bengaluru |

**Result**: 10+ Unicode variants normalized

### 4. Case Inconsistencies

| Before | After |
|--------|-------|
| WEST BENGAL | West Bengal |
| uttar pradesh | Uttar Pradesh |
| KOLKATA | Kolkata |

**Result**: All names in proper Title Case

## Function Reference

### `clean_location_columns()`

Main cleaning function for location data.

```python
def clean_location_columns(
    df: pd.DataFrame,
    location_cols: List[str] = None,
    remove_invalid: bool = True,
    add_raw_columns: bool = True
) -> pd.DataFrame
```

**Parameters:**
- `df` - Input DataFrame with location columns
- `location_cols` - List of columns to clean (default: `['state', 'district', 'sub_district']`)
- `remove_invalid` - Remove rows with invalid locations (default: `True`)
- `add_raw_columns` - Preserve originals in `*_raw` columns (default: `True`)

**Returns:**
- Cleaned DataFrame with:
  - Original columns cleaned
  - `*_raw` columns added (if `add_raw_columns=True`)
  - Invalid rows removed (if `remove_invalid=True`)

**Example:**
```python
df_clean = clean_location_columns(
    df,
    location_cols=['state', 'district'],  # Only clean these
    remove_invalid=True,                   # Remove bad rows
    add_raw_columns=True                   # Keep originals
)
```

### `validate_cleaned_data()`

Generate cleaning quality report.

```python
def validate_cleaned_data(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    location_cols: List[str] = None
) -> Dict[str, any]
```

**Returns:**
- Dictionary with metrics:
  - `rows_before` - Original row count
  - `rows_after` - Cleaned row count
  - `rows_removed` - Rows removed
  - `removal_rate` - Percentage removed
  - `unique_values_before` - Unique values per column (before)
  - `unique_values_after` - Unique values per column (after)

**Example:**
```python
report = validate_cleaned_data(df_original, df_clean)
print(f"Removed {report['removal_rate']:.2f}% of rows")
print(f"Districts: {report['unique_values_before']['district']} → {report['unique_values_after']['district']}")
```

## Cleaning Rules

### Unicode Normalization
1. NFKD decomposition for compatibility
2. Replace 10+ Unicode hyphen variants with `-`
3. Remove zero-width spaces and invisible characters

### Text Cleaning
1. Keep only alphabetic characters, spaces, hyphens, parentheses
2. Collapse multiple spaces to single space
3. Strip leading/trailing whitespace
4. Apply Title Case for consistency

### Invalid Entry Detection

**Rejected if:**
- Purely numeric (e.g., "100000")
- Contains digits (e.g., "5th cross", "Plot 45")
- Length < 3 characters
- Contains keywords: cross, road, colony, near, plot, house, sector, temple, market, etc.
- Matches address patterns: "Near X", "Behind Y", "Opposite Z"

### Canonical Resolution

**Method**: Fuzzy string matching via rapidfuzz
- **Threshold**: 90% similarity required
- **Scorer**: Levenshtein ratio
- **Mapping**: 200+ official district/state mappings
- **Source**: Government of India Census 2011 + 2020-2025 administrative changes

## Performance Benchmarks

**Test Environment**: Python 3.10, Windows 11, 16GB RAM

| Dataset Size | Processing Time | Throughput |
|--------------|----------------|------------|
| 1,000 rows | 0.05 seconds | 20,000 rows/sec |
| 10,000 rows | 0.49 seconds | 20,400 rows/sec |
| 100,000 rows | 4.92 seconds | 20,325 rows/sec |
| 1,000,000 rows | ~50 seconds* | ~20,000 rows/sec |

*Estimated based on linear scaling

**Memory**: ~13.5 MB per 100K rows (very efficient)

## Test Results

```
TEST 1: Unicode Normalization - [PASS] 4/4
TEST 2: Text Cleaning - [PASS] 11/11  
TEST 3: Canonical Resolution - [PASS] 8/8
TEST 4: DataFrame Cleaning - [PASS]
  - 5 rows input → 4 rows output (1 invalid removed)
  - Bangalore → Bengaluru
  - Allahabad → Prayagraj
  - 5th cross → REMOVED

TEST 5: Real Data Processing - [PASS]
  - 1,000 rows processed
  - 0 rows removed (clean data)
  - 104 unique districts maintained

TEST 6: Performance Test - [PASS]
  - 99,999 rows in 4.92 seconds
  - 20,313 rows/second throughput
```

## Logging Output

The module logs detailed statistics during cleaning:

```
INFO - Starting location cleaning for 5 rows
INFO - Processing column: state
INFO -   - Marked 0 invalid entries as null
INFO -   - Resolved 2 variants to canonical names
INFO - Processing column: district
INFO -   - Marked 1 invalid entries as null
INFO -   - Resolved 3 variants to canonical names
INFO - Removed 1 rows with invalid locations (20.00%)
INFO - Cleaning complete: 4 rows retained

INFO - Top 10 corrections for 'state':
  west bengal                    -> West Bengal                     (1 times)
  uttar pradesh                  -> Uttar Pradesh                   (1 times)

INFO - Top 10 corrections for 'district':
  bangalore                      -> Bengaluru                       (1 times)
  kolkata                        -> Kolkata                         (1 times)
  allahabad                      -> Prayagraj                       (1 times)
```

## Advanced Usage

### Custom Column Selection

```python
# Only clean specific columns
df_clean = clean_location_columns(
    df,
    location_cols=['state', 'district']  # Skip sub_district
)
```

### Keep Invalid Rows (Mark Only)

```python
# Mark invalid as NaN but keep rows
df_clean = clean_location_columns(
    df,
    remove_invalid=False  # Keep all rows
)

# Later, you can filter manually
df_valid = df_clean[df_clean['district'].notna()]
df_invalid = df_clean[df_clean['district'].isna()]
```

### Batch Processing

```python
import glob
from pathlib import Path

input_dir = Path('dataset/raw')
output_dir = Path('dataset/cleaned')
output_dir.mkdir(exist_ok=True)

for file_path in input_dir.glob('*.csv'):
    print(f"Processing {file_path.name}...")
    
    df = pd.read_csv(file_path)
    df_clean = clean_location_columns(df)
    
    output_path = output_dir / file_path.name
    df_clean.to_csv(output_path, index=False)
    
    print(f"  {len(df):,} → {len(df_clean):,} rows")
```

### Chunked Processing for Very Large Files

```python
from src.data_loading import load_dataset

# Process in chunks to manage memory
all_chunks = []

for chunk in load_dataset('dataset/api_data_aadhar_demographic', 'demographic'):
    chunk_clean = clean_location_columns(chunk)
    all_chunks.append(chunk_clean)
    
    print(f"Processed chunk: {len(chunk_clean):,} rows")

# Combine all cleaned chunks
df_full = pd.concat(all_chunks, ignore_index=True)
```

## Canonical Mapping Sources

### Official Name Changes (Karnataka)
- Bangalore → Bengaluru (2014)
- Mysore → Mysuru (2014)
- Belgaum → Belagavi (2014)
- Gulbarga → Kalaburagi (2014)

### Official Name Changes (Other States)
- Allahabad → Prayagraj (UP, 2018)
- Faizabad → Ayodhya (UP, 2020)
- Gurgaon → Gurugram (Haryana, 2016)
- Aurangabad → Chhatrapati Sambhajinagar (MH, 2023)

### Regional Naming Conventions
- West Bengal: Purba/Paschim (Bengali) instead of East/West
- Bihar: Purbi/Pashchimi for Singhbhum districts
- Odisha: Official spellings per 2011 Census

## Troubleshooting

### Issue: Module not found

```bash
# Ensure you're in the correct directory
cd d:\HACKATHONS\UIDAI Hackathon\asewis

# Verify module structure
ls src/data_cleaning/
# Should show: __init__.py, location_cleaner.py
```

### Issue: rapidfuzz not installed

```bash
pip install rapidfuzz==3.6.1
```

### Issue: Too many rows removed

```python
# Check what's being removed
df_removed = df[~df.index.isin(df_clean.index)]
print(df_removed[['state', 'district']].value_counts())

# Adjust by keeping invalid rows
df_clean = clean_location_columns(df, remove_invalid=False)
```

### Issue: Canonical mapping incorrect

```python
# Check specific mapping
from src.data_cleaning.location_cleaner import CANONICAL_DISTRICT_MAPPING

print(CANONICAL_DISTRICT_MAPPING.get('bangalore'))
# Output: 'Bengaluru'

# Add custom mappings by editing location_cleaner.py
# Line 50+: CANONICAL_DISTRICT_MAPPING dictionary
```

## Files Created

```
src/data_cleaning/
├── __init__.py                 # Module initialization
└── location_cleaner.py         # Main cleaning logic (727 lines)

scripts/
└── test_location_cleaner.py    # Test suite (248 lines)

requirements.txt                # Updated with rapidfuzz dependency
```

## Next Steps

1. **Integrate with Pipeline**: Add `clean_location_columns()` at start of `run_pipeline.py`
2. **Batch Clean Existing Data**: Run batch processing script on all CSV files
3. **Monitor Corrections**: Review logging output for unexpected mappings
4. **Extend Mappings**: Add more canonical mappings as needed
5. **Validate Results**: Use `validate_cleaned_data()` on production data

---

**Module Version**: 1.0  
**Last Updated**: January 14, 2026  
**Maintainer**: ASEWIS Data Engineering Team
