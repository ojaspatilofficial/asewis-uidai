"""
Location data cleaner for Aadhaar datasets.

This module provides deterministic, rule-based cleaning for location columns
(state, district, sub_district) in Aadhaar enrollment data.

Key Features:
- Preserves original data in *_raw columns for auditability
- Removes invalid location entries (addresses, codes, landmarks)
- Normalizes Unicode and special characters
- Resolves spelling variants using fuzzy matching
- Fully vectorized for performance with 1M+ rows

Author: ASEWIS Data Engineering Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import logging
import unicodedata
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter

# Import fuzzy matching library
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logging.warning(
        "rapidfuzz not available. Install with: pip install rapidfuzz"
    )

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Canonical Location Mappings
# ============================================================================

# These mappings are derived from official Government of India sources
# and Census 2011 data, updated with recent administrative changes
CANONICAL_DISTRICT_MAPPING = {
    # Karnataka - Official name changes
    'bangalore': 'Bengaluru',
    'bangalore urban': 'Bengaluru Urban',
    'bangalore rural': 'Bengaluru Rural',
    'mysore': 'Mysuru',
    'belgaum': 'Belagavi',
    'gulbarga': 'Kalaburagi',
    'bellary': 'Ballari',
    'bijapur': 'Vijayapura',
    'shimoga': 'Shivamogga',
    'tumkur': 'Tumakuru',
    'chikmagalur': 'Chikkamagaluru',
    'chickmagalur': 'Chikkamagaluru',
    'davangere': 'Davanagere',
    'chamrajanagar': 'Chamarajanagar',
    'chamrajnagar': 'Chamarajanagar',
    
    # Maharashtra - Recent changes
    'ahmadnagar': 'Ahmednagar',
    'ahmed nagar': 'Ahmednagar',
    'ahilyanagar': 'Ahmednagar',
    'aurangabad': 'Chhatrapati Sambhajinagar',
    'osmanabad': 'Dharashiv',
    
    # Gujarat
    'ahmadabad': 'Ahmedabad',
    'mahesana': 'Mehsana',
    
    # Uttar Pradesh - Recent changes
    'allahabad': 'Prayagraj',
    'faizabad': 'Ayodhya',
    'jyotiba phule nagar': 'Amroha',
    'sant ravidas nagar': 'Sant Ravidas Nagar Bhadohi',
    'bhadohi': 'Sant Ravidas Nagar Bhadohi',
    'bara banki': 'Barabanki',
    'rae bareli': 'Raebareli',
    
    # Rajasthan
    'ganganagar': 'Sri Ganganagar',
    'chittaurgarh': 'Chittorgarh',
    'jhunjhunun': 'Jhunjhunu',
    
    # West Bengal - Regional naming
    'hooghiy': 'Hooghly',
    'hugli': 'Hooghly',
    'haora': 'Howrah',
    'hawrah': 'Howrah',
    'barddhaman': 'Bardhaman',
    'cooch behar': 'Cooch Behar',
    'koch bihar': 'Cooch Behar',
    'darjiling': 'Darjeeling',
    'north 24 parganas': 'North Twenty Four Parganas',
    'south 24 parganas': 'South Twenty Four Parganas',
    'south 24 pargana': 'South Twenty Four Parganas',
    'east midnapore': 'Purba Medinipur',
    'east midnapur': 'Purba Medinipur',
    'west medinipur': 'Paschim Medinipur',
    'west midnapore': 'Paschim Medinipur',
    'north dinajpur': 'Uttar Dinajpur',
    'south dinajpur': 'Dakshin Dinajpur',
    'dinajpur uttar': 'Uttar Dinajpur',
    
    # Odisha
    'baleshwar': 'Balasore',
    'baleswar': 'Balasore',
    'anugul': 'Angul',
    'anugal': 'Angul',
    'debagarh': 'Deogarh',
    'boudh': 'Baudh',
    'jajapur': 'Jajpur',
    'sundergarh': 'Sundargarh',
    'subarnapur': 'Sonepur',
    'sonapur': 'Sonepur',
    
    # Bihar - Regional naming
    'purnia': 'Purnea',
    'monghyr': 'Munger',
    'east champaran': 'Purba Champaran',
    'west champaran': 'Pashchim Champaran',
    
    # Jharkhand
    'east singhbhum': 'Purbi Singhbhum',
    'west singhbhum': 'Pashchimi Singhbhum',
    'seraikela-kharsawan': 'Seraikela-Kharsawan',
    
    # Telangana
    'rangareddi': 'Rangareddy',
    'k.v.rangareddy': 'Rangareddy',
    'k.v. rangareddy': 'Rangareddy',
    'medchal-malkajgiri': 'Medchal-Malkajgiri',
    'medchalmalkajgiri': 'Medchal-Malkajgiri',
    'warangal': 'Warangal Urban',
    'mahabub nagar': 'Mahabubnagar',
    'mahbubnagar': 'Mahabubnagar',
    'karim nagar': 'Karimnagar',
    'jangoan': 'Jangaon',
    
    # Andhra Pradesh
    'anantapur': 'Anantapuramu',
    'ananthapur': 'Anantapuramu',
    'cuddapah': 'YSR Kadapa',
    'nellore': 'Sri Potti Sriramulu Nellore',
    
    # Haryana
    'gurgaon': 'Gurugram',
    'mewat': 'Nuh',
    
    # Punjab
    'mohali': 'SAS Nagar',
    's.a.s nagar': 'SAS Nagar',
    'ferozepur': 'Firozpur',
    
    # Himachal Pradesh
    'lahaul and spiti': 'Lahaul And Spiti',
    'lahul spiti': 'Lahaul And Spiti',
    'lahul and spiti': 'Lahaul And Spiti',
    
    # Chhattisgarh
    'kabeerdham': 'Kabirdham',
    'kawardha': 'Kabirdham',
    'janjgir champa': 'Janjgir-Champa',
    
    # Delhi
    'central delhi': 'Central Delhi',
    'east delhi': 'East Delhi',
    'new delhi': 'New Delhi',
    'north delhi': 'North Delhi',
    'north east delhi': 'North East Delhi',
    'north west delhi': 'North West Delhi',
    'shahdara': 'Shahdara',
    'south delhi': 'South Delhi',
    'south east delhi': 'South East Delhi',
    'south west delhi': 'South West Delhi',
    'west delhi': 'West Delhi',
}

# State name standardization
CANONICAL_STATE_MAPPING = {
    'andaman and nicobar islands': 'Andaman And Nicobar Islands',
    'andaman nicobar': 'Andaman And Nicobar Islands',
    'andhra pradesh': 'Andhra Pradesh',
    'arunachal pradesh': 'Arunachal Pradesh',
    'assam': 'Assam',
    'bihar': 'Bihar',
    'chandigarh': 'Chandigarh',
    'chhattisgarh': 'Chhattisgarh',
    'dadra and nagar haveli': 'Dadra And Nagar Haveli',
    'dadra nagar haveli': 'Dadra And Nagar Haveli',
    'daman and diu': 'Daman And Diu',
    'daman diu': 'Daman And Diu',
    'delhi': 'Delhi',
    'goa': 'Goa',
    'gujarat': 'Gujarat',
    'haryana': 'Haryana',
    'himachal pradesh': 'Himachal Pradesh',
    'jammu and kashmir': 'Jammu And Kashmir',
    'jammu kashmir': 'Jammu And Kashmir',
    'jharkhand': 'Jharkhand',
    'karnataka': 'Karnataka',
    'kerala': 'Kerala',
    'ladakh': 'Ladakh',
    'lakshadweep': 'Lakshadweep',
    'madhya pradesh': 'Madhya Pradesh',
    'maharashtra': 'Maharashtra',
    'manipur': 'Manipur',
    'meghalaya': 'Meghalaya',
    'mizoram': 'Mizoram',
    'nagaland': 'Nagaland',
    'odisha': 'Odisha',
    'orissa': 'Odisha',
    'puducherry': 'Puducherry',
    'pondicherry': 'Puducherry',
    'punjab': 'Punjab',
    'rajasthan': 'Rajasthan',
    'sikkim': 'Sikkim',
    'tamil nadu': 'Tamil Nadu',
    'telangana': 'Telangana',
    'tripura': 'Tripura',
    'uttar pradesh': 'Uttar Pradesh',
    'uttarakhand': 'Uttarakhand',
    'uttaranchal': 'Uttarakhand',
    'west bengal': 'West Bengal',
}

# Invalid location patterns - these are NOT valid district/location names
INVALID_LOCATION_PATTERNS = [
    # Numeric codes
    r'^\d+$',  # Pure numbers like "100000"
    r'^\d{5,}$',  # 5+ digit codes
    
    # Address components
    r'\d+\s*(st|nd|rd|th)\s*cross',  # "5th cross"
    r'\d+\s*cross',  # "2 cross"
    r'\b(cross|road|street|lane|avenue|colony|sector|plot|house|building)\b',
    r'\bnear\s+',  # "near something"
    r'\b(beside|behind|opposite|adjacent)\b',
    
    # Landmarks (not administrative divisions)
    r'\b(temple|church|mosque|ashram|hospital|school|college|market|mall)\b',
    r'\b(station|airport|railway|bus stand)\b',
    r'\b(garden|park|nagar|layout)\s*\d*$',  # "XYZ Nagar" as suffix
    
    # Coordinates or codes
    r'\d+\.\d+',  # Decimal numbers (coordinates)
    
    # Too short or suspicious
    r'^.{1,2}$',  # Single/two character entries
]

# Words that indicate invalid locations
INVALID_LOCATION_WORDS = {
    'cross', 'road', 'street', 'lane', 'avenue', 'colony', 'sector',
    'plot', 'house', 'building', 'near', 'beside', 'behind', 'opposite',
    'adjacent', 'temple', 'church', 'mosque', 'ashram', 'hospital',
    'school', 'college', 'market', 'mall', 'station', 'airport',
    'railway', 'garden', 'park', 'idpl', 'layout', 'phase'
}


# ============================================================================
# Unicode and Text Normalization Functions
# ============================================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text to canonical form.
    
    Converts various Unicode representations to standard form:
    - NFKD decomposition for compatibility
    - Replaces Unicode hyphens/dashes with standard hyphen
    - Removes zero-width spaces and other invisible characters
    
    Args:
        text: Input text with potential Unicode issues
        
    Returns:
        Normalized text string
        
    Example:
        >>> normalize_unicode("Medchal−Malkajgiri")  # Unicode minus
        'Medchal-Malkajgiri'
    """
    if pd.isna(text):
        return None
    
    # Convert to string
    text = str(text)
    
    # Normalize Unicode to NFKD form (compatibility decomposition)
    text = unicodedata.normalize('NFKD', text)
    
    # Replace various Unicode hyphens/dashes with standard hyphen
    unicode_hyphens = [
        '\u2010',  # Hyphen
        '\u2011',  # Non-breaking hyphen
        '\u2012',  # Figure dash
        '\u2013',  # En dash
        '\u2014',  # Em dash
        '\u2015',  # Horizontal bar
        '\u2212',  # Minus sign
        '\uFE58',  # Small em dash
        '\uFE63',  # Small hyphen-minus
        '\uFF0D',  # Fullwidth hyphen-minus
    ]
    for dash in unicode_hyphens:
        text = text.replace(dash, '-')
    
    # Remove zero-width spaces and other invisible characters
    invisible_chars = [
        '\u200B',  # Zero-width space
        '\u200C',  # Zero-width non-joiner
        '\u200D',  # Zero-width joiner
        '\uFEFF',  # Zero-width no-break space
    ]
    for char in invisible_chars:
        text = text.replace(char, '')
    
    return text


def clean_location_text(text: str) -> Optional[str]:
    """
    Clean and normalize location text using deterministic rules.
    
    Processing steps:
    1. Unicode normalization
    2. Remove non-alphabetic characters (except spaces and hyphens)
    3. Collapse multiple spaces
    4. Strip leading/trailing whitespace
    5. Apply Title Case
    6. Validate against rejection rules
    
    Args:
        text: Raw location name
        
    Returns:
        Cleaned location name or None if invalid
        
    Example:
        >>> clean_location_text("  BENGALURU  ")
        'Bengaluru'
        >>> clean_location_text("5th cross")
        None
    """
    if pd.isna(text):
        return None
    
    # Normalize Unicode
    text = normalize_unicode(text)
    if not text:
        return None
    
    # Convert to lowercase for processing
    text_lower = text.lower().strip()
    
    # Check if purely numeric
    if text_lower.isdigit():
        return None
    
    # Check minimum length
    if len(text_lower) < 3:
        return None
    
    # Check for invalid patterns
    for pattern in INVALID_LOCATION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return None
    
    # Check for invalid words
    text_words = set(text_lower.replace('-', ' ').split())
    if text_words & INVALID_LOCATION_WORDS:
        return None
    
    # Keep only alphabetic characters, spaces, hyphens, and parentheses
    # Parentheses are kept for entries like "Aurangabad (MH)"
    text_cleaned = re.sub(r'[^a-zA-Z\s\-()]', '', text)
    
    # Collapse multiple spaces
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned)
    
    # Strip whitespace
    text_cleaned = text_cleaned.strip()
    
    # Check if anything remains
    if not text_cleaned or len(text_cleaned) < 3:
        return None
    
    # Apply Title Case
    text_cleaned = text_cleaned.title()
    
    return text_cleaned


# ============================================================================
# Fuzzy Matching for Canonical Resolution
# ============================================================================

def resolve_to_canonical(
    location: str,
    canonical_mapping: Dict[str, str],
    threshold: int = 90
) -> str:
    """
    Resolve location name to canonical form using fuzzy matching.
    
    Uses rapidfuzz for efficient fuzzy string matching against
    known canonical names. Falls back to exact mapping lookup.
    
    Args:
        location: Cleaned location name
        canonical_mapping: Dictionary of known variant -> canonical mappings
        threshold: Minimum similarity score (0-100) for fuzzy match
        
    Returns:
        Canonical location name or original if no match found
        
    Example:
        >>> resolve_to_canonical("bangalor", CANONICAL_DISTRICT_MAPPING)
        'Bengaluru'
    """
    if not location:
        return location
    
    location_lower = location.lower()
    
    # First, check exact mapping
    if location_lower in canonical_mapping:
        return canonical_mapping[location_lower]
    
    # If rapidfuzz available, try fuzzy matching
    if RAPIDFUZZ_AVAILABLE:
        # Search against all keys in canonical mapping
        keys = list(canonical_mapping.keys())
        
        # Use extractOne for best match
        result = process.extractOne(
            location_lower,
            keys,
            scorer=fuzz.ratio,
            score_cutoff=threshold
        )
        
        if result:
            matched_key, score, _ = result
            canonical_name = canonical_mapping[matched_key]
            
            # Log fuzzy matches for monitoring
            if score < 100:
                logger.debug(
                    f"Fuzzy match: '{location}' -> '{canonical_name}' "
                    f"(score: {score})"
                )
            
            return canonical_name
    
    # No match found, return original (cleaned) name
    return location


# ============================================================================
# Main Cleaning Function
# ============================================================================

def clean_location_columns(
    df: pd.DataFrame,
    location_cols: List[str] = None,
    remove_invalid: bool = True,
    add_raw_columns: bool = True
) -> pd.DataFrame:
    """
    Clean location columns in Aadhaar dataset.
    
    This function provides comprehensive location data cleaning:
    - Preserves original data in *_raw columns for auditability
    - Normalizes Unicode and special characters
    - Removes invalid entries (addresses, codes, landmarks)
    - Resolves spelling variants to canonical names
    - Fully vectorized for performance
    
    Args:
        df: Input DataFrame with location columns
        location_cols: List of column names to clean.
                      Defaults to ['state', 'district', 'sub_district']
        remove_invalid: If True, removes rows with invalid locations
        add_raw_columns: If True, preserves original values in *_raw columns
        
    Returns:
        Cleaned DataFrame with:
        - Original columns cleaned
        - *_raw columns added (if add_raw_columns=True)
        - Invalid rows removed (if remove_invalid=True)
        
    Example:
        >>> df_clean = clean_location_columns(df)
        >>> print(df_clean.columns)
        ['state', 'state_raw', 'district', 'district_raw', ...]
    """
    # Default location columns
    if location_cols is None:
        location_cols = ['state', 'district', 'sub_district']
    
    # Only process columns that exist in DataFrame
    location_cols = [col for col in location_cols if col in df.columns]
    
    if not location_cols:
        logger.warning("No location columns found in DataFrame")
        return df
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Track statistics
    stats = {
        'rows_before': len(df_clean),
        'rows_removed': 0,
        'corrections': {}
    }
    
    logger.info(f"Starting location cleaning for {stats['rows_before']:,} rows")
    
    # Process each location column
    for col in location_cols:
        logger.info(f"Processing column: {col}")
        
        # Preserve original values
        if add_raw_columns:
            df_clean[f'{col}_raw'] = df_clean[col].copy()
        
        # Apply cleaning function vectorized
        df_clean[col] = df_clean[col].apply(clean_location_text)
        
        # Count nulls introduced by cleaning (invalid entries)
        invalid_count = df_clean[col].isna().sum() - df[col].isna().sum()
        if invalid_count > 0:
            logger.info(f"  - Marked {invalid_count:,} invalid entries as null")
        
        # Apply canonical mapping based on column type
        if col == 'state':
            mapping = CANONICAL_STATE_MAPPING
        elif col == 'district':
            mapping = CANONICAL_DISTRICT_MAPPING
        else:
            mapping = {}  # No canonical mapping for sub_district
        
        if mapping:
            # Count corrections
            before_resolution = df_clean[col].copy()
            
            # Apply fuzzy matching
            df_clean[col] = df_clean[col].apply(
                lambda x: resolve_to_canonical(x, mapping) if pd.notna(x) else x
            )
            
            # Track what was corrected
            corrections = before_resolution != df_clean[col]
            correction_count = corrections.sum()
            
            if correction_count > 0:
                logger.info(f"  - Resolved {correction_count:,} variants to canonical names")
                
                # Track top corrections for this column
                correction_pairs = [
                    (before_resolution[i], df_clean[col].iloc[i])
                    for i in corrections[corrections].index
                ]
                stats['corrections'][col] = Counter(correction_pairs)
    
    # Remove rows with invalid locations if requested
    if remove_invalid:
        # A row is invalid if ANY location column is null after cleaning
        valid_mask = df_clean[location_cols].notna().all(axis=1)
        rows_before_removal = len(df_clean)
        df_clean = df_clean[valid_mask].reset_index(drop=True)
        stats['rows_removed'] = rows_before_removal - len(df_clean)
        
        if stats['rows_removed'] > 0:
            logger.info(
                f"Removed {stats['rows_removed']:,} rows with invalid locations "
                f"({stats['rows_removed']/stats['rows_before']*100:.2f}%)"
            )
    
    # Log final statistics
    stats['rows_after'] = len(df_clean)
    logger.info(f"Cleaning complete: {stats['rows_after']:,} rows retained")
    
    # Log top corrections for each column
    for col, corrections in stats['corrections'].items():
        if corrections:
            logger.info(f"\nTop 10 corrections for '{col}':")
            for (original, corrected), count in corrections.most_common(10):
                # Handle None values in logging
                orig_str = str(original) if original is not None else "None"
                corr_str = str(corrected) if corrected is not None else "None"
                logger.info(f"  {orig_str:30s} -> {corr_str:30s} ({count:,} times)")
    
    return df_clean


# ============================================================================
# Validation and Quality Checks
# ============================================================================

def validate_cleaned_data(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    location_cols: List[str] = None
) -> Dict[str, any]:
    """
    Validate cleaned data quality and generate report.
    
    Args:
        df_original: Original DataFrame before cleaning
        df_cleaned: Cleaned DataFrame after processing
        location_cols: List of location columns to check
        
    Returns:
        Dictionary with validation metrics:
        - rows_before: Original row count
        - rows_after: Cleaned row count
        - rows_removed: Number of rows removed
        - removal_rate: Percentage of rows removed
        - unique_values_before: Unique values per column (before)
        - unique_values_after: Unique values per column (after)
        - null_count_before: Null counts per column (before)
        - null_count_after: Null counts per column (after)
    """
    if location_cols is None:
        location_cols = ['state', 'district', 'sub_district']
    
    location_cols = [col for col in location_cols if col in df_original.columns]
    
    report = {
        'rows_before': len(df_original),
        'rows_after': len(df_cleaned),
        'rows_removed': len(df_original) - len(df_cleaned),
        'unique_values_before': {},
        'unique_values_after': {},
        'null_count_before': {},
        'null_count_after': {}
    }
    
    report['removal_rate'] = (
        report['rows_removed'] / report['rows_before'] * 100
        if report['rows_before'] > 0 else 0
    )
    
    for col in location_cols:
        report['unique_values_before'][col] = df_original[col].nunique()
        report['unique_values_after'][col] = df_cleaned[col].nunique()
        report['null_count_before'][col] = df_original[col].isna().sum()
        report['null_count_after'][col] = df_cleaned[col].isna().sum()
    
    return report


# ============================================================================
# Example Usage (Commented)
# ============================================================================

"""
# Example 1: Basic usage
import pandas as pd
from src.data_cleaning import clean_location_columns

# Load data
df = pd.read_csv('dataset/api_data_aadhar_demographic_0_500000.csv')

# Clean location columns
df_clean = clean_location_columns(df)

# Original values are preserved in *_raw columns
print(df_clean[['district', 'district_raw']].head())

# Save cleaned data
df_clean.to_csv('dataset/cleaned/demographic_clean.csv', index=False)


# Example 2: Custom columns and options
df_clean = clean_location_columns(
    df,
    location_cols=['state', 'district'],  # Only clean these columns
    remove_invalid=True,  # Remove invalid rows
    add_raw_columns=True  # Keep original values
)


# Example 3: Validate cleaning quality
from src.data_cleaning.location_cleaner import validate_cleaned_data

report = validate_cleaned_data(df, df_clean)
print(f"Removed {report['removal_rate']:.2f}% of rows")
print(f"District unique values: {report['unique_values_before']['district']} -> {report['unique_values_after']['district']}")


# Example 4: Process multiple files
import glob

for file_path in glob.glob('dataset/*.csv'):
    df = pd.read_csv(file_path)
    df_clean = clean_location_columns(df)
    
    output_path = file_path.replace('dataset/', 'dataset/cleaned/')
    df_clean.to_csv(output_path, index=False)
    
    print(f"Cleaned {file_path}: {len(df)} -> {len(df_clean)} rows")


# Example 5: Integration with existing pipeline
# Add this at the START of your data pipeline, before any other processing

from src.data_loading import load_dataset
from src.data_cleaning import clean_location_columns

# Load raw data
for chunk in load_dataset('dataset/api_data_aadhar_demographic', 'demographic'):
    # Clean location data (preserves all other columns)
    chunk_clean = clean_location_columns(chunk)
    
    # Continue with existing pipeline
    # aggregated = aggregate_data(chunk_clean)
    # features = engineer_features(chunk_clean)
    # etc.
"""
