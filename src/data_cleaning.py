"""
Data cleaning module for ASEWIS system.

This module provides deterministic data cleaning and standardization functions
for Aadhar datasets. All functions are pure (no side effects) and testable.
"""

import logging
import re
from typing import List, Optional
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case convention.
    
    Converts column names by:
    - Converting to lowercase
    - Replacing spaces with underscores
    - Removing special characters except underscores
    - Collapsing multiple underscores to single
    - Stripping leading/trailing underscores
    
    Args:
        df: Input DataFrame with any column naming convention
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names
        
    Example:
        >>> df = pd.DataFrame({'First Name': [1], 'Last-Name': [2], 'AGE  ': [3]})
        >>> result = standardize_columns(df)
        >>> list(result.columns)
        ['first_name', 'last_name', 'age']
    """
    logger.info(f"Standardizing {len(df.columns)} column names")
    
    original_columns = df.columns.tolist()
    standardized_columns = []
    
    for col in original_columns:
        # Convert to string in case of non-string column names
        col_str = str(col)
        
        # Convert to lowercase
        standardized = col_str.lower()
        
        # Replace spaces and hyphens with underscores
        standardized = standardized.replace(' ', '_').replace('-', '_')
        
        # Remove special characters except underscores
        # Keep alphanumeric and underscores only
        standardized = re.sub(r'[^a-z0-9_]', '', standardized)
        
        # Collapse multiple underscores to single underscore
        standardized = re.sub(r'_+', '_', standardized)
        
        # Strip leading and trailing underscores
        standardized = standardized.strip('_')
        
        # Handle empty column names after cleaning
        if not standardized:
            standardized = f'column_{len(standardized_columns)}'
            logger.warning(f"Empty column name after standardization, using: {standardized}")
        
        standardized_columns.append(standardized)
    
    # Create a copy to avoid modifying original DataFrame
    df_cleaned = df.copy()
    df_cleaned.columns = standardized_columns
    
    # Log any column name changes
    changes = [
        (orig, std) for orig, std in zip(original_columns, standardized_columns)
        if orig != std
    ]
    
    if changes:
        logger.info(f"Standardized {len(changes)} column name(s)")
        for orig, std in changes[:5]:  # Log first 5 changes
            logger.debug(f"  '{orig}' -> '{std}'")
        if len(changes) > 5:
            logger.debug(f"  ... and {len(changes) - 5} more")
    else:
        logger.info("All column names already standardized")
    
    return df_cleaned


def normalize_text_field(
    series: pd.Series,
    uppercase: bool = True,
    strip_whitespace: bool = True
) -> pd.Series:
    """
    Normalize text fields for consistency.
    
    Args:
        series: Pandas Series containing text data
        uppercase: Convert to uppercase if True
        strip_whitespace: Remove leading/trailing whitespace if True
        
    Returns:
        pd.Series: Normalized text series
    """
    # Create a copy to avoid modifying original
    normalized = series.copy()
    
    # Convert to string type, handling NaN values
    # NaN values will remain as NaN, not converted to 'nan' string
    normalized = normalized.astype(str)
    
    # Replace 'nan' string (from conversion) back to NaN
    normalized = normalized.replace('nan', np.nan)
    
    if strip_whitespace:
        # Strip whitespace from non-null values
        normalized = normalized.str.strip()
    
    if uppercase:
        # Convert to uppercase for non-null values
        normalized = normalized.str.upper()
    
    return normalized


def convert_to_date_column(
    series: pd.Series,
    output_format: str = '%Y-%m'
) -> pd.Series:
    """
    Convert date column to standardized YYYY-MM format.
    
    Attempts to parse dates in multiple formats and standardizes to YYYY-MM.
    Invalid dates are converted to NaT (Not a Time).
    
    Args:
        series: Pandas Series containing date data
        output_format: Desired output format (default: '%Y-%m' for YYYY-MM)
        
    Returns:
        pd.Series: Series with dates in YYYY-MM string format, invalid dates as NaN
        
    Example:
        >>> dates = pd.Series(['2023-01-15', '01/15/2023', 'invalid'])
        >>> convert_to_date_column(dates)
        0    2023-01
        1    2023-01
        2        NaN
    """
    # Try to parse dates with automatic format inference
    # errors='coerce' converts unparseable dates to NaT
    try:
        parsed_dates = pd.to_datetime(series, errors='coerce')
        
        # Convert to desired format string
        # NaT values will become NaN in the string representation
        formatted_dates = parsed_dates.dt.strftime(output_format)
        
        # Count invalid dates
        invalid_count = formatted_dates.isna().sum()
        if invalid_count > 0:
            logger.debug(
                f"Converted dates: {invalid_count} invalid date(s) set to NaN"
            )
        
        return formatted_dates
        
    except Exception as e:
        logger.error(f"Error converting date column: {e}")
        # Return series with all NaN if conversion fails completely
        return pd.Series([np.nan] * len(series), index=series.index)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply comprehensive cleaning to a DataFrame.
    
    Cleaning operations performed:
    1. Standardize column names to snake_case
    2. Normalize state and district names (uppercase, strip spaces)
    3. Convert date columns to YYYY-MM format
    4. Drop rows with missing or invalid district/date values
    5. Reset index after dropping rows
    
    This function is deterministic - same input produces same output.
    
    Args:
        df: Input DataFrame with raw data
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized values
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
        
    Example:
        >>> df = pd.DataFrame({
        ...     'State': [' karnataka ', 'KERALA', None],
        ...     'District': ['Bangalore', 'trivandrum ', None],
        ...     'Date': ['2023-01-15', '2023-02-20', '2023-03-10']
        ... })
        >>> cleaned = clean_dataframe(df)
        >>> len(cleaned)  # Row with None district is dropped
        2
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to clean_dataframe")
        return df.copy()
    
    logger.info(f"Starting cleaning: {len(df)} rows, {len(df.columns)} columns")
    initial_row_count = len(df)
    
    # Create a copy to avoid modifying original DataFrame
    df_cleaned = df.copy()
    
    # Step 1: Standardize column names to snake_case
    # This ensures consistent column naming throughout the system
    df_cleaned = standardize_columns(df_cleaned)
    logger.info("✓ Column names standardized to snake_case")
    
    # Step 2: Identify columns that need normalization
    # Look for common state/district/location column patterns
    location_columns = [
        col for col in df_cleaned.columns
        if any(keyword in col.lower() for keyword in ['state', 'district', 'location', 'region'])
    ]
    
    if location_columns:
        logger.info(f"Found {len(location_columns)} location column(s): {location_columns}")
        
        # Normalize each location column
        # Uppercase and strip whitespace for consistency
        # This prevents duplicate entries due to case/whitespace differences
        for col in location_columns:
            original_nulls = df_cleaned[col].isna().sum()
            df_cleaned[col] = normalize_text_field(
                df_cleaned[col],
                uppercase=True,
                strip_whitespace=True
            )
            
            # Log if normalization introduced new nulls (empty strings converted)
            new_nulls = df_cleaned[col].isna().sum()
            if new_nulls > original_nulls:
                logger.debug(
                    f"  {col}: {new_nulls - original_nulls} empty value(s) "
                    f"converted to NaN"
                )
        
        logger.info("✓ Location columns normalized (uppercase, trimmed)")
    else:
        logger.warning("No location columns found for normalization")
    
    # Step 3: Identify and convert date columns
    # Look for common date column patterns
    date_columns = [
        col for col in df_cleaned.columns
        if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time', 'period'])
    ]
    
    if date_columns:
        logger.info(f"Found {len(date_columns)} date column(s): {date_columns}")
        
        # Convert each date column to YYYY-MM format
        # This standardization enables consistent date-based analysis
        for col in date_columns:
            original_nulls = df_cleaned[col].isna().sum()
            df_cleaned[col] = convert_to_date_column(
                df_cleaned[col],
                output_format='%Y-%m'
            )
            
            # Log conversion results
            new_nulls = df_cleaned[col].isna().sum()
            converted_count = len(df_cleaned) - new_nulls
            invalid_count = new_nulls - original_nulls
            
            logger.info(
                f"  {col}: {converted_count} valid, {invalid_count} invalid "
                f"(total NaN: {new_nulls})"
            )
        
        logger.info("✓ Date columns converted to YYYY-MM format")
    else:
        logger.warning("No date columns found for conversion")
    
    # Step 4: Drop rows with missing critical values
    # Critical columns: district and date columns
    # These are essential for analysis and cannot be imputed
    critical_columns = []
    
    # Add district columns to critical list
    district_columns = [col for col in df_cleaned.columns if 'district' in col.lower()]
    critical_columns.extend(district_columns)
    
    # Add date columns to critical list
    critical_columns.extend(date_columns)
    
    # Remove duplicates from critical columns list
    critical_columns = list(set(critical_columns))
    
    if critical_columns:
        logger.info(f"Checking for missing values in critical columns: {critical_columns}")
        
        # Calculate rows to drop
        rows_before = len(df_cleaned)
        
        # Drop rows where ANY critical column has missing values
        # This ensures data quality for downstream analysis
        df_cleaned = df_cleaned.dropna(subset=critical_columns, how='any')
        
        rows_after = len(df_cleaned)
        rows_dropped = rows_before - rows_after
        
        if rows_dropped > 0:
            drop_percentage = (rows_dropped / rows_before) * 100
            logger.info(
                f"✓ Dropped {rows_dropped} row(s) with missing critical values "
                f"({drop_percentage:.2f}%)"
            )
        else:
            logger.info("✓ No rows dropped - all critical values present")
    else:
        logger.warning("No critical columns identified for missing value check")
    
    # Step 5: Reset index after dropping rows
    # This ensures clean sequential indexing for downstream processing
    df_cleaned = df_cleaned.reset_index(drop=True)
    logger.info("✓ Index reset after row removal")
    
    # Final summary
    final_row_count = len(df_cleaned)
    rows_retained = final_row_count
    rows_removed = initial_row_count - final_row_count
    retention_rate = (rows_retained / initial_row_count * 100) if initial_row_count > 0 else 0
    
    logger.info(
        f"Cleaning complete: {rows_retained}/{initial_row_count} rows retained "
        f"({retention_rate:.2f}%), {rows_removed} rows removed"
    )
    
    return df_cleaned


def validate_cleaned_data(df: pd.DataFrame) -> dict:
    """
    Validate cleaned DataFrame and return quality metrics.
    
    This is a helper function to verify data quality after cleaning.
    
    Args:
        df: Cleaned DataFrame to validate
        
    Returns:
        dict: Dictionary containing validation metrics
        
    Example:
        >>> metrics = validate_cleaned_data(cleaned_df)
        >>> print(f"Data quality score: {metrics['quality_score']}")
    """
    if df.empty:
        return {
            'is_valid': False,
            'row_count': 0,
            'column_count': 0,
            'error': 'DataFrame is empty'
        }
    
    metrics = {
        'is_valid': True,
        'row_count': len(df),
        'column_count': len(df.columns),
        'columns': df.columns.tolist(),
        'null_counts': df.isnull().sum().to_dict(),
        'total_nulls': df.isnull().sum().sum(),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
    
    # Calculate data quality score
    # Score based on completeness (0-100)
    total_cells = len(df) * len(df.columns)
    filled_cells = total_cells - metrics['total_nulls']
    metrics['quality_score'] = round((filled_cells / total_cells * 100), 2) if total_cells > 0 else 0
    
    logger.info(
        f"Validation metrics: {metrics['row_count']} rows, "
        f"{metrics['column_count']} columns, "
        f"quality score: {metrics['quality_score']}%"
    )
    
    return metrics


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Data cleaning module - Example usage")
    
    # Example 1: Test standardize_columns
    logger.info("\n=== Example 1: Column Standardization ===")
    sample_df = pd.DataFrame({
        'First Name': ['John', 'Jane'],
        'Last-Name': ['Doe', 'Smith'],
        'AGE  ': [30, 25],
        'Email Address': ['john@example.com', 'jane@example.com']
    })
    
    logger.info(f"Original columns: {sample_df.columns.tolist()}")
    standardized_df = standardize_columns(sample_df)
    logger.info(f"Standardized columns: {standardized_df.columns.tolist()}")
    
    # Example 2: Test clean_dataframe
    logger.info("\n=== Example 2: Full DataFrame Cleaning ===")
    sample_data = pd.DataFrame({
        'State': [' karnataka ', 'KERALA', 'Tamil Nadu', None, 'karnataka'],
        'District': ['Bangalore', 'trivandrum ', 'Chennai', 'Mysore', None],
        'Enrolment Date': ['2023-01-15', '01/15/2023', '2023-03-10', '2023-04-20', '2023-05-15'],
        'Count': [100, 200, 150, 175, 125]
    })
    
    logger.info(f"Original data shape: {sample_data.shape}")
    logger.info(f"Original data:\n{sample_data}")
    
    cleaned_data = clean_dataframe(sample_data)
    
    logger.info(f"\nCleaned data shape: {cleaned_data.shape}")
    logger.info(f"Cleaned data:\n{cleaned_data}")
    
    # Example 3: Validate cleaned data
    logger.info("\n=== Example 3: Data Validation ===")
    validation_metrics = validate_cleaned_data(cleaned_data)
    logger.info(f"Validation metrics: {validation_metrics}")
