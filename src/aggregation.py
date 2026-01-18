"""
Aggregation module for district-month level metrics.

This module provides memory-efficient aggregation functions that work with
chunked data iterators to produce district-month level summaries without
loading entire datasets into memory.
"""

import logging
from pathlib import Path
from typing import Iterator, List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def identify_metric_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify columns for different metric categories based on column names.
    
    This function categorizes columns to determine how they should be aggregated.
    
    Args:
        df: Input DataFrame with standardized column names
        
    Returns:
        dict: Dictionary mapping metric types to column names
        
    Example:
        >>> columns = identify_metric_columns(df)
        >>> print(columns['count_columns'])
        ['enrolment_count', 'update_count']
    """
    columns_map = {
        'count_columns': [],      # Columns to sum (counts, totals)
        'demographic_columns': [], # Demographic-related columns
        'biometric_columns': [],   # Biometric-related columns
        'age_columns': [],         # Age group columns
        'groupby_columns': []      # Columns to group by (district, month)
    }
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Identify grouping columns (district, state, date/month)
        if any(kw in col_lower for kw in ['district', 'month', 'date', 'state']):
            columns_map['groupby_columns'].append(col)
        
        # Identify count/total columns (numeric columns with count/total keywords)
        elif any(kw in col_lower for kw in ['count', 'total', 'number', 'num']):
            if pd.api.types.is_numeric_dtype(df[col]):
                columns_map['count_columns'].append(col)
        
        # Identify demographic columns
        elif 'demographic' in col_lower or 'demo' in col_lower:
            if pd.api.types.is_numeric_dtype(df[col]):
                columns_map['demographic_columns'].append(col)
        
        # Identify biometric columns
        elif 'biometric' in col_lower or 'bio' in col_lower:
            if pd.api.types.is_numeric_dtype(df[col]):
                columns_map['biometric_columns'].append(col)
        
        # Identify age group columns
        elif any(kw in col_lower for kw in ['age', 'child', 'adult', 'senior']):
            if pd.api.types.is_numeric_dtype(df[col]):
                columns_map['age_columns'].append(col)
    
    # Log identified columns
    logger.debug(f"Identified columns for aggregation:")
    for key, cols in columns_map.items():
        if cols:
            logger.debug(f"  {key}: {cols}")
    
    return columns_map


def prepare_chunk_for_aggregation(
    df: pd.DataFrame,
    source_type: str = None
) -> pd.DataFrame:
    """
    Prepare a data chunk for aggregation.
    
    This function ensures the chunk has required columns and proper data types
    before aggregation. It creates derived columns based on source_type.
    
    Args:
        df: Input DataFrame chunk
        source_type: Type of data (biometric, demographic, enrolment)
        
    Returns:
        pd.DataFrame: Prepared chunk ready for aggregation
    """
    # Create a copy to avoid modifying original
    df_prep = df.copy()
    
    # Use source_type from column if not provided
    if source_type is None and 'source_type' in df_prep.columns:
        # Assume all rows in chunk have same source_type
        source_type = df_prep['source_type'].iloc[0] if len(df_prep) > 0 else None
    
    # Identify district and date columns
    # These are required for grouping
    district_col = None
    date_col = None
    
    for col in df_prep.columns:
        if 'district' in col.lower() and district_col is None:
            district_col = col
        if any(kw in col.lower() for kw in ['date', 'month']) and date_col is None:
            date_col = col
    
    if district_col is None:
        logger.warning("No district column found in chunk")
    if date_col is None:
        logger.warning("No date column found in chunk")
    
    # Create count column based on source_type
    # Each row represents one transaction/record
    # We count rows to get total transactions per district-month
    if source_type:
        count_col_name = f'{source_type}_count'
        df_prep[count_col_name] = 1
        logger.debug(f"Created count column: {count_col_name}")
    
    # Handle age-related categorization if age column exists
    age_columns = [col for col in df_prep.columns if 'age' in col.lower()]
    if age_columns:
        age_col = age_columns[0]
        
        # Create age group categories for population analysis
        # Child: 0-17, Adult: 18-59, Senior: 60+
        if pd.api.types.is_numeric_dtype(df_prep[age_col]):
            df_prep['age_group_child'] = (df_prep[age_col] < 18).astype(int)
            df_prep['age_group_adult'] = ((df_prep[age_col] >= 18) & (df_prep[age_col] < 60)).astype(int)
            df_prep['age_group_senior'] = (df_prep[age_col] >= 60).astype(int)
            logger.debug("Created age group categorization columns")
    
    return df_prep


def aggregate_single_chunk(
    df: pd.DataFrame,
    groupby_columns: List[str]
) -> pd.DataFrame:
    """
    Aggregate a single chunk to district-month level.
    
    This function performs the core aggregation logic:
    - Groups by district and month
    - Sums all numeric columns
    - Produces one row per district-month combination
    
    Args:
        df: Prepared DataFrame chunk
        groupby_columns: List of columns to group by (e.g., ['district', 'month'])
        
    Returns:
        pd.DataFrame: Aggregated chunk at district-month level
    """
    # Verify groupby columns exist
    valid_groupby_cols = [col for col in groupby_columns if col in df.columns]
    
    if not valid_groupby_cols:
        logger.error(f"No valid groupby columns found in chunk")
        return pd.DataFrame()
    
    # Identify numeric columns to aggregate
    # We sum all numeric columns (counts, totals, etc.)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove groupby columns from aggregation columns
    agg_columns = [col for col in numeric_columns if col not in valid_groupby_cols]
    
    if not agg_columns:
        logger.warning("No numeric columns found for aggregation")
        # Return just the groupby columns with unique combinations
        return df[valid_groupby_cols].drop_duplicates().reset_index(drop=True)
    
    try:
        # Perform aggregation: group by district-month and sum all numeric columns
        # This produces one row per unique district-month combination
        # All counts/totals are summed within each group
        aggregated = df.groupby(valid_groupby_cols, as_index=False)[agg_columns].sum()
        
        logger.debug(
            f"Aggregated chunk: {len(df)} rows -> {len(aggregated)} groups, "
            f"summed {len(agg_columns)} columns"
        )
        
        return aggregated
        
    except Exception as e:
        logger.error(f"Error aggregating chunk: {e}")
        return pd.DataFrame()


def aggregate_chunks(
    chunk_iterator: Iterator[pd.DataFrame],
    cache_dir: str = "data_cache/temp"
) -> pd.DataFrame:
    """
    Aggregate multiple data chunks incrementally to district-month level.
    
    This function processes chunks one at a time, aggregates each chunk,
    and then combines results. This approach is memory-efficient as it
    never loads the full dataset into memory.
    
    Aggregation strategy:
    1. Process each chunk individually (aggregate to district-month)
    2. Store intermediate aggregations
    3. Combine all intermediate results
    4. Perform final aggregation (in case same district-month appears in multiple chunks)
    
    Args:
        chunk_iterator: Iterator yielding DataFrame chunks
        cache_dir: Directory to store intermediate results (default: data_cache/temp)
        
    Returns:
        pd.DataFrame: Final aggregated metrics with one row per district-month
        
    Example:
        >>> chunks = load_all_datasets()
        >>> metrics = aggregate_chunks(chunks)
        >>> print(metrics.groupby('district').size())  # Months per district
    """
    logger.info("Starting incremental chunk aggregation")
    
    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # List to store intermediate aggregated chunks
    aggregated_chunks = []
    
    chunk_count = 0
    total_input_rows = 0
    
    for chunk in chunk_iterator:
        chunk_count += 1
        chunk_size = len(chunk)
        total_input_rows += chunk_size
        
        logger.info(f"Processing chunk {chunk_count}: {chunk_size:,} rows")
        
        # Step 1: Prepare chunk for aggregation
        # This adds derived columns and ensures proper data types
        source_type = chunk['source_type'].iloc[0] if 'source_type' in chunk.columns and len(chunk) > 0 else None
        prepared_chunk = prepare_chunk_for_aggregation(chunk, source_type)
        
        # Step 2: Identify grouping columns (district, month/date)
        # These define the aggregation grain: one row per district-month
        groupby_cols = []
        for col in prepared_chunk.columns:
            if any(kw in col.lower() for kw in ['district', 'state']):
                groupby_cols.append(col)
            elif any(kw in col.lower() for kw in ['month', 'date']):
                groupby_cols.append(col)
        
        if not groupby_cols:
            logger.warning(f"Chunk {chunk_count}: No groupby columns found, skipping")
            continue
        
        # Remove duplicates from groupby columns
        groupby_cols = list(dict.fromkeys(groupby_cols))
        logger.debug(f"Chunk {chunk_count}: Grouping by {groupby_cols}")
        
        # Step 3: Aggregate this chunk to district-month level
        # This reduces chunk size significantly (thousands of rows -> hundreds)
        aggregated_chunk = aggregate_single_chunk(prepared_chunk, groupby_cols)
        
        if aggregated_chunk.empty:
            logger.warning(f"Chunk {chunk_count}: Aggregation produced empty result")
            continue
        
        # Store aggregated chunk
        aggregated_chunks.append(aggregated_chunk)
        
        logger.info(
            f"Chunk {chunk_count} aggregated: {chunk_size:,} rows -> "
            f"{len(aggregated_chunk)} district-month combinations"
        )
        
        # Periodic memory management: combine accumulated chunks
        # This prevents accumulating too many small DataFrames in memory
        if len(aggregated_chunks) >= 10:
            logger.info(f"Combining {len(aggregated_chunks)} accumulated chunks...")
            combined = pd.concat(aggregated_chunks, ignore_index=True)
            
            # Re-aggregate the combined chunks
            # This is necessary because same district-month may appear in different chunks
            groupby_cols_combined = [col for col in groupby_cols if col in combined.columns]
            if groupby_cols_combined:
                combined = aggregate_single_chunk(combined, groupby_cols_combined)
                logger.info(f"Re-aggregated to {len(combined)} district-month combinations")
            
            # Replace list with single combined DataFrame
            aggregated_chunks = [combined]
    
    # Final step: Combine all aggregated chunks
    if not aggregated_chunks:
        logger.error("No chunks were successfully aggregated")
        return pd.DataFrame()
    
    logger.info(f"Combining final {len(aggregated_chunks)} aggregated chunk(s)...")
    final_combined = pd.concat(aggregated_chunks, ignore_index=True)
    
    # Perform final aggregation to handle any remaining duplicates
    # This ensures each district-month appears exactly once
    groupby_cols_final = []
    for col in ['district', 'state', 'month', 'date']:
        matching_cols = [c for c in final_combined.columns if col in c.lower()]
        groupby_cols_final.extend(matching_cols)
    
    groupby_cols_final = list(dict.fromkeys(groupby_cols_final))
    
    if groupby_cols_final:
        logger.info("Performing final aggregation across all chunks...")
        final_metrics = aggregate_single_chunk(final_combined, groupby_cols_final)
    else:
        logger.warning("No groupby columns found for final aggregation")
        final_metrics = final_combined
    
    # Summary statistics
    logger.info(
        f"Aggregation complete: {total_input_rows:,} input rows -> "
        f"{len(final_metrics)} district-month combinations "
        f"(from {chunk_count} chunks)"
    )
    
    # Log sample of metrics
    if not final_metrics.empty:
        logger.info(f"Final metrics columns: {final_metrics.columns.tolist()}")
        logger.info(f"Sample metrics:\n{final_metrics.head()}")
    
    return final_metrics


def save_aggregated_metrics(
    df: pd.DataFrame,
    output_path: str = "data_cache/aggregated_metrics.parquet",
    compression: str = "snappy"
) -> str:
    """
    Save aggregated metrics to Parquet format.
    
    Parquet is chosen for:
    - Efficient columnar storage (better compression)
    - Fast read performance for analytics
    - Schema preservation (data types maintained)
    - Wide industry adoption
    
    Args:
        df: Aggregated metrics DataFrame
        output_path: Path to save Parquet file (default: data_cache/aggregated_metrics.parquet)
        compression: Compression algorithm (default: snappy for balance of speed/size)
        
    Returns:
        str: Absolute path where file was saved
        
    Raises:
        ValueError: If DataFrame is empty
        
    Example:
        >>> metrics = aggregate_chunks(chunks)
        >>> path = save_aggregated_metrics(metrics)
        >>> print(f"Saved to: {path}")
    """
    if df.empty:
        logger.error("Cannot save empty DataFrame")
        raise ValueError("DataFrame is empty, nothing to save")
    
    logger.info(f"Saving aggregated metrics: {len(df)} rows, {len(df.columns)} columns")
    
    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata columns
    df_to_save = df.copy()
    df_to_save['aggregated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # Save as Parquet with compression
        # Snappy provides good balance between compression ratio and speed
        df_to_save.to_parquet(
            output_file,
            engine='pyarrow',
            compression=compression,
            index=False
        )
        
        # Get file size
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        logger.info(
            f"✓ Saved aggregated metrics to: {output_file.absolute()}\n"
            f"  File size: {file_size_mb:.2f} MB\n"
            f"  Compression: {compression}\n"
            f"  Rows: {len(df):,}\n"
            f"  Columns: {len(df.columns)}"
        )
        
        return str(output_file.absolute())
        
    except Exception as e:
        logger.error(f"Error saving Parquet file: {e}")
        raise


def load_aggregated_metrics(
    input_path: str = "data_cache/aggregated_metrics.parquet"
) -> pd.DataFrame:
    """
    Load previously saved aggregated metrics from Parquet.
    
    Args:
        input_path: Path to Parquet file to load
        
    Returns:
        pd.DataFrame: Loaded aggregated metrics
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Example:
        >>> metrics = load_aggregated_metrics()
        >>> print(metrics.describe())
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        logger.error(f"Parquet file not found: {input_path}")
        raise FileNotFoundError(f"File does not exist: {input_path}")
    
    logger.info(f"Loading aggregated metrics from: {input_file}")
    
    try:
        df = pd.read_parquet(input_file, engine='pyarrow')
        
        logger.info(
            f"✓ Loaded aggregated metrics: {len(df):,} rows, "
            f"{len(df.columns)} columns"
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading Parquet file: {e}")
        raise


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Aggregation module - Example usage")
    
    # Example: Create sample data and demonstrate aggregation
    logger.info("\n=== Example: District-Month Aggregation ===")
    
    # Create sample data simulating multiple chunks
    sample_data_1 = pd.DataFrame({
        'district': ['BANGALORE', 'BANGALORE', 'MYSORE', 'MYSORE'],
        'month': ['2023-01', '2023-01', '2023-01', '2023-01'],
        'enrolment_count': [100, 150, 80, 90],
        'age': [25, 35, 45, 55],
        'source_type': ['enrolment', 'enrolment', 'enrolment', 'enrolment']
    })
    
    sample_data_2 = pd.DataFrame({
        'district': ['BANGALORE', 'BANGALORE', 'MYSORE'],
        'month': ['2023-01', '2023-02', '2023-02'],
        'enrolment_count': [120, 200, 100],
        'age': [30, 40, 50],
        'source_type': ['enrolment', 'enrolment', 'enrolment']
    })
    
    # Create iterator from sample chunks
    def sample_chunk_iterator():
        yield sample_data_1
        yield sample_data_2
    
    # Aggregate chunks
    logger.info("Input chunk 1:")
    logger.info(f"\n{sample_data_1}")
    logger.info("\nInput chunk 2:")
    logger.info(f"\n{sample_data_2}")
    
    aggregated = aggregate_chunks(sample_chunk_iterator())
    
    logger.info("\nAggregated result (district-month level):")
    logger.info(f"\n{aggregated}")
    
    # Save aggregated metrics
    if not aggregated.empty:
        try:
            saved_path = save_aggregated_metrics(
                aggregated,
                output_path="data_cache/test_aggregated_metrics.parquet"
            )
            logger.info(f"\nMetrics saved successfully to: {saved_path}")
            
            # Load back to verify
            loaded = load_aggregated_metrics("data_cache/test_aggregated_metrics.parquet")
            logger.info(f"\nVerification - loaded {len(loaded)} rows")
            
        except Exception as e:
            logger.error(f"Error in save/load example: {e}")
