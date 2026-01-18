"""
Data Pipeline Execution Script

Orchestrates the complete data processing pipeline from raw CSV files to
aggregated district-level metrics ready for analytics and scoring.

Pipeline Stages:
1. Dataset Verification - Ensure all required data files exist
2. Data Loading - Load raw CSV files in memory-safe chunks
3. Data Cleaning - Standardize and validate each chunk
4. Incremental Aggregation - Build district-month metrics
5. Persistence - Save results to Parquet for fast access

Design Principles:
- Memory-safe: Process large datasets without memory overflow
- Fail-fast: Detect and report errors immediately
- Progress tracking: Log progress every 1M rows processed
- Deterministic: Reproducible results
- Production-ready: Comprehensive error handling and logging
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import time
from datetime import datetime
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from verify_dataset import verify_dataset_structure
from data_loading import load_all_datasets, get_dataset_info
from data_cleaning import clean_dataframe
from data_cleaning.location_cleaner import clean_location_columns
from aggregation import aggregate_chunks, save_aggregated_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent.parent / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    ]
)
logger = logging.getLogger(__name__)


# Pipeline configuration
OUTPUT_DIR = Path(__file__).parent.parent / "data_cache"
OUTPUT_FILE = "district_metrics.parquet"
PROGRESS_LOG_INTERVAL = 1_000_000  # Log every 1 million rows


def ensure_output_directory() -> Path:
    """
    Ensure output directory exists for storing processed data.
    
    Returns:
        Path: Output directory path
    """
    output_path = OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")
    return output_path


def verify_prerequisites() -> None:
    """
    Verify that all prerequisites are met before running pipeline.
    
    Checks:
    - Dataset folder structure
    - Required CSV files
    - Output directory writable
    
    Raises:
        RuntimeError: If any prerequisite check fails
    """
    logger.info("=" * 80)
    logger.info("PIPELINE PREREQUISITES CHECK")
    logger.info("=" * 80)
    
    # Verify dataset structure
    try:
        logger.info("\n[1/2] Verifying dataset structure...")
        verify_dataset_structure(skip_readability=False)
        logger.info("✅ Dataset structure verified\n")
    except RuntimeError as e:
        logger.error(f"❌ Dataset verification failed: {e}")
        raise RuntimeError(f"Prerequisites not met: {e}")
    
    # Verify output directory
    try:
        logger.info("[2/2] Verifying output directory...")
        output_dir = ensure_output_directory()
        
        # Test write permissions
        test_file = output_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        
        logger.info(f"✅ Output directory writable: {output_dir}\n")
    except Exception as e:
        logger.error(f"❌ Output directory check failed: {e}")
        raise RuntimeError(f"Cannot write to output directory: {e}")
    
    logger.info("=" * 80)
    logger.info("✅ ALL PREREQUISITES MET - Starting pipeline\n")
    logger.info("=" * 80)


def process_dataset_type(
    dataset_type: str,
    folder_path: Path,
    total_rows_processed: int,
    aggregated_chunks: list
) -> int:
    """
    Process all chunks for a specific dataset type.
    
    Args:
        dataset_type: Type of dataset ('biometric', 'demographic', 'enrolment')
        folder_path: Path to the dataset folder
        total_rows_processed: Running total of rows processed
        aggregated_chunks: List to append aggregated chunks to
        
    Returns:
        int: Updated total rows processed
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING: {dataset_type.upper()}")
    logger.info(f"{'=' * 80}\n")
    
    dataset_start_time = time.time()
    dataset_row_count = 0
    chunk_count = 0
    
    try:
        # Load dataset in chunks
        from data_loading import load_dataset
        for chunk_df in load_dataset(str(folder_path), dataset_type):
            chunk_count += 1
            chunk_rows = len(chunk_df)
            dataset_row_count += chunk_rows
            total_rows_processed += chunk_rows
            
            logger.info(f"Processing chunk {chunk_count} ({chunk_rows:,} rows)...")
            
            # Clean the chunk
            try:
                cleaned_df = clean_dataframe(chunk_df)
                cleaned_rows = len(cleaned_df)
                dropped_rows = chunk_rows - cleaned_rows
                
                if dropped_rows > 0:
                    logger.warning(
                        f"⚠️  Dropped {dropped_rows:,} invalid rows "
                        f"({dropped_rows/chunk_rows*100:.2f}%)"
                    )
                
                logger.debug(f"Cleaned chunk: {cleaned_rows:,} valid rows")
            
            except Exception as e:
                logger.error(f"❌ Cleaning failed for chunk {chunk_count}: {e}")
                raise RuntimeError(f"Data cleaning failed: {e}")
            
            # Apply location cleaning to remove invalid districts
            try:
                pre_location_clean_rows = len(cleaned_df)
                cleaned_df = clean_location_columns(cleaned_df)
                post_location_clean_rows = len(cleaned_df)
                location_dropped = pre_location_clean_rows - post_location_clean_rows
                
                if location_dropped > 0:
                    logger.info(
                        f"🗺️  Location cleaning: Removed {location_dropped:,} rows with invalid districts "
                        f"({location_dropped/pre_location_clean_rows*100:.2f}%)"
                    )
                
                logger.debug(f"Location-cleaned chunk: {post_location_clean_rows:,} valid rows")
            
            except Exception as e:
                logger.error(f"❌ Location cleaning failed for chunk {chunk_count}: {e}")
                raise RuntimeError(f"Location cleaning failed: {e}")
            
            # Aggregate the chunk
            try:
                # Prepare chunk for aggregation (adds count columns, etc.)
                from aggregation import prepare_chunk_for_aggregation, aggregate_single_chunk
                prepared_df = prepare_chunk_for_aggregation(cleaned_df, dataset_type)
                
                # Determine groupby columns dynamically
                groupby_cols = []
                for col in prepared_df.columns:
                    col_lower = col.lower()
                    if 'district' in col_lower:
                        groupby_cols.append(col)
                    elif any(kw in col_lower for kw in ['month', 'date']):
                        groupby_cols.append(col)
                
                if not groupby_cols:
                    logger.error("No groupby columns (district/month) found in chunk")
                    raise RuntimeError("Cannot aggregate without district/month columns")
                
                # Aggregate this single chunk
                aggregated = aggregate_single_chunk(prepared_df, groupby_cols)
                
                if aggregated is not None and len(aggregated) > 0:
                    aggregated_chunks.append(aggregated)
                    logger.debug(f"Aggregated to {len(aggregated):,} district-month records")
                else:
                    logger.warning(f"⚠️  No aggregatable data in chunk {chunk_count}")
            
            except Exception as e:
                logger.error(f"❌ Aggregation failed for chunk {chunk_count}: {e}")
                raise RuntimeError(f"Data aggregation failed: {e}")
            
            # Progress logging (every 1M rows)
            if total_rows_processed // PROGRESS_LOG_INTERVAL > (total_rows_processed - chunk_rows) // PROGRESS_LOG_INTERVAL:
                logger.info(
                    f"📊 PROGRESS: {total_rows_processed:,} rows processed "
                    f"({chunk_count} chunks, {len(aggregated_chunks)} aggregated chunks)"
                )
        
        # Dataset completion summary
        dataset_elapsed = time.time() - dataset_start_time
        logger.info(f"\n✅ {dataset_type.upper()} COMPLETE:")
        logger.info(f"   - Rows processed: {dataset_row_count:,}")
        logger.info(f"   - Chunks processed: {chunk_count}")
        logger.info(f"   - Time elapsed: {dataset_elapsed:.2f}s")
        logger.info(f"   - Processing rate: {dataset_row_count/dataset_elapsed:.0f} rows/sec\n")
    
    except Exception as e:
        logger.error(f"❌ Failed to process {dataset_type}: {e}")
        raise
    
    return total_rows_processed


def main() -> int:
    """
    Execute the complete data processing pipeline.
    
    Pipeline Flow:
    1. Verify prerequisites (dataset structure, output directory)
    2. Load dataset info (discover available files)
    3. For each dataset type:
       a. Load chunks
       b. Clean each chunk
       c. Aggregate each chunk
    4. Combine all aggregated chunks
    5. Save final results to Parquet
    
    Returns:
        int: Exit code (0 = success, 1 = failure)
    """
    pipeline_start_time = time.time()
    
    logger.info("\n" + "=" * 80)
    logger.info("ASEWIS DATA PIPELINE")
    logger.info("Aadhar System Engineering & Workflow Intelligence System")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")
    
    try:
        # Step 1: Verify prerequisites
        verify_prerequisites()
        
        # Step 2: Get dataset information
        logger.info("\n" + "=" * 80)
        logger.info("DATASET DISCOVERY")
        logger.info("=" * 80 + "\n")
        
        # Discover datasets from the dataset folder
        dataset_base = Path(__file__).parent.parent / "dataset"
        dataset_folders = {
            'biometric': dataset_base / 'api_data_aadhar_biometric',
            'demographic': dataset_base / 'api_data_aadhar_demographic',
            'enrolment': dataset_base / 'api_data_aadhar_enrolment'
        }
        
        dataset_info = {}
        for dataset_type, folder_path in dataset_folders.items():
            if folder_path.exists():
                dataset_info[dataset_type] = get_dataset_info(str(folder_path))
        
        logger.info("Available datasets:")
        for dataset_type, info in dataset_info.items():
            logger.info(f"  - {dataset_type}: {info['file_count']} files, {info['total_size_mb']:.2f} MB")
        
        total_files = sum(info['file_count'] for info in dataset_info.values())
        total_size = sum(info['total_size_mb'] for info in dataset_info.values())
        
        logger.info(f"\nTotal: {total_files} files, {total_size:.2f} MB")
        logger.info("=" * 80 + "\n")
        
        # Step 3: Process all datasets
        total_rows_processed = 0
        aggregated_chunks = []
        
        dataset_types = ['biometric', 'demographic', 'enrolment']
        
        for dataset_type in dataset_types:
            if dataset_type in dataset_info:
                folder_path = dataset_folders[dataset_type]
                total_rows_processed = process_dataset_type(
                    dataset_type,
                    folder_path,
                    total_rows_processed,
                    aggregated_chunks
                )
            else:
                logger.warning(f"⚠️  Dataset type '{dataset_type}' not found, skipping")
        
        # Step 4: Combine all aggregated chunks
        logger.info("\n" + "=" * 80)
        logger.info("COMBINING AGGREGATED DATA")
        logger.info("=" * 80 + "\n")
        
        if not aggregated_chunks:
            raise RuntimeError("No aggregated data produced - pipeline failed")
        
        logger.info(f"Combining {len(aggregated_chunks)} aggregated chunks...")
        
        # Combine all chunks and re-aggregate to handle overlaps
        combined_df = pd.concat(aggregated_chunks, ignore_index=True)
        logger.info(f"Combined shape before final aggregation: {combined_df.shape}")
        
        # Determine groupby columns from the combined dataframe
        groupby_cols = []
        for col in combined_df.columns:
            col_lower = col.lower()
            if 'district' in col_lower or any(kw in col_lower for kw in ['month', 'date']):
                groupby_cols.append(col)
        
        if not groupby_cols:
            raise RuntimeError("No groupby columns (district/month) found in aggregated data")
        
        # Identify numeric columns to aggregate
        numeric_cols = combined_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Remove groupby columns from numeric columns
        agg_cols = [col for col in numeric_cols if col not in groupby_cols]
        
        logger.info(f"Final aggregation by {groupby_cols}...")
        logger.info(f"Aggregating columns: {agg_cols[:5]}... ({len(agg_cols)} total)")
        
        final_aggregated = combined_df.groupby(groupby_cols)[agg_cols].sum().reset_index()
        
        logger.info(f"Final aggregated shape: {final_aggregated.shape}")
        
        # Report on groupby columns found
        for col in groupby_cols:
            if col in final_aggregated.columns:
                logger.info(f"{col.capitalize()}s: {final_aggregated[col].nunique()}")
        
        logger.info(f"Total records: {len(final_aggregated):,}")
        
        # Step 5: Save to Parquet
        logger.info("\n" + "=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80 + "\n")
        
        output_path = OUTPUT_DIR / OUTPUT_FILE
        
        try:
            save_aggregated_metrics(final_aggregated, output_path)
            
            # Verify saved file
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"✅ Saved: {output_path.absolute()}")
            logger.info(f"   File size: {file_size_mb:.2f} MB")
            
            # Verify readability
            test_load = pd.read_parquet(output_path)
            logger.info(f"   Verified: {len(test_load):,} records readable")
        
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            raise RuntimeError(f"Failed to save output file: {e}")
        
        # Pipeline completion summary
        pipeline_elapsed = time.time() - pipeline_start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total rows processed: {total_rows_processed:,}")
        logger.info(f"Final aggregated records: {len(final_aggregated):,}")
        logger.info(f"Output file: {output_path.absolute()}")
        logger.info(f"Total time: {pipeline_elapsed:.2f}s ({pipeline_elapsed/60:.2f} min)")
        logger.info(f"Average rate: {total_rows_processed/pipeline_elapsed:.0f} rows/sec")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        logger.info("\n✅ PIPELINE SUCCESSFUL\n")
        logger.info("Next steps:")
        logger.info("  1. python src/intelligence/feature_engineering.py")
        logger.info("  2. python src/scoring.py")
        logger.info("  3. streamlit run app/streamlit_app.py\n")
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        logger.info("Partial results may be incomplete - consider restarting\n")
        return 130  # Standard exit code for Ctrl+C
    
    except Exception as e:
        logger.error(f"\n\n❌ PIPELINE FAILED: {e}", exc_info=True)
        logger.error("\nPlease fix the error and restart the pipeline\n")
        return 1


if __name__ == "__main__":
    """
    Direct execution entry point.
    
    Usage:
        python src/run_pipeline.py
    """
    exit_code = main()
    sys.exit(exit_code)
