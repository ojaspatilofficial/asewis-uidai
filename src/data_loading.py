"""
Data loading module for large CSV datasets.

This module provides memory-efficient CSV loading functionality for the ASEWIS system.
It uses chunked reading to handle large datasets that may not fit into memory.
"""

import logging
from pathlib import Path
from typing import Generator, Iterator, Optional
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


def load_dataset(
    folder_path: str,
    source_type: str,
    chunksize: int = 100_000
) -> Generator[pd.DataFrame, None, None]:
    """
    Load CSV files from a folder with chunking for memory efficiency.
    
    This function discovers all CSV files in the given folder and yields
    data chunks with an added source_type column for identification.
    
    Args:
        folder_path: Path to folder containing CSV files
        source_type: Type identifier to add as a column (e.g., 'biometric', 'demographic')
        chunksize: Number of rows to read per chunk (default: 100,000)
        
    Yields:
        pd.DataFrame: Chunks of data with source_type column added
        
    Raises:
        FileNotFoundError: If folder_path does not exist
        ValueError: If no CSV files found in folder
        
    Example:
        >>> for chunk in load_dataset('dataset/api_data_aadhar_biometric', 'biometric'):
        ...     process(chunk)
    """
    folder = Path(folder_path)
    
    # Validate folder exists
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        logger.error(f"Path is not a directory: {folder_path}")
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Discover CSV files
    csv_files = sorted(folder.glob("**/*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in: {folder_path}")
        raise ValueError(f"No CSV files found in folder: {folder_path}")
    
    logger.info(f"Found {len(csv_files)} CSV file(s) in {folder_path}")
    
    # Process each CSV file
    total_chunks = 0
    total_rows = 0
    
    for csv_file in csv_files:
        logger.info(f"Loading file: {csv_file.name}")
        
        try:
            # Read CSV in chunks for memory efficiency
            chunk_iterator = pd.read_csv(
                csv_file,
                chunksize=chunksize,
                low_memory=True
            )
            
            file_chunks = 0
            file_rows = 0
            
            for chunk in chunk_iterator:
                # Add source_type column
                chunk['source_type'] = source_type
                
                file_chunks += 1
                file_rows += len(chunk)
                total_chunks += 1
                total_rows += len(chunk)
                
                yield chunk
            
            logger.info(
                f"Completed {csv_file.name}: {file_chunks} chunks, "
                f"{file_rows:,} rows"
            )
            
        except pd.errors.EmptyDataError:
            logger.warning(f"Empty CSV file skipped: {csv_file.name}")
            continue
            
        except pd.errors.ParserError as e:
            logger.error(f"Parser error in {csv_file.name}: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Error loading {csv_file.name}: {e}")
            raise
    
    logger.info(
        f"Dataset loading complete: {total_chunks} total chunks, "
        f"{total_rows:,} total rows from {len(csv_files)} files"
    )


def load_all_datasets(
    base_path: str = "dataset",
    chunksize: int = 100_000
) -> Generator[pd.DataFrame, None, None]:
    """
    Load all datasets from standard ASEWIS folder structure.
    
    This function automatically discovers and loads data from all three
    standard subdirectories: biometric, demographic, and enrolment.
    
    Args:
        base_path: Base directory containing dataset subfolders (default: 'dataset')
        chunksize: Number of rows to read per chunk (default: 100,000)
        
    Yields:
        pd.DataFrame: Chunks of data from all datasets with source_type column
        
    Raises:
        FileNotFoundError: If base_path does not exist
        
    Example:
        >>> for chunk in load_all_datasets():
        ...     print(chunk['source_type'].unique())
    """
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        logger.error(f"Base dataset path not found: {base_path}")
        raise FileNotFoundError(f"Base dataset path does not exist: {base_path}")
    
    # Define dataset configurations
    dataset_configs = [
        {
            'folder': 'api_data_aadhar_biometric',
            'source_type': 'biometric'
        },
        {
            'folder': 'api_data_aadhar_demographic',
            'source_type': 'demographic'
        },
        {
            'folder': 'api_data_aadhar_enrolment',
            'source_type': 'enrolment'
        }
    ]
    
    logger.info(f"Starting load_all_datasets from: {base_path}")
    
    total_datasets_loaded = 0
    
    for config in dataset_configs:
        folder_name = config['folder']
        source_type = config['source_type']
        folder_path = base_dir / folder_name / folder_name
        
        # Check if folder exists before attempting to load
        if not folder_path.exists():
            logger.warning(
                f"Skipping {source_type}: folder not found at {folder_path}"
            )
            continue
        
        logger.info(f"Loading {source_type} dataset from: {folder_path}")
        
        try:
            # Yield chunks from this dataset
            for chunk in load_dataset(
                str(folder_path),
                source_type,
                chunksize=chunksize
            ):
                yield chunk
            
            total_datasets_loaded += 1
            
        except Exception as e:
            logger.error(f"Error loading {source_type} dataset: {e}")
            # Continue with next dataset instead of failing completely
            continue
    
    if total_datasets_loaded == 0:
        logger.warning("No datasets were successfully loaded")
    else:
        logger.info(
            f"Completed load_all_datasets: {total_datasets_loaded} "
            f"dataset(s) processed"
        )


def get_dataset_info(folder_path: str) -> dict:
    """
    Get metadata about a dataset folder without loading data.
    
    Args:
        folder_path: Path to folder containing CSV files
        
    Returns:
        dict: Metadata including file count, names, and estimated size
        
    Example:
        >>> info = get_dataset_info('dataset/api_data_aadhar_biometric')
        >>> print(f"Files: {info['file_count']}")
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    
    csv_files = sorted(folder.glob("**/*.csv"))
    
    total_size = sum(f.stat().st_size for f in csv_files)
    
    info = {
        'folder_path': str(folder),
        'file_count': len(csv_files),
        'file_names': [f.name for f in csv_files],
        'total_size_bytes': total_size,
        'total_size_mb': round(total_size / (1024 * 1024), 2)
    }
    
    logger.info(
        f"Dataset info for {folder_path}: {info['file_count']} files, "
        f"{info['total_size_mb']} MB"
    )
    
    return info


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    logger.info("Data loading module - Example usage")
    
    # Example 1: Load single dataset
    try:
        chunk_count = 0
        row_count = 0
        
        for chunk in load_dataset(
            "dataset/api_data_aadhar_biometric/api_data_aadhar_biometric",
            "biometric",
            chunksize=100_000
        ):
            chunk_count += 1
            row_count += len(chunk)
            logger.info(f"Processed chunk {chunk_count}: {len(chunk)} rows")
            
            # Process only first chunk for demo
            if chunk_count >= 1:
                break
        
        logger.info(f"Example 1 complete: {chunk_count} chunks, {row_count:,} rows")
        
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
    
    # Example 2: Get dataset info
    try:
        info = get_dataset_info(
            "dataset/api_data_aadhar_biometric/api_data_aadhar_biometric"
        )
        logger.info(f"Dataset metadata: {info}")
        
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")
