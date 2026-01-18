"""
Dataset Verification Module

Verifies dataset folder integrity before processing to fail fast and prevent
downstream errors. Ensures all required data sources are present and readable.

This module performs pre-flight checks on the dataset structure to ensure:
1. Dataset directory exists and is accessible
2. All required subdirectories are present
3. Each subdirectory contains at least one CSV file
4. CSV files are readable and not corrupted

Design Principles:
- Fail fast: Detect issues before processing starts
- Clear diagnostics: Provide actionable error messages
- Zero dependencies: Uses only Python standard library (except pandas for CSV validation)
- Executable: Can be run directly for manual verification
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Required dataset structure
REQUIRED_SUBFOLDERS = [
    'api_data_aadhar_biometric',
    'api_data_aadhar_demographic',
    'api_data_aadhar_enrolment'
]


def get_dataset_path() -> Path:
    """
    Get the dataset directory path relative to this module.
    
    Returns:
        Path: Absolute path to dataset directory
    """
    # Assume this file is in src/, dataset is at project root
    module_dir = Path(__file__).parent.parent
    dataset_dir = module_dir / "dataset"
    return dataset_dir


def check_directory_exists(path: Path, description: str) -> None:
    """
    Verify that a directory exists and is accessible.
    
    Args:
        path: Directory path to check
        description: Human-readable description for error messages
        
    Raises:
        RuntimeError: If directory does not exist or is not accessible
    """
    logger.debug(f"Checking {description}: {path}")
    
    if not path.exists():
        error_msg = (
            f"❌ {description} does not exist: {path}\n"
            f"Expected location: {path.absolute()}\n"
            f"Please ensure the dataset folder is in the correct location."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    if not path.is_dir():
        error_msg = (
            f"❌ {description} exists but is not a directory: {path}\n"
            f"Found: {path} (type: {type(path)})"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"✅ {description} exists: {path}")


def check_subfolder_exists(dataset_path: Path, subfolder_name: str) -> Path:
    """
    Verify that a required subfolder exists within the dataset.
    
    Args:
        dataset_path: Root dataset directory
        subfolder_name: Name of required subfolder
        
    Returns:
        Path: Absolute path to the subfolder
        
    Raises:
        RuntimeError: If subfolder does not exist
    """
    subfolder_path = dataset_path / subfolder_name
    
    logger.debug(f"Checking required subfolder: {subfolder_name}")
    
    if not subfolder_path.exists():
        error_msg = (
            f"❌ Required subfolder missing: {subfolder_name}\n"
            f"Expected location: {subfolder_path.absolute()}\n"
            f"Available folders in dataset/: {[p.name for p in dataset_path.iterdir() if p.is_dir()]}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    if not subfolder_path.is_dir():
        error_msg = (
            f"❌ {subfolder_name} exists but is not a directory: {subfolder_path}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"✅ Required subfolder exists: {subfolder_name}")
    return subfolder_path


def find_csv_files(directory: Path) -> List[Path]:
    """
    Find all CSV files in a directory (non-recursive).
    
    Args:
        directory: Directory to search
        
    Returns:
        list: List of CSV file paths
    """
    csv_files = list(directory.glob("*.csv"))
    logger.debug(f"Found {len(csv_files)} CSV files in {directory.name}")
    return csv_files


def check_csv_files_exist(subfolder_path: Path, subfolder_name: str) -> List[Path]:
    """
    Verify that a subfolder contains at least one CSV file.
    
    Args:
        subfolder_path: Path to subfolder
        subfolder_name: Name of subfolder (for logging)
        
    Returns:
        list: List of CSV file paths found
        
    Raises:
        RuntimeError: If no CSV files are found
    """
    csv_files = find_csv_files(subfolder_path)
    
    if len(csv_files) == 0:
        error_msg = (
            f"❌ No CSV files found in {subfolder_name}\n"
            f"Location checked: {subfolder_path.absolute()}\n"
            f"Expected: At least one .csv file\n"
            f"Available files: {[p.name for p in subfolder_path.iterdir() if p.is_file()]}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"✅ Found {len(csv_files)} CSV file(s) in {subfolder_name}")
    for csv_file in csv_files:
        logger.debug(f"   - {csv_file.name} ({csv_file.stat().st_size / 1024 / 1024:.2f} MB)")
    
    return csv_files


def verify_csv_readable(csv_path: Path) -> bool:
    """
    Verify that a CSV file is readable and not corrupted.
    
    Performs a quick check by reading the first few lines without loading
    the entire file into memory.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        bool: True if readable, False otherwise
    """
    try:
        # Quick check: try to read first 5 lines
        with open(csv_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                # Basic validation: ensure line has content
                if not line.strip():
                    logger.warning(f"⚠️  Empty line found in {csv_path.name} at line {i+1}")
        
        logger.debug(f"✅ CSV readable: {csv_path.name}")
        return True
    
    except UnicodeDecodeError as e:
        logger.error(f"❌ Encoding error in {csv_path.name}: {e}")
        return False
    
    except PermissionError as e:
        logger.error(f"❌ Permission denied reading {csv_path.name}: {e}")
        return False
    
    except Exception as e:
        logger.error(f"❌ Error reading {csv_path.name}: {e}")
        return False


def check_all_csv_files_readable(csv_files: List[Path], subfolder_name: str) -> None:
    """
    Verify that all CSV files in a list are readable.
    
    Args:
        csv_files: List of CSV file paths
        subfolder_name: Name of subfolder (for logging)
        
    Raises:
        RuntimeError: If any CSV file is not readable
    """
    logger.info(f"Verifying readability of CSV files in {subfolder_name}...")
    
    unreadable_files = []
    
    for csv_file in csv_files:
        if not verify_csv_readable(csv_file):
            unreadable_files.append(csv_file.name)
    
    if unreadable_files:
        error_msg = (
            f"❌ Some CSV files in {subfolder_name} are not readable:\n" +
            "\n".join(f"   - {filename}" for filename in unreadable_files) +
            "\n\nPlease check file permissions and integrity."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"✅ All CSV files in {subfolder_name} are readable")


def verify_dataset_structure(dataset_path: Optional[Path] = None, skip_readability: bool = False) -> Dict[str, Any]:
    """
    Verify complete dataset folder structure and integrity.
    
    Performs comprehensive validation:
    1. Dataset directory exists
    2. All required subfolders exist
    3. Each subfolder contains at least one CSV file
    4. All CSV files are readable (optional)
    
    Args:
        dataset_path: Optional custom dataset path (defaults to project dataset/)
        skip_readability: If True, skip CSV readability checks (faster)
        
    Returns:
        dict: Verification report with statistics
        
    Raises:
        RuntimeError: If any validation check fails
        
    Example:
        >>> from verify_dataset import verify_dataset_structure
        >>> report = verify_dataset_structure()
        >>> print(f"Total CSV files: {report['total_csv_files']}")
    """
    logger.info("=" * 80)
    logger.info("ASEWIS Dataset Verification")
    logger.info("=" * 80)
    
    # Step 1: Get and verify dataset directory
    if dataset_path is None:
        dataset_path = get_dataset_path()
    
    logger.info(f"\n📁 Verifying dataset directory: {dataset_path.absolute()}\n")
    check_directory_exists(dataset_path, "Dataset directory")
    
    # Step 2: Verify required subfolders
    logger.info(f"\n📂 Verifying required subfolders...\n")
    
    subfolder_paths = {}
    for subfolder_name in REQUIRED_SUBFOLDERS:
        subfolder_path = check_subfolder_exists(dataset_path, subfolder_name)
        subfolder_paths[subfolder_name] = subfolder_path
    
    logger.info(f"\n✅ All {len(REQUIRED_SUBFOLDERS)} required subfolders exist\n")
    
    # Step 3: Check CSV files in each subfolder
    logger.info(f"📄 Verifying CSV files in each subfolder...\n")
    
    csv_files_by_folder = {}
    total_csv_files = 0
    total_size_bytes = 0
    
    for subfolder_name, subfolder_path in subfolder_paths.items():
        csv_files = check_csv_files_exist(subfolder_path, subfolder_name)
        csv_files_by_folder[subfolder_name] = csv_files
        total_csv_files += len(csv_files)
        
        # Calculate total size
        for csv_file in csv_files:
            total_size_bytes += csv_file.stat().st_size
    
    logger.info(f"\n✅ Found {total_csv_files} CSV files across all subfolders\n")
    
    # Step 4: Verify CSV readability (optional)
    if not skip_readability:
        logger.info(f"🔍 Verifying CSV file readability...\n")
        
        for subfolder_name, csv_files in csv_files_by_folder.items():
            check_all_csv_files_readable(csv_files, subfolder_name)
        
        logger.info(f"\n✅ All CSV files are readable\n")
    else:
        logger.info(f"⏭️  Skipping CSV readability checks (skip_readability=True)\n")
    
    # Generate report
    report = {
        'dataset_path': str(dataset_path.absolute()),
        'required_subfolders': REQUIRED_SUBFOLDERS,
        'subfolders_found': list(subfolder_paths.keys()),
        'total_csv_files': total_csv_files,
        'total_size_mb': total_size_bytes / 1024 / 1024,
        'csv_files_by_folder': {
            name: [str(f.name) for f in files]
            for name, files in csv_files_by_folder.items()
        },
        'verification_status': 'PASSED'
    }
    
    # Summary
    logger.info("=" * 80)
    logger.info("Verification Summary")
    logger.info("=" * 80)
    logger.info(f"Dataset Path: {report['dataset_path']}")
    logger.info(f"Required Subfolders: {len(report['required_subfolders'])}")
    logger.info(f"Subfolders Verified: {len(report['subfolders_found'])}")
    logger.info(f"Total CSV Files: {report['total_csv_files']}")
    logger.info(f"Total Data Size: {report['total_size_mb']:.2f} MB")
    logger.info(f"Status: ✅ {report['verification_status']}")
    logger.info("=" * 80)
    
    logger.info("\n✅ Dataset verification complete - All checks passed!\n")
    
    return report


if __name__ == "__main__":
    """
    Direct execution: Verify dataset structure and report results.
    
    Usage:
        python verify_dataset.py
        python verify_dataset.py --skip-readability
    """
    
    # Parse command line arguments
    skip_readability = '--skip-readability' in sys.argv
    
    try:
        # Run verification
        report = verify_dataset_structure(skip_readability=skip_readability)
        
        # Success - print detailed report
        print("\n" + "=" * 80)
        print("DETAILED REPORT")
        print("=" * 80)
        
        for subfolder_name, csv_files in report['csv_files_by_folder'].items():
            print(f"\n📂 {subfolder_name}:")
            for csv_file in csv_files:
                print(f"   ✅ {csv_file}")
        
        print("\n" + "=" * 80)
        print("✅ VERIFICATION SUCCESSFUL")
        print("=" * 80)
        print("\nDataset is ready for processing.")
        print("You can now run the data pipeline:")
        print("  python src/aggregation.py")
        print("  python src/feature_engineering.py")
        print("  python src/scoring.py")
        print("\n")
        
        sys.exit(0)
    
    except RuntimeError as e:
        # Verification failed
        print("\n" + "=" * 80)
        print("❌ VERIFICATION FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        print("\nPlease fix the issues above before running the data pipeline.")
        print("\n")
        
        sys.exit(1)
    
    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error during verification: {e}", exc_info=True)
        print("\n" + "=" * 80)
        print("❌ UNEXPECTED ERROR")
        print("=" * 80)
        print(f"\nError: {e}")
        print("\n")
        
        sys.exit(1)
