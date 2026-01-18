"""
Intelligence Pipeline Execution Script

Validates and generates intelligence outputs from aggregated district metrics.
This script processes district-level data through feature engineering, anomaly
detection, and scoring to produce actionable insights for the dashboard.

Pipeline Stages:
1. Load aggregated district metrics from Parquet
2. Compute engineered features (explainable, no ML)
3. Detect anomalies across multiple dimensions
4. Compute NASRI (readiness) and ASRS (risk) scores
5. Validate outputs and report key insights
6. Save processed results for dashboard consumption

Design Principles:
- Fast execution: Complete in under 30 seconds
- Fail-fast validation: Detect data quality issues immediately
- Comprehensive logging: Track progress and diagnostics
- Production-ready: Robust error handling
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from scoring import compute_nasri, compute_asrs
from anomaly import detect_anomalies, get_anomaly_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent.parent / "logs" / f"intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    ]
)
logger = logging.getLogger(__name__)


# Configuration
DATA_CACHE_DIR = Path(__file__).parent.parent / "data_cache"
PROCESSED_DIR = Path(__file__).parent.parent / "dataset" / "processed"
INPUT_FILE = "district_metrics.parquet"
OUTPUT_FILES = {
    'features': 'features.parquet',
    'anomalies': 'anomalies.parquet',
    'scores': 'scores.parquet'
}


def load_district_metrics() -> pd.DataFrame:
    """
    Load aggregated district metrics from Parquet.
    
    Returns:
        pd.DataFrame: District-month aggregated metrics
        
    Raises:
        FileNotFoundError: If input file does not exist
        RuntimeError: If file cannot be loaded
    """
    input_path = DATA_CACHE_DIR / INPUT_FILE
    
    if not input_path.exists():
        error_msg = (
            f"District metrics file not found: {input_path}\n"
            f"Please run the data pipeline first:\n"
            f"  python src/run_pipeline.py"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"Loading district metrics from: {input_path}")
        df = pd.read_parquet(input_path)
        
        file_size_mb = input_path.stat().st_size / 1024 / 1024
        logger.info(f"✅ Loaded {len(df):,} records ({file_size_mb:.2f} MB)")
        logger.info(f"   Columns: {len(df.columns)}")
        logger.info(f"   Shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to load district metrics: {e}")
        raise RuntimeError(f"Cannot load input file: {e}")


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engineered features from district metrics.
    
    Features computed:
    - Eligibility projections (demographic cohorts)
    - Completion ratios (service delivery efficiency)
    - Demand velocity (trend analysis)
    - Compliance debt (backlog accumulation)
    - Stability scores (consistency metrics)
    - Capacity proxy (operational pressure)
    
    Args:
        df: District metrics DataFrame
        
    Returns:
        pd.DataFrame: Metrics with additional feature columns
    """
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 80 + "\n")
    
    start_time = time.time()
    
    try:
        # Import feature engineering functions
        from intelligence.feature_engineering import (
            compute_eligibility_projections,
            compute_completion_ratios,
            compute_demand_velocity,
            compute_compliance_debt,
            compute_stability_score,
            compute_capacity_proxy
        )
        
        df_features = df.copy()
        
        # Feature 1: Eligibility Projections
        logger.info("[1/6] Computing eligibility projections...")
        df_features = compute_eligibility_projections(df_features)
        
        # Feature 2: Completion Ratios
        logger.info("[2/6] Computing completion ratios...")
        df_features = compute_completion_ratios(df_features)
        
        # Feature 3: Demand Velocity
        logger.info("[3/6] Computing demand velocity...")
        df_features = compute_demand_velocity(df_features)
        
        # Feature 4: Compliance Debt
        logger.info("[4/6] Computing compliance debt...")
        df_features = compute_compliance_debt(df_features)
        
        # Feature 5: Stability Score
        logger.info("[5/6] Computing stability score...")
        df_features = compute_stability_score(df_features)
        
        # Feature 6: Capacity Proxy
        logger.info("[6/6] Computing capacity proxy...")
        df_features = compute_capacity_proxy(df_features)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Feature engineering complete: {elapsed:.2f}s")
        
        # Count new features added
        new_features = len(df_features.columns) - len(df.columns)
        logger.info(f"   Added {new_features} feature columns")
        logger.info(f"   Total columns: {len(df_features.columns)}")
        
        return df_features
    
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        raise RuntimeError(f"Feature computation failed: {e}")


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies across multiple dimensions.
    
    Detection Methods:
    - Z-score (statistical outliers)
    - Percentile thresholds (extreme values)
    - Peer comparison (district vs district)
    
    Args:
        df: DataFrame with features
        
    Returns:
        pd.DataFrame: Data with anomaly flags and scores
    """
    logger.info("\n" + "=" * 80)
    logger.info("ANOMALY DETECTION")
    logger.info("=" * 80 + "\n")
    
    start_time = time.time()
    
    try:
        logger.info("Running multi-method anomaly detection...")
        df_anomalies = detect_anomalies(df)
        
        # Get anomaly summary
        summary = get_anomaly_summary(df_anomalies)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Anomaly detection complete: {elapsed:.2f}s")
        
        # Report findings
        logger.info("\nAnomaly Summary:")
        logger.info(f"   Total districts analyzed: {summary.get('total_districts', 0)}")
        logger.info(f"   Districts with anomalies: {summary.get('districts_with_anomalies', 0)}")
        logger.info(f"   Z-score anomalies: {summary.get('zscore_anomaly_count', 0)}")
        logger.info(f"   Percentile anomalies: {summary.get('percentile_anomaly_count', 0)}")
        logger.info(f"   Peer anomalies: {summary.get('peer_anomaly_count', 0)}")
        
        return df_anomalies
    
    except Exception as e:
        logger.error(f"❌ Anomaly detection failed: {e}")
        raise RuntimeError(f"Anomaly detection failed: {e}")


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NASRI and ASRS scores for each district-month.
    
    NASRI (0-100): National Aadhar System Readiness Index
    - Higher is better
    - Components: completion, capacity, stability, velocity, debt
    
    ASRS (0-1): Aadhar System Risk Score
    - Lower is better
    - Components: capacity stress, instability, compliance gap, negative velocity
    
    Args:
        df: DataFrame with features
        
    Returns:
        pd.DataFrame: Data with NASRI and ASRS scores
    """
    logger.info("\n" + "=" * 80)
    logger.info("SCORING: NASRI & ASRS")
    logger.info("=" * 80 + "\n")
    
    start_time = time.time()
    
    try:
        logger.info("Computing NASRI scores (readiness index)...")
        df_scores = compute_nasri(df)
        
        logger.info("Computing ASRS scores (risk probability)...")
        df_scores = compute_asrs(df_scores)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Scoring complete: {elapsed:.2f}s")
        
        # Report score statistics
        logger.info("\nScore Statistics:")
        logger.info(f"   NASRI - Mean: {df_scores['nasri_score'].mean():.2f}, "
                   f"Std: {df_scores['nasri_score'].std():.2f}, "
                   f"Min: {df_scores['nasri_score'].min():.2f}, "
                   f"Max: {df_scores['nasri_score'].max():.2f}")
        logger.info(f"   ASRS  - Mean: {df_scores['asrs_score'].mean():.3f}, "
                   f"Std: {df_scores['asrs_score'].std():.3f}, "
                   f"Min: {df_scores['asrs_score'].min():.3f}, "
                   f"Max: {df_scores['asrs_score'].max():.3f}")
        
        return df_scores
    
    except Exception as e:
        logger.error(f"❌ Scoring failed: {e}")
        raise RuntimeError(f"Score computation failed: {e}")


def validate_scores(df: pd.DataFrame) -> None:
    """
    Validate that scores are computed correctly.
    
    Validation Checks:
    - No NaN values in NASRI or ASRS
    - NASRI in valid range [0, 100]
    - ASRS in valid range [0, 1]
    - All districts have scores
    
    Args:
        df: DataFrame with scores
        
    Raises:
        RuntimeError: If validation fails
    """
    logger.info("\n" + "=" * 80)
    logger.info("SCORE VALIDATION")
    logger.info("=" * 80 + "\n")
    
    # Check for required columns
    if 'nasri_score' not in df.columns:
        raise RuntimeError("NASRI score column missing")
    if 'asrs_score' not in df.columns:
        raise RuntimeError("ASRS score column missing")
    
    # Check for NaN values
    nasri_nans = df['nasri_score'].isna().sum()
    asrs_nans = df['asrs_score'].isna().sum()
    
    if nasri_nans > 0:
        logger.error(f"❌ Found {nasri_nans} NaN values in NASRI scores")
        raise RuntimeError(f"NASRI scores contain {nasri_nans} NaN values - data quality issue")
    
    if asrs_nans > 0:
        logger.error(f"❌ Found {asrs_nans} NaN values in ASRS scores")
        raise RuntimeError(f"ASRS scores contain {asrs_nans} NaN values - data quality issue")
    
    logger.info("✅ No NaN values found in scores")
    
    # Check score ranges
    nasri_out_of_range = ((df['nasri_score'] < 0) | (df['nasri_score'] > 100)).sum()
    asrs_out_of_range = ((df['asrs_score'] < 0) | (df['asrs_score'] > 1)).sum()
    
    if nasri_out_of_range > 0:
        logger.error(f"❌ Found {nasri_out_of_range} NASRI scores out of range [0, 100]")
        raise RuntimeError(f"NASRI scores out of valid range")
    
    if asrs_out_of_range > 0:
        logger.error(f"❌ Found {asrs_out_of_range} ASRS scores out of range [0, 1]")
        raise RuntimeError(f"ASRS scores out of valid range")
    
    logger.info("✅ All scores within valid ranges")
    logger.info("   NASRI: [0, 100]")
    logger.info("   ASRS: [0, 1]")
    
    logger.info(f"\n✅ Validation passed - {len(df):,} records validated")


def report_key_insights(df: pd.DataFrame) -> None:
    """
    Print key insights and summary statistics.
    
    Reports:
    - Top 5 highest ASRS districts (highest risk)
    - NASRI distribution summary
    - Critical alerts
    
    Args:
        df: DataFrame with scores
    """
    logger.info("\n" + "=" * 80)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 80 + "\n")
    
    # Get latest data per district
    if 'date' in df.columns:
        latest_df = df.sort_values('date').groupby('district').tail(1)
    else:
        latest_df = df
    
    # Top 5 Highest ASRS Districts (Highest Risk)
    logger.info("🚨 TOP 5 HIGHEST RISK DISTRICTS (ASRS):")
    logger.info("-" * 80)
    
    top_risk = latest_df.nlargest(5, 'asrs_score')[['district', 'asrs_score', 'nasri_score']]
    
    for idx, (_, row) in enumerate(top_risk.iterrows(), 1):
        district = row.get('district', 'Unknown')
        asrs = row.get('asrs_score', 0)
        nasri = row.get('nasri_score', 0)
        logger.info(f"  {idx}. {district}")
        logger.info(f"     ASRS: {asrs:.3f} | NASRI: {nasri:.1f}")
    
    # NASRI Distribution Summary
    logger.info("\n📊 NASRI READINESS DISTRIBUTION:")
    logger.info("-" * 80)
    
    nasri_bins = [0, 40, 60, 80, 100]
    nasri_labels = ['Critical (0-40)', 'Low (40-60)', 'Medium (60-80)', 'High (80-100)']
    latest_df['nasri_category'] = pd.cut(latest_df['nasri_score'], bins=nasri_bins, labels=nasri_labels)
    
    distribution = latest_df['nasri_category'].value_counts().sort_index()
    total = len(latest_df)
    
    for category, count in distribution.items():
        pct = count / total * 100
        logger.info(f"  {category:20s}: {count:4d} districts ({pct:5.1f}%)")
    
    # Critical Alerts
    logger.info("\n⚠️  CRITICAL ALERTS:")
    logger.info("-" * 80)
    
    critical_count = (latest_df['asrs_score'] > 0.7).sum()
    low_readiness = (latest_df['nasri_score'] < 40).sum()
    
    logger.info(f"  Districts with ASRS > 0.7 (critical risk): {critical_count}")
    logger.info(f"  Districts with NASRI < 40 (critical readiness): {low_readiness}")
    
    if critical_count > 0:
        logger.warning(f"  ⚠️  {critical_count} districts require immediate intervention")
    else:
        logger.info(f"  ✅ No critical risk districts")


def save_outputs(df_features: pd.DataFrame, df_anomalies: pd.DataFrame, df_scores: pd.DataFrame) -> None:
    """
    Save processed outputs to Parquet for dashboard consumption.
    
    Args:
        df_features: DataFrame with engineered features
        df_anomalies: DataFrame with anomaly flags
        df_scores: DataFrame with NASRI/ASRS scores
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 80 + "\n")
    
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    outputs = {
        'features': df_features,
        'anomalies': df_anomalies,
        'scores': df_scores
    }
    
    for output_type, df in outputs.items():
        output_path = PROCESSED_DIR / OUTPUT_FILES[output_type]
        
        try:
            df.to_parquet(output_path, compression='snappy', index=False)
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"✅ Saved {output_type}: {output_path}")
            logger.info(f"   {len(df):,} records, {file_size_mb:.2f} MB")
        except Exception as e:
            logger.error(f"❌ Failed to save {output_type}: {e}")
            raise RuntimeError(f"Cannot save {output_type}: {e}")
    
    logger.info(f"\n✅ All outputs saved to: {PROCESSED_DIR}")


def main() -> int:
    """
    Execute the intelligence pipeline.
    
    Returns:
        int: Exit code (0 = success, 1 = failure)
    """
    pipeline_start_time = time.time()
    
    logger.info("\n" + "=" * 80)
    logger.info("ASEWIS INTELLIGENCE PIPELINE")
    logger.info("Feature Engineering → Anomaly Detection → Scoring")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")
    
    try:
        # Step 1: Load district metrics
        df = load_district_metrics()
        
        # Step 2: Compute features
        df_features = compute_features(df)
        
        # Step 3: Run anomaly detection
        df_anomalies = run_anomaly_detection(df_features)
        
        # Step 4: Compute NASRI & ASRS scores
        df_scores = compute_scores(df_anomalies)
        
        # Step 5: Validate scores
        validate_scores(df_scores)
        
        # Step 6: Report key insights
        report_key_insights(df_scores)
        
        # Step 7: Save outputs
        save_outputs(df_features, df_anomalies, df_scores)
        
        # Pipeline completion
        pipeline_elapsed = time.time() - pipeline_start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {pipeline_elapsed:.2f}s")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if pipeline_elapsed > 30:
            logger.warning(f"⚠️  Pipeline exceeded 30s target ({pipeline_elapsed:.2f}s)")
        else:
            logger.info(f"✅ Pipeline completed within 30s target")
        
        logger.info("=" * 80)
        
        logger.info("\n✅ INTELLIGENCE PIPELINE SUCCESSFUL\n")
        logger.info("Next step:")
        logger.info("  streamlit run app/streamlit_app.py\n")
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)\n")
        return 130
    
    except Exception as e:
        logger.error(f"\n\n❌ PIPELINE FAILED: {e}", exc_info=True)
        logger.error("\nPlease fix the error and restart the pipeline\n")
        return 1


if __name__ == "__main__":
    """
    Direct execution entry point.
    
    Usage:
        python src/run_intelligence.py
    """
    exit_code = main()
    sys.exit(exit_code)
