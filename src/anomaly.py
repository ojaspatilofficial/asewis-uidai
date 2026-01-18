"""
Anomaly detection module for data quality and operational monitoring.

This module provides statistical anomaly detection methods to identify unusual
patterns in Aadhar enrollment and update data. Uses multiple detection strategies
to provide robust anomaly identification with explainable reasons.
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


def detect_zscore_anomalies(
    series: pd.Series,
    threshold: float = 3.0,
    column_name: str = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect anomalies using Z-score method.
    
    Method: Standard Score (Z-score)
    Measures how many standard deviations a value is from the mean.
    
    Formula:
    Z = (X - μ) / σ
    Where:
    - X = observation value
    - μ = mean of distribution
    - σ = standard deviation
    
    Detection Rule:
    - |Z| > threshold → Anomaly
    - Typically threshold = 3 (99.7% of normal data within ±3σ)
    
    Assumptions:
    - Data approximately normally distributed
    - Mean and std dev represent "normal" behavior
    - Outliers are rare (< 0.3% of data)
    
    Pros:
    - Simple and interpretable
    - Works well for unimodal distributions
    - Fast computation
    
    Cons:
    - Sensitive to extreme outliers (affects mean/std)
    - Assumes normal distribution
    - May miss anomalies in skewed data
    
    Args:
        series: Numeric series to analyze
        threshold: Z-score threshold for anomaly (default: 3.0)
        column_name: Name of column for logging
        
    Returns:
        Tuple of (anomaly_flags, reasons) as boolean and string Series
    """
    col_name = column_name or 'value'
    
    # Calculate mean and standard deviation
    mean_val = series.mean()
    std_val = series.std()
    
    # Handle case where std is 0 (all values same)
    if std_val == 0 or pd.isna(std_val):
        logger.debug(f"Z-score: {col_name} has zero variance, no anomalies")
        return pd.Series(False, index=series.index), pd.Series('', index=series.index)
    
    # Calculate Z-scores
    z_scores = (series - mean_val) / std_val
    
    # Detect anomalies: absolute Z-score exceeds threshold
    anomaly_flags = np.abs(z_scores) > threshold
    
    # Generate reasons for anomalies
    reasons = pd.Series('', index=series.index)
    
    # High anomalies (positive spike)
    high_mask = z_scores > threshold
    reasons[high_mask] = reasons[high_mask].apply(
        lambda x: f"Z-score spike: {z_scores[high_mask.index[0]]:.2f}σ above mean"
    )
    
    # Low anomalies (negative spike)
    low_mask = z_scores < -threshold
    reasons[low_mask] = reasons[low_mask].apply(
        lambda x: f"Z-score drop: {abs(z_scores[low_mask.index[0]]):.2f}σ below mean"
    )
    
    anomaly_count = anomaly_flags.sum()
    if anomaly_count > 0:
        logger.debug(
            f"Z-score: {col_name} - {anomaly_count} anomalies "
            f"(mean={mean_val:.1f}, std={std_val:.1f}, threshold={threshold})"
        )
    
    return anomaly_flags, reasons


def detect_percentile_anomalies(
    series: pd.Series,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    column_name: str = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect anomalies using percentile thresholds.
    
    Method: Percentile-based Outlier Detection
    Flags values outside the normal range defined by percentiles.
    
    Detection Rule:
    - Value < P_lower → Low anomaly
    - Value > P_upper → High anomaly
    - Typically: P_lower = 1%, P_upper = 99%
    
    Assumptions:
    - Extreme percentiles represent abnormal values
    - Distribution shape doesn't matter (non-parametric)
    - Historical data captures normal operating range
    
    Pros:
    - Robust to distribution shape (works with skewed data)
    - Not affected by extreme outliers
    - Easy to interpret (e.g., "top 1% unusual")
    
    Cons:
    - Fixed percentage of data always flagged
    - Less sensitive than Z-score for normal distributions
    - Requires sufficient historical data
    
    Args:
        series: Numeric series to analyze
        lower_percentile: Lower threshold percentile (default: 1.0)
        upper_percentile: Upper threshold percentile (default: 99.0)
        column_name: Name of column for logging
        
    Returns:
        Tuple of (anomaly_flags, reasons) as boolean and string Series
    """
    col_name = column_name or 'value'
    
    # Calculate percentile thresholds
    lower_threshold = series.quantile(lower_percentile / 100.0)
    upper_threshold = series.quantile(upper_percentile / 100.0)
    
    # Detect anomalies: values outside percentile range
    low_anomalies = series < lower_threshold
    high_anomalies = series > upper_threshold
    anomaly_flags = low_anomalies | high_anomalies
    
    # Generate reasons
    reasons = pd.Series('', index=series.index)
    
    reasons[low_anomalies] = reasons[low_anomalies].apply(
        lambda x: f"Below {lower_percentile}th percentile ({lower_threshold:.1f})"
    )
    
    reasons[high_anomalies] = reasons[high_anomalies].apply(
        lambda x: f"Above {upper_percentile}th percentile ({upper_threshold:.1f})"
    )
    
    anomaly_count = anomaly_flags.sum()
    if anomaly_count > 0:
        logger.debug(
            f"Percentile: {col_name} - {anomaly_count} anomalies "
            f"(thresholds: [{lower_threshold:.1f}, {upper_threshold:.1f}])"
        )
    
    return anomaly_flags, reasons


def detect_peer_comparison_anomalies(
    df: pd.DataFrame,
    value_column: str,
    peer_group_column: str,
    threshold_multiplier: float = 2.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect anomalies by comparing to peer districts.
    
    Method: Peer Group Comparison
    Compares each district to similar districts (peers) to identify outliers.
    
    Detection Logic:
    1. Group districts by peer characteristic (e.g., state, region)
    2. Calculate peer group statistics (median, IQR)
    3. Flag districts significantly different from peers
    
    Detection Rule:
    - Value > Peer_Median + threshold * IQR → High anomaly
    - Value < Peer_Median - threshold * IQR → Low anomaly
    - IQR = Interquartile Range (P75 - P25)
    
    Why IQR:
    - Robust to outliers (uses quartiles, not mean/std)
    - Works with skewed distributions
    - Industry standard for peer comparison
    
    Assumptions:
    - Peer districts have similar normal operating ranges
    - Significant deviation from peers indicates anomaly
    - Peer groups are meaningful (not arbitrary)
    
    Use Cases:
    - District X has 10x more enrollments than similar districts
    - Rural district showing urban-level activity (potential error)
    - Peer comparison reveals relative under-performance
    
    Pros:
    - Context-aware (compares similar entities)
    - Robust to different scales across peer groups
    - Identifies relative anomalies
    
    Cons:
    - Requires meaningful peer grouping
    - May miss anomalies if whole peer group is anomalous
    - Needs sufficient peers per group (min 3-5)
    
    Args:
        df: DataFrame with data to analyze
        value_column: Column to check for anomalies
        peer_group_column: Column defining peer groups (e.g., state)
        threshold_multiplier: IQR multiplier for threshold (default: 2.0)
        
    Returns:
        Tuple of (anomaly_flags, reasons) as boolean and string Series
    """
    # Calculate peer group statistics
    # For each peer group, compute median and IQR
    peer_stats = df.groupby(peer_group_column)[value_column].agg([
        ('median', 'median'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ])
    
    # Calculate IQR (Interquartile Range)
    peer_stats['iqr'] = peer_stats['q75'] - peer_stats['q25']
    
    # Calculate thresholds for each peer group
    peer_stats['lower_bound'] = peer_stats['median'] - threshold_multiplier * peer_stats['iqr']
    peer_stats['upper_bound'] = peer_stats['median'] + threshold_multiplier * peer_stats['iqr']
    
    # Merge peer stats back to original dataframe
    df_with_peers = df.merge(
        peer_stats,
        left_on=peer_group_column,
        right_index=True,
        how='left'
    )
    
    # Detect anomalies by comparing to peer bounds
    low_anomalies = df_with_peers[value_column] < df_with_peers['lower_bound']
    high_anomalies = df_with_peers[value_column] > df_with_peers['upper_bound']
    anomaly_flags = low_anomalies | high_anomalies
    
    # Generate reasons
    reasons = pd.Series('', index=df.index)
    
    # Low anomalies
    low_mask = low_anomalies[low_anomalies].index
    for idx in low_mask:
        peer_group = df_with_peers.loc[idx, peer_group_column]
        peer_median = df_with_peers.loc[idx, 'median']
        actual = df_with_peers.loc[idx, value_column]
        deviation_pct = ((actual - peer_median) / peer_median * 100) if peer_median > 0 else 0
        reasons[idx] = f"Below peer group ({peer_group}) median by {abs(deviation_pct):.1f}%"
    
    # High anomalies
    high_mask = high_anomalies[high_anomalies].index
    for idx in high_mask:
        peer_group = df_with_peers.loc[idx, peer_group_column]
        peer_median = df_with_peers.loc[idx, 'median']
        actual = df_with_peers.loc[idx, value_column]
        deviation_pct = ((actual - peer_median) / peer_median * 100) if peer_median > 0 else 0
        reasons[idx] = f"Above peer group ({peer_group}) median by {deviation_pct:.1f}%"
    
    anomaly_count = anomaly_flags.sum()
    if anomaly_count > 0:
        logger.debug(
            f"Peer comparison: {value_column} - {anomaly_count} anomalies "
            f"(threshold: {threshold_multiplier} * IQR)"
        )
    
    return anomaly_flags, reasons


def detect_anomalies(
    df: pd.DataFrame,
    value_columns: Optional[List[str]] = None,
    methods: List[str] = ['zscore', 'percentile', 'peer'],
    zscore_threshold: float = 3.0,
    percentile_lower: float = 1.0,
    percentile_upper: float = 99.0,
    peer_column: str = None
) -> pd.DataFrame:
    """
    Detect anomalies using multiple statistical methods.
    
    This is the main anomaly detection entry point that:
    1. Applies multiple detection methods to specified columns
    2. Combines results using OR logic (any method flags → anomaly)
    3. Provides detailed reasons for each anomaly
    4. Returns DataFrame with anomaly flags and explanations
    
    Detection Strategy:
    - Use multiple methods for robust detection
    - Each method captures different anomaly types
    - Combined approach reduces false negatives
    
    Methods:
    1. Z-score: Catches extreme statistical outliers
    2. Percentile: Robust to distribution shape
    3. Peer comparison: Context-aware relative anomalies
    
    Output Columns Added:
    - anomaly_flag: Boolean indicating if any anomaly detected
    - anomaly_reason: Detailed explanation of anomaly
    - anomaly_methods: Comma-separated list of methods that flagged
    - anomaly_severity: Count of methods that flagged (1-3)
    
    Args:
        df: DataFrame with data to analyze
        value_columns: Columns to check for anomalies (default: auto-detect count columns)
        methods: List of detection methods to use (default: all)
        zscore_threshold: Z-score threshold (default: 3.0)
        percentile_lower: Lower percentile threshold (default: 1.0)
        percentile_upper: Upper percentile threshold (default: 99.0)
        peer_column: Column for peer grouping (default: auto-detect state/region)
        
    Returns:
        pd.DataFrame: Input DataFrame with anomaly detection columns added
        
    Example:
        >>> df_with_anomalies = detect_anomalies(
        ...     df,
        ...     value_columns=['enrolment_count'],
        ...     methods=['zscore', 'percentile', 'peer']
        ... )
        >>> anomalies = df_with_anomalies[df_with_anomalies['anomaly_flag']]
        >>> print(anomalies[['district', 'anomaly_reason']])
    """
    logger.info(f"Starting anomaly detection on {len(df)} records")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for anomaly detection")
        df['anomaly_flag'] = False
        df['anomaly_reason'] = ''
        return df
    
    # Create a copy to avoid modifying original
    df_result = df.copy()
    
    # Auto-detect value columns if not specified
    if value_columns is None:
        # Use count columns by default
        value_columns = [col for col in df.columns if 'count' in col.lower()]
        if not value_columns:
            # Fallback: use all numeric columns
            value_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Auto-detected value columns: {value_columns}")
    
    if not value_columns:
        logger.warning("No numeric columns found for anomaly detection")
        df_result['anomaly_flag'] = False
        df_result['anomaly_reason'] = ''
        return df_result
    
    # Auto-detect peer column if not specified and peer method enabled
    if 'peer' in methods and peer_column is None:
        # Look for state or region columns
        for col in df.columns:
            if any(kw in col.lower() for kw in ['state', 'region', 'zone']):
                peer_column = col
                logger.info(f"Auto-detected peer column: {peer_column}")
                break
    
    # Initialize anomaly tracking columns
    df_result['anomaly_flag'] = False
    df_result['anomaly_reason'] = ''
    df_result['anomaly_methods'] = ''
    df_result['anomaly_severity'] = 0
    
    # Apply anomaly detection for each value column
    for col in value_columns:
        if col not in df_result.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df_result[col]):
            logger.warning(f"Column {col} is not numeric, skipping")
            continue
        
        logger.info(f"Analyzing column: {col}")
        
        # Track anomalies from all methods
        method_flags = []
        method_reasons = []
        
        # Method 1: Z-score detection
        if 'zscore' in methods:
            try:
                zscore_flags, zscore_reasons = detect_zscore_anomalies(
                    df_result[col],
                    threshold=zscore_threshold,
                    column_name=col
                )
                method_flags.append(('zscore', zscore_flags))
                method_reasons.append(zscore_reasons)
            except Exception as e:
                logger.error(f"Z-score detection failed for {col}: {e}")
        
        # Method 2: Percentile detection
        if 'percentile' in methods:
            try:
                percentile_flags, percentile_reasons = detect_percentile_anomalies(
                    df_result[col],
                    lower_percentile=percentile_lower,
                    upper_percentile=percentile_upper,
                    column_name=col
                )
                method_flags.append(('percentile', percentile_flags))
                method_reasons.append(percentile_reasons)
            except Exception as e:
                logger.error(f"Percentile detection failed for {col}: {e}")
        
        # Method 3: Peer comparison detection
        if 'peer' in methods and peer_column is not None:
            try:
                peer_flags, peer_reasons = detect_peer_comparison_anomalies(
                    df_result,
                    value_column=col,
                    peer_group_column=peer_column,
                    threshold_multiplier=2.0
                )
                method_flags.append(('peer', peer_flags))
                method_reasons.append(peer_reasons)
            except Exception as e:
                logger.error(f"Peer comparison detection failed for {col}: {e}")
        
        # Combine results from all methods
        if method_flags:
            # OR logic: anomaly if ANY method flags it
            combined_flags = pd.Series(False, index=df_result.index)
            for method_name, flags in method_flags:
                combined_flags = combined_flags | flags
            
            # Update anomaly flag
            df_result['anomaly_flag'] = df_result['anomaly_flag'] | combined_flags
            
            # Compile reasons and severity for rows with anomalies
            for idx in combined_flags[combined_flags].index:
                active_methods = []
                active_reasons = []
                
                for (method_name, flags), reasons in zip(method_flags, method_reasons):
                    if flags[idx]:
                        active_methods.append(method_name)
                        if reasons[idx]:
                            active_reasons.append(reasons[idx])
                
                # Update reason (combine all method reasons)
                if active_reasons:
                    combined_reason = f"{col}: " + "; ".join(active_reasons)
                    if df_result.loc[idx, 'anomaly_reason']:
                        df_result.loc[idx, 'anomaly_reason'] += f" | {combined_reason}"
                    else:
                        df_result.loc[idx, 'anomaly_reason'] = combined_reason
                
                # Update methods list
                if active_methods:
                    methods_str = ",".join(active_methods)
                    if df_result.loc[idx, 'anomaly_methods']:
                        df_result.loc[idx, 'anomaly_methods'] += f",{methods_str}"
                    else:
                        df_result.loc[idx, 'anomaly_methods'] = methods_str
                
                # Update severity (count of methods)
                df_result.loc[idx, 'anomaly_severity'] += len(active_methods)
    
    # Summary statistics
    total_anomalies = df_result['anomaly_flag'].sum()
    anomaly_rate = (total_anomalies / len(df_result) * 100) if len(df_result) > 0 else 0
    
    logger.info(
        f"Anomaly detection complete: {total_anomalies} anomalies detected "
        f"({anomaly_rate:.2f}% of records)"
    )
    
    # Log severity distribution
    if total_anomalies > 0:
        severity_dist = df_result[df_result['anomaly_flag']]['anomaly_severity'].value_counts().sort_index()
        logger.info(f"Severity distribution: {severity_dist.to_dict()}")
    
    return df_result


def get_anomaly_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for detected anomalies.
    
    Args:
        df: DataFrame with anomaly detection results
        
    Returns:
        dict: Summary statistics including counts, rates, and top anomalies
    """
    if 'anomaly_flag' not in df.columns:
        logger.warning("DataFrame does not have anomaly_flag column")
        return {'error': 'No anomaly detection results found'}
    
    anomalies = df[df['anomaly_flag']]
    
    summary = {
        'total_records': len(df),
        'total_anomalies': len(anomalies),
        'anomaly_rate_pct': (len(anomalies) / len(df) * 100) if len(df) > 0 else 0,
        'severity_distribution': anomalies['anomaly_severity'].value_counts().to_dict() if len(anomalies) > 0 else {},
        'methods_distribution': {},
        'top_anomalies': []
    }
    
    # Method distribution
    if len(anomalies) > 0 and 'anomaly_methods' in anomalies.columns:
        all_methods = []
        for methods_str in anomalies['anomaly_methods']:
            all_methods.extend(methods_str.split(','))
        from collections import Counter
        summary['methods_distribution'] = dict(Counter(all_methods))
    
    # Top anomalies by severity
    if len(anomalies) > 0:
        top = anomalies.nlargest(10, 'anomaly_severity')
        summary['top_anomalies'] = top[['anomaly_severity', 'anomaly_reason']].to_dict('records')
    
    return summary


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Anomaly detection module - Example usage")
    
    # Example: Create sample data with anomalies
    logger.info("\n=== Example: Anomaly Detection ===")
    
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'district': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'state': ['S1', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S2', 'S2'],
        'enrolment_count': [100, 110, 105, 500, 108, 200, 210, 205, 215, 10],  # D and J are anomalies
        'update_count': [80, 85, 82, 88, 84, 160, 165, 162, 168, 155]
    })
    
    logger.info("Input data:")
    logger.info(f"\n{sample_data}")
    
    # Detect anomalies
    result = detect_anomalies(
        sample_data,
        value_columns=['enrolment_count'],
        methods=['zscore', 'percentile', 'peer'],
        peer_column='state'
    )
    
    logger.info("\n=== Anomaly Detection Results ===")
    logger.info(f"\n{result[['district', 'enrolment_count', 'anomaly_flag', 'anomaly_severity', 'anomaly_reason']]}")
    
    # Show only anomalies
    anomalies = result[result['anomaly_flag']]
    logger.info(f"\n=== Detected Anomalies ({len(anomalies)}) ===")
    if len(anomalies) > 0:
        logger.info(f"\n{anomalies[['district', 'state', 'enrolment_count', 'anomaly_severity', 'anomaly_reason']]}")
    
    # Get summary
    summary = get_anomaly_summary(result)
    logger.info("\n=== Anomaly Summary ===")
    for key, value in summary.items():
        if key != 'top_anomalies':
            logger.info(f"{key}: {value}")
