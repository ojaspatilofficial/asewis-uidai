"""
Feature engineering module for district-month analytics.

This module computes explainable features from aggregated district-month data.
All features are interpretable and based on domain knowledge, not machine learning.
Each feature has a clear business meaning and can be explained to stakeholders.
"""

import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def compute_eligibility_projections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project future Aadhar eligibility for age cohorts.
    
    Business Logic:
    - Children 0-5 years will need enrollment when they turn 5
    - Children 5-18 years need updates as they transition to adulthood
    - These projections help predict future demand for enrollment centers
    
    Cohorts:
    1. Ages 0-5: Will become eligible for enrollment (5-year window)
    2. Ages 5-18: Will need updates as they mature (13-year window)
    
    Formula:
    - Eligibility_0_to_5 = (count of 0-5 age group) / 5
      → Average annual enrollment need from this cohort
    - Eligibility_5_to_18 = (count of 5-18 age group) / 13
      → Average annual update need from this cohort
    
    Args:
        df: Aggregated district-month DataFrame
        
    Returns:
        pd.DataFrame: Input DataFrame with eligibility projection columns added
    """
    logger.info("Computing eligibility projections for age cohorts")
    
    df_with_features = df.copy()
    
    # Identify age group columns
    age_child_cols = [col for col in df.columns if 'age_group_child' in col.lower()]
    
    if age_child_cols:
        # Use existing age group column
        child_col = age_child_cols[0]
        
        # Project annual eligibility for 0-5 cohort
        # Divide by 5 because children spread across 5 years become eligible annually
        df_with_features['eligibility_proj_0_to_5'] = df_with_features[child_col] / 5.0
        
        # Project annual eligibility for 5-18 cohort
        # Divide by 13 because children spread across 13 years need updates annually
        df_with_features['eligibility_proj_5_to_18'] = df_with_features[child_col] / 13.0
        
        logger.info(f"✓ Added eligibility projections using {child_col}")
    else:
        # Fallback: estimate from total counts if age groups not available
        logger.warning("No age group columns found, using fallback estimation")
        
        # Look for any count columns
        count_cols = [col for col in df.columns if 'count' in col.lower()]
        if count_cols:
            total_col = count_cols[0]
            # Assume 15% of population is 0-5 (rough demographic estimate)
            df_with_features['eligibility_proj_0_to_5'] = (df_with_features[total_col] * 0.15) / 5.0
            # Assume 20% of population is 5-18
            df_with_features['eligibility_proj_5_to_18'] = (df_with_features[total_col] * 0.20) / 13.0
            logger.info("✓ Added eligibility projections using demographic estimates")
        else:
            # No data available for projections
            df_with_features['eligibility_proj_0_to_5'] = 0
            df_with_features['eligibility_proj_5_to_18'] = 0
            logger.warning("No suitable columns for eligibility projections")
    
    return df_with_features


def compute_completion_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate completion ratios for different update types.
    
    Business Logic:
    Completion ratio indicates what percentage of required updates are being completed.
    High completion (>80%) = good service delivery
    Low completion (<50%) = potential bottleneck or low awareness
    
    Ratios computed:
    1. Biometric completion = biometric_updates / total_enrolments
    2. Demographic completion = demographic_updates / total_enrolments
    3. Overall completion = (bio + demo) / (2 * enrolments)
    
    Why it matters:
    - Identifies districts with low update rates (need intervention)
    - Tracks service quality and accessibility
    - Helps prioritize resource allocation
    
    Args:
        df: Aggregated district-month DataFrame
        
    Returns:
        pd.DataFrame: Input DataFrame with completion ratio columns added
    """
    logger.info("Computing completion ratios for update types")
    
    df_with_features = df.copy()
    
    # Identify update type columns
    biometric_cols = [col for col in df.columns if 'biometric' in col.lower() and 'count' in col.lower()]
    demographic_cols = [col for col in df.columns if 'demographic' in col.lower() and 'count' in col.lower()]
    enrolment_cols = [col for col in df.columns if 'enrolment' in col.lower() and 'count' in col.lower()]
    
    # Calculate biometric completion ratio
    if biometric_cols and enrolment_cols:
        bio_col = biometric_cols[0]
        enrol_col = enrolment_cols[0]
        
        # Ratio = updates / enrolments
        # Add small epsilon to avoid division by zero
        df_with_features['completion_ratio_biometric'] = (
            df_with_features[bio_col] / (df_with_features[enrol_col] + 1e-6)
        )
        
        # Cap ratio at 1.0 (100%) for interpretability
        # Values >1 indicate more updates than enrolments (possible for backlog clearance)
        df_with_features['completion_ratio_biometric'] = df_with_features['completion_ratio_biometric'].clip(upper=1.0)
        
        logger.info(f"✓ Added biometric completion ratio ({bio_col}/{enrol_col})")
    else:
        df_with_features['completion_ratio_biometric'] = 0
        logger.warning("Cannot compute biometric completion ratio - missing columns")
    
    # Calculate demographic completion ratio
    if demographic_cols and enrolment_cols:
        demo_col = demographic_cols[0]
        enrol_col = enrolment_cols[0]
        
        df_with_features['completion_ratio_demographic'] = (
            df_with_features[demo_col] / (df_with_features[enrol_col] + 1e-6)
        )
        df_with_features['completion_ratio_demographic'] = df_with_features['completion_ratio_demographic'].clip(upper=1.0)
        
        logger.info(f"✓ Added demographic completion ratio ({demo_col}/{enrol_col})")
    else:
        df_with_features['completion_ratio_demographic'] = 0
        logger.warning("Cannot compute demographic completion ratio - missing columns")
    
    # Calculate overall completion ratio
    # Average of biometric and demographic completions
    if 'completion_ratio_biometric' in df_with_features.columns and 'completion_ratio_demographic' in df_with_features.columns:
        df_with_features['completion_ratio_overall'] = (
            df_with_features['completion_ratio_biometric'] + 
            df_with_features['completion_ratio_demographic']
        ) / 2.0
        
        logger.info("✓ Added overall completion ratio (average of bio + demo)")
    
    return df_with_features


def compute_demand_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate month-over-month demand velocity (rate of change).
    
    Business Logic:
    Velocity measures how quickly demand is growing or shrinking.
    - Positive velocity = increasing demand (need to scale up capacity)
    - Negative velocity = decreasing demand (can optimize resources)
    - High velocity (>20%) = rapid change requiring attention
    
    Formula:
    Velocity = (Current_Month - Previous_Month) / Previous_Month * 100
    
    Example:
    - Jan: 1000 enrolments, Feb: 1200 enrolments
    - Velocity = (1200-1000)/1000 * 100 = +20% growth
    
    Why it matters:
    - Early warning system for capacity planning
    - Identifies seasonal patterns
    - Helps predict future resource needs
    
    Args:
        df: Aggregated district-month DataFrame (must have district and month columns)
        
    Returns:
        pd.DataFrame: Input DataFrame with demand velocity columns added
    """
    logger.info("Computing month-over-month demand velocity")
    
    df_with_features = df.copy()
    
    # Identify district and date columns for grouping
    district_col = None
    date_col = None
    
    for col in df.columns:
        if 'district' in col.lower() and district_col is None:
            district_col = col
        if any(kw in col.lower() for kw in ['month', 'date']) and date_col is None:
            date_col = col
    
    if district_col is None or date_col is None:
        logger.warning("Cannot compute demand velocity - missing district or date column")
        df_with_features['demand_velocity_pct'] = 0
        return df_with_features
    
    # Ensure date column is sorted
    # This is critical for computing correct month-over-month changes
    df_with_features = df_with_features.sort_values([district_col, date_col])
    
    # Identify count columns to compute velocity for
    count_cols = [col for col in df.columns if 'count' in col.lower() and col not in [district_col, date_col]]
    
    if not count_cols:
        logger.warning("No count columns found for velocity computation")
        df_with_features['demand_velocity_pct'] = 0
        return df_with_features
    
    # Use first count column (typically enrolment count) for velocity
    count_col = count_cols[0]
    
    # Compute previous month value for each district
    # shift(1) moves values down by 1 row within each district group
    df_with_features['prev_month_count'] = df_with_features.groupby(district_col)[count_col].shift(1)
    
    # Calculate percentage change from previous month
    # Formula: ((current - previous) / previous) * 100
    df_with_features['demand_velocity_pct'] = (
        (df_with_features[count_col] - df_with_features['prev_month_count']) / 
        (df_with_features['prev_month_count'] + 1e-6)  # Avoid division by zero
    ) * 100
    
    # First month for each district will have NaN (no previous month)
    # Fill with 0 (no change can be computed)
    df_with_features['demand_velocity_pct'] = df_with_features['demand_velocity_pct'].fillna(0)
    
    # Cap extreme values for interpretability
    # Velocities beyond ±500% are likely data quality issues
    df_with_features['demand_velocity_pct'] = df_with_features['demand_velocity_pct'].clip(-500, 500)
    
    # Clean up temporary column
    df_with_features = df_with_features.drop('prev_month_count', axis=1)
    
    logger.info(f"✓ Added demand velocity using {count_col}")
    
    return df_with_features


def compute_compliance_debt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative compliance debt over time.
    
    Business Logic:
    Compliance debt represents unmet obligations accumulated over time.
    - Debt accumulates when updates don't keep pace with eligibility
    - High debt = backlog that needs attention
    - Negative debt = system ahead of requirements (good performance)
    
    Formula:
    Monthly_Debt = Eligible_Population - Updates_Completed
    Cumulative_Debt = Sum of Monthly_Debt over time
    
    Example:
    - Month 1: 1000 eligible, 800 updated → Debt = 200
    - Month 2: 1000 eligible, 900 updated → Debt = 100
    - Cumulative Debt = 200 + 100 = 300 (total backlog)
    
    Why it matters:
    - Quantifies service delivery gaps
    - Prioritizes districts needing intervention
    - Tracks improvement over time
    
    Args:
        df: Aggregated district-month DataFrame
        
    Returns:
        pd.DataFrame: Input DataFrame with compliance debt columns added
    """
    logger.info("Computing cumulative compliance debt")
    
    df_with_features = df.copy()
    
    # Identify district and date columns
    district_col = None
    date_col = None
    
    for col in df.columns:
        if 'district' in col.lower() and district_col is None:
            district_col = col
        if any(kw in col.lower() for kw in ['month', 'date']) and date_col is None:
            date_col = col
    
    if district_col is None or date_col is None:
        logger.warning("Cannot compute compliance debt - missing district or date column")
        df_with_features['compliance_debt_cumulative'] = 0
        return df_with_features
    
    # Ensure date column is sorted
    df_with_features = df_with_features.sort_values([district_col, date_col])
    
    # Compute monthly debt: eligible - completed
    # Use eligibility projections if available
    if 'eligibility_proj_0_to_5' in df_with_features.columns:
        eligible = df_with_features['eligibility_proj_0_to_5'] + df_with_features.get('eligibility_proj_5_to_18', 0)
    else:
        # Fallback: use count columns
        count_cols = [col for col in df.columns if 'count' in col.lower()]
        if count_cols:
            eligible = df_with_features[count_cols[0]]
        else:
            logger.warning("No eligibility or count columns for debt computation")
            df_with_features['compliance_debt_cumulative'] = 0
            return df_with_features
    
    # Find completion columns
    bio_cols = [col for col in df.columns if 'biometric' in col.lower() and 'count' in col.lower()]
    demo_cols = [col for col in df.columns if 'demographic' in col.lower() and 'count' in col.lower()]
    
    if bio_cols or demo_cols:
        completed = 0
        if bio_cols:
            completed += df_with_features[bio_cols[0]]
        if demo_cols:
            completed += df_with_features[demo_cols[0]]
    else:
        # Fallback: assume 80% completion rate
        completed = eligible * 0.8
    
    # Calculate monthly debt
    # Positive = backlog, Negative = ahead of schedule
    df_with_features['compliance_debt_monthly'] = eligible - completed
    
    # Calculate cumulative debt over time for each district
    # cumsum() adds up all previous months' debt
    df_with_features['compliance_debt_cumulative'] = df_with_features.groupby(district_col)['compliance_debt_monthly'].cumsum()
    
    logger.info("✓ Added compliance debt (monthly and cumulative)")
    
    return df_with_features


def compute_stability_score(df: pd.DataFrame, window_months: int = 3) -> pd.DataFrame:
    """
    Calculate stability score based on variance in demand.
    
    Business Logic:
    Stability measures how predictable/consistent demand is over time.
    - High stability (>0.8) = predictable demand, easy to plan
    - Low stability (<0.5) = volatile demand, needs flexible capacity
    
    Formula:
    Stability = 1 / (1 + Coefficient_of_Variation)
    Where CV = (Standard_Deviation / Mean) over rolling window
    
    Interpretation:
    - CV = 0 (no variation) → Stability = 1.0 (perfectly stable)
    - CV = 1 (high variation) → Stability = 0.5 (moderate stability)
    - CV = 3 (very high variation) → Stability = 0.25 (unstable)
    
    Why it matters:
    - Helps classify districts as stable vs volatile
    - Stable districts need less buffer capacity
    - Volatile districts need surge capacity planning
    
    Args:
        df: Aggregated district-month DataFrame
        window_months: Rolling window size for stability calculation (default: 3)
        
    Returns:
        pd.DataFrame: Input DataFrame with stability score column added
    """
    logger.info(f"Computing stability score with {window_months}-month window")
    
    df_with_features = df.copy()
    
    # Identify district and date columns
    district_col = None
    date_col = None
    
    for col in df.columns:
        if 'district' in col.lower() and district_col is None:
            district_col = col
        if any(kw in col.lower() for kw in ['month', 'date']) and date_col is None:
            date_col = col
    
    if district_col is None or date_col is None:
        logger.warning("Cannot compute stability score - missing district or date column")
        df_with_features['stability_score'] = 0.5  # Neutral score
        return df_with_features
    
    # Ensure date column is sorted
    df_with_features = df_with_features.sort_values([district_col, date_col])
    
    # Find count column to measure stability
    count_cols = [col for col in df.columns if 'count' in col.lower()]
    if not count_cols:
        logger.warning("No count columns found for stability computation")
        df_with_features['stability_score'] = 0.5
        return df_with_features
    
    count_col = count_cols[0]
    
    # Calculate rolling statistics for each district
    # Rolling window captures recent trend, not entire history
    grouped = df_with_features.groupby(district_col)[count_col]
    
    # Rolling mean over window
    rolling_mean = grouped.rolling(window=window_months, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Rolling standard deviation over window
    rolling_std = grouped.rolling(window=window_months, min_periods=1).std().reset_index(level=0, drop=True)
    
    # Calculate Coefficient of Variation (CV)
    # CV = std / mean (normalized measure of variability)
    cv = rolling_std / (rolling_mean + 1e-6)  # Avoid division by zero
    
    # Convert CV to stability score
    # Formula: 1 / (1 + CV)
    # This creates a score between 0 and 1, where higher = more stable
    df_with_features['stability_score'] = 1 / (1 + cv)
    
    # Handle NaN values (first few months with insufficient history)
    df_with_features['stability_score'] = df_with_features['stability_score'].fillna(0.5)
    
    # Ensure score is bounded [0, 1]
    df_with_features['stability_score'] = df_with_features['stability_score'].clip(0, 1)
    
    logger.info(f"✓ Added stability score using {count_col} over {window_months} months")
    
    return df_with_features


def compute_capacity_proxy(df: pd.DataFrame, percentile: int = 90) -> pd.DataFrame:
    """
    Estimate capacity proxy from historical throughput.
    
    Business Logic:
    Capacity proxy estimates what a district can handle based on past performance.
    - Uses high percentile (90th) to represent peak capacity, not average
    - This is the "maximum proven throughput" for each district
    
    Formula:
    Capacity = 90th percentile of historical monthly throughput
    
    Example:
    - District processed [1000, 1200, 1500, 1100, 1300] over 5 months
    - 90th percentile = 1450 (can handle up to this level)
    - Average = 1220 (but average doesn't show capacity limits)
    
    Why it matters:
    - Identifies districts operating at/near capacity (need expansion)
    - Helps benchmark performance across districts
    - Informs resource allocation decisions
    
    Args:
        df: Aggregated district-month DataFrame
        percentile: Percentile to use for capacity estimation (default: 90)
        
    Returns:
        pd.DataFrame: Input DataFrame with capacity proxy column added
    """
    logger.info(f"Computing capacity proxy using {percentile}th percentile of throughput")
    
    df_with_features = df.copy()
    
    # Identify district column
    district_col = None
    for col in df.columns:
        if 'district' in col.lower():
            district_col = col
            break
    
    if district_col is None:
        logger.warning("Cannot compute capacity proxy - missing district column")
        df_with_features['capacity_proxy'] = 0
        return df_with_features
    
    # Find throughput column (total count of transactions)
    count_cols = [col for col in df.columns if 'count' in col.lower()]
    
    if not count_cols:
        logger.warning("No count columns found for capacity computation")
        df_with_features['capacity_proxy'] = 0
        return df_with_features
    
    # Use first count column as throughput measure
    count_col = count_cols[0]
    
    # Calculate percentile capacity for each district
    # This represents the maximum proven throughput
    capacity_by_district = df_with_features.groupby(district_col)[count_col].transform(
        lambda x: x.quantile(percentile / 100.0)
    )
    
    df_with_features['capacity_proxy'] = capacity_by_district
    
    # Calculate capacity utilization (current / capacity)
    # This shows how close to maximum capacity each district is operating
    df_with_features['capacity_utilization_pct'] = (
        df_with_features[count_col] / (df_with_features['capacity_proxy'] + 1e-6)
    ) * 100
    
    # Cap at 100% for interpretability (though >100% is possible during surge)
    df_with_features['capacity_utilization_pct'] = df_with_features['capacity_utilization_pct'].clip(upper=150)
    
    logger.info(f"✓ Added capacity proxy and utilization using {percentile}th percentile")
    
    return df_with_features


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all explainable features for district-month analytics.
    
    This is the main entry point that orchestrates all feature computations.
    All features are deterministic, explainable, and based on domain knowledge.
    
    Features computed:
    1. Eligibility projections (0→5, 5→18 cohorts)
    2. Completion ratios (biometric, demographic, overall)
    3. Demand velocity (month-over-month growth rate)
    4. Compliance debt (cumulative backlog)
    5. Stability score (variance-based predictability)
    6. Capacity proxy (historical throughput percentile)
    
    Args:
        df: Aggregated district-month DataFrame
        
    Returns:
        pd.DataFrame: Input DataFrame with all feature columns added
        
    Example:
        >>> aggregated_df = aggregate_chunks(chunks)
        >>> features_df = compute_features(aggregated_df)
        >>> print(features_df.columns)
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to compute_features")
        return df
    
    logger.info(f"Computing features for {len(df)} district-month records")
    initial_cols = len(df.columns)
    
    # Apply all feature computations in sequence
    # Each function is pure and returns a new DataFrame
    
    # Step 1: Eligibility projections
    df_features = compute_eligibility_projections(df)
    
    # Step 2: Completion ratios
    df_features = compute_completion_ratios(df_features)
    
    # Step 3: Demand velocity
    df_features = compute_demand_velocity(df_features)
    
    # Step 4: Compliance debt
    df_features = compute_compliance_debt(df_features)
    
    # Step 5: Stability score
    df_features = compute_stability_score(df_features, window_months=3)
    
    # Step 6: Capacity proxy
    df_features = compute_capacity_proxy(df_features, percentile=90)
    
    # Summary
    final_cols = len(df_features.columns)
    features_added = final_cols - initial_cols
    
    logger.info(
        f"✓ Feature engineering complete: {features_added} feature(s) added "
        f"({initial_cols} → {final_cols} columns)"
    )
    
    # Log feature summary
    new_features = [col for col in df_features.columns if col not in df.columns]
    if new_features:
        logger.info(f"New features: {new_features}")
    
    return df_features


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Feature engineering module - Example usage")
    
    # Example: Create sample aggregated data
    logger.info("\n=== Example: Feature Engineering ===")
    
    sample_aggregated = pd.DataFrame({
        'district': ['BANGALORE', 'BANGALORE', 'BANGALORE', 'MYSORE', 'MYSORE', 'MYSORE'],
        'month': ['2023-01', '2023-02', '2023-03', '2023-01', '2023-02', '2023-03'],
        'enrolment_count': [1000, 1200, 1100, 800, 850, 900],
        'biometric_count': [800, 950, 900, 600, 650, 700],
        'demographic_count': [750, 900, 850, 550, 600, 650],
        'age_group_child': [300, 350, 320, 240, 260, 280],
        'age_group_adult': [600, 720, 660, 480, 510, 540],
        'age_group_senior': [100, 130, 120, 80, 80, 80]
    })
    
    logger.info("Input aggregated data:")
    logger.info(f"\n{sample_aggregated}")
    
    # Compute all features
    features_df = compute_features(sample_aggregated)
    
    logger.info("\nOutput with features:")
    logger.info(f"\n{features_df}")
    
    # Show specific feature examples
    logger.info("\n=== Feature Interpretations ===")
    
    for idx, row in features_df.iterrows():
        logger.info(f"\n{row['district']} - {row['month']}:")
        
        if 'eligibility_proj_0_to_5' in features_df.columns:
            logger.info(f"  Eligibility (0-5): {row.get('eligibility_proj_0_to_5', 0):.1f} per year")
        
        if 'completion_ratio_overall' in features_df.columns:
            logger.info(f"  Completion ratio: {row.get('completion_ratio_overall', 0):.2%}")
        
        if 'demand_velocity_pct' in features_df.columns:
            logger.info(f"  Demand velocity: {row.get('demand_velocity_pct', 0):+.1f}%")
        
        if 'stability_score' in features_df.columns:
            logger.info(f"  Stability score: {row.get('stability_score', 0):.2f}/1.0")
        
        if 'capacity_utilization_pct' in features_df.columns:
            logger.info(f"  Capacity utilization: {row.get('capacity_utilization_pct', 0):.1f}%")
