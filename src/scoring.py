"""
Scoring module for NASRI and ASRS computation.

This module implements two key scoring systems:
1. NASRI (National Aadhaar Service Readiness Index) - 0 to 100 scale
2. ASRS (Aadhaar Service Risk Score) - 0 to 1 probability scale

All scores are explainable with transparent formulas and configurable weights.
"""

import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


# Default configuration weights
# These should be loaded from config.py in production
DEFAULT_NASRI_WEIGHTS = {
    'completion_ratio': 0.25,      # 25% - Service delivery quality
    'capacity_utilization': 0.20,  # 20% - Resource efficiency
    'stability_score': 0.20,       # 20% - Operational predictability
    'demand_velocity': 0.15,       # 15% - Growth management
    'compliance_debt': 0.20        # 20% - Backlog management
}

DEFAULT_ASRS_WEIGHTS = {
    'capacity_stress': 0.30,       # 30% - Over-utilization risk
    'instability': 0.25,           # 25% - Volatility risk
    'compliance_gap': 0.25,        # 25% - Backlog accumulation risk
    'negative_velocity': 0.20      # 20% - Declining demand risk
}

# Risk category thresholds
NASRI_THRESHOLDS = {
    'excellent': 80,    # >= 80: Excellent performance
    'good': 60,         # >= 60: Good performance
    'fair': 40,         # >= 40: Fair performance
    'poor': 20          # >= 20: Poor performance
                        # < 20: Critical
}

ASRS_THRESHOLDS = {
    'critical': 0.75,   # >= 0.75: Critical risk
    'high': 0.50,       # >= 0.50: High risk
    'medium': 0.30,     # >= 0.30: Medium risk
    'low': 0.15         # >= 0.15: Low risk
                        # < 0.15: Minimal risk
}


def normalize_to_0_1(
    series: pd.Series,
    invert: bool = False,
    clip_lower: float = None,
    clip_upper: float = None
) -> pd.Series:
    """
    Normalize a series to 0-1 range using min-max normalization.
    
    Formula:
    normalized = (value - min) / (max - min)
    
    If invert=True:
    normalized = 1 - (value - min) / (max - min)
    
    Use invert=True for metrics where lower is better (e.g., debt, risk).
    
    Args:
        series: Series to normalize
        invert: If True, invert scale (1=worst, 0=best)
        clip_lower: Minimum value to clip before normalization
        clip_upper: Maximum value to clip before normalization
        
    Returns:
        pd.Series: Normalized series in [0, 1] range
    """
    # Apply clipping if specified
    if clip_lower is not None or clip_upper is not None:
        series = series.clip(lower=clip_lower, upper=clip_upper)
    
    # Calculate min and max
    min_val = series.min()
    max_val = series.max()
    
    # Handle case where all values are the same
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)  # Neutral score
    
    # Min-max normalization
    normalized = (series - min_val) / (max_val - min_val)
    
    # Invert if specified (for negative metrics)
    if invert:
        normalized = 1 - normalized
    
    # Ensure bounds [0, 1]
    normalized = normalized.clip(0, 1)
    
    return normalized


def compute_nasri(
    df: pd.DataFrame,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Compute NASRI (National Aadhaar Service Readiness Index) score.
    
    NASRI Score Methodology:
    ========================
    
    NASRI is a composite index (0-100) measuring district-level service readiness
    and operational excellence. Higher scores indicate better performance.
    
    Components (5 dimensions):
    
    1. Completion Ratio (25%):
       - Measures service delivery quality
       - Formula: Average of biometric and demographic completion rates
       - Interpretation: % of required updates completed
       - Range: 0-1, normalized to 0-100
    
    2. Capacity Utilization (20%):
       - Measures resource efficiency
       - Formula: Current throughput / Historical maximum capacity
       - Interpretation: How efficiently resources are used
       - Optimal range: 70-90% (not too low, not overloaded)
       - Scoring: Penalize both under-utilization and over-utilization
    
    3. Stability Score (20%):
       - Measures operational predictability
       - Formula: 1 / (1 + Coefficient of Variation)
       - Interpretation: Consistency of demand patterns
       - Higher = more predictable, easier to plan
    
    4. Demand Velocity (15%):
       - Measures growth management capability
       - Formula: Normalized month-over-month growth rate
       - Interpretation: Ability to handle changing demand
       - Moderate velocity (5-10%) is optimal
       - Penalize both stagnation and explosive growth
    
    5. Compliance Debt (20%):
       - Measures backlog management
       - Formula: Inverted cumulative debt (lower debt = higher score)
       - Interpretation: Service delivery gaps
       - Zero or negative debt = excellent (ahead of schedule)
    
    Final NASRI Formula:
    NASRI = Σ(component_i * weight_i) * 100
    
    Score Interpretation:
    - 80-100: Excellent - Best-in-class performance
    - 60-79:  Good - Above average, minor improvements needed
    - 40-59:  Fair - Average, needs attention
    - 20-39:  Poor - Below standard, requires intervention
    - 0-19:   Critical - Immediate action required
    
    Args:
        df: DataFrame with feature-engineered district-month data
        weights: Custom weights dict (default: uses DEFAULT_NASRI_WEIGHTS)
        
    Returns:
        pd.DataFrame: Input DataFrame with NASRI score and components added
        
    Example:
        >>> df_with_nasri = compute_nasri(features_df)
        >>> print(df_with_nasri[['district', 'nasri_score', 'nasri_category']])
    """
    logger.info(f"Computing NASRI scores for {len(df)} records")
    
    if df.empty:
        logger.warning("Empty DataFrame provided to compute_nasri")
        df['nasri_score'] = 0
        df['nasri_category'] = 'unknown'
        return df
    
    # Use default weights if not provided
    if weights is None:
        weights = DEFAULT_NASRI_WEIGHTS
        logger.debug("Using default NASRI weights")
    
    # Validate weights sum to 1.0
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        logger.warning(f"NASRI weights sum to {total_weight}, normalizing to 1.0")
        weights = {k: v / total_weight for k, v in weights.items()}
    
    df_result = df.copy()
    
    # Component 1: Completion Ratio (25%)
    # Higher is better (more updates completed)
    completion_cols = [col for col in df.columns if 'completion_ratio' in col.lower()]
    if completion_cols:
        # Use overall completion ratio if available, else average of available ratios
        if 'completion_ratio_overall' in df.columns:
            completion_component = df['completion_ratio_overall']
        else:
            completion_component = df[completion_cols].mean(axis=1)
        
        # Already in [0, 1] range
        df_result['nasri_completion'] = completion_component
        logger.debug("✓ NASRI component: completion_ratio")
    else:
        logger.warning("No completion ratio columns found, using 0.5 default")
        df_result['nasri_completion'] = 0.5
    
    # Component 2: Capacity Utilization (20%)
    # Optimal range: 70-90% (penalize both under and over utilization)
    if 'capacity_utilization_pct' in df.columns:
        # Convert percentage to 0-1 scale
        capacity_util = df['capacity_utilization_pct'] / 100.0
        
        # Score based on proximity to optimal range (70-90%)
        # Perfect score at 80%, declining as we move away
        optimal_target = 0.80
        capacity_component = 1 - np.abs(capacity_util - optimal_target) / optimal_target
        capacity_component = capacity_component.clip(0, 1)
        
        df_result['nasri_capacity'] = capacity_component
        logger.debug("✓ NASRI component: capacity_utilization")
    else:
        logger.warning("No capacity utilization found, using 0.5 default")
        df_result['nasri_capacity'] = 0.5
    
    # Component 3: Stability Score (20%)
    # Higher is better (more stable operations)
    if 'stability_score' in df.columns:
        # Already in [0, 1] range
        df_result['nasri_stability'] = df['stability_score']
        logger.debug("✓ NASRI component: stability_score")
    else:
        logger.warning("No stability score found, using 0.5 default")
        df_result['nasri_stability'] = 0.5
    
    # Component 4: Demand Velocity (15%)
    # Moderate velocity is optimal (5-10% growth)
    # Penalize both stagnation (<0%) and explosive growth (>30%)
    if 'demand_velocity_pct' in df.columns:
        velocity = df['demand_velocity_pct']
        
        # Score based on proximity to optimal range (5-10%)
        # Map velocity to score:
        # -10% to 0%: 0.3 to 0.6 (stagnation/decline)
        # 0% to 5%: 0.6 to 0.9 (slow growth)
        # 5% to 10%: 0.9 to 1.0 (optimal growth)
        # 10% to 20%: 1.0 to 0.8 (high growth)
        # 20% to 50%: 0.8 to 0.5 (very high growth)
        # >50%: 0.5 to 0.2 (explosive growth, likely data quality issue)
        
        velocity_component = pd.Series(0.5, index=df.index)
        
        # Optimal range: 5-10%
        optimal_mask = (velocity >= 5) & (velocity <= 10)
        velocity_component[optimal_mask] = 0.9 + 0.1 * (10 - velocity[optimal_mask]) / 5
        
        # Moderate growth: 0-5%
        moderate_mask = (velocity >= 0) & (velocity < 5)
        velocity_component[moderate_mask] = 0.6 + 0.3 * (velocity[moderate_mask] / 5)
        
        # High growth: 10-20%
        high_mask = (velocity > 10) & (velocity <= 20)
        velocity_component[high_mask] = 1.0 - 0.2 * ((velocity[high_mask] - 10) / 10)
        
        # Very high growth: 20-50%
        very_high_mask = (velocity > 20) & (velocity <= 50)
        velocity_component[very_high_mask] = 0.8 - 0.3 * ((velocity[very_high_mask] - 20) / 30)
        
        # Explosive growth: >50%
        explosive_mask = velocity > 50
        velocity_component[explosive_mask] = 0.2
        
        # Decline/stagnation: <0%
        decline_mask = velocity < 0
        velocity_component[decline_mask] = 0.6 + 0.3 * (velocity[decline_mask] / 10).clip(-1, 0)
        
        df_result['nasri_velocity'] = velocity_component.clip(0, 1)
        logger.debug("✓ NASRI component: demand_velocity")
    else:
        logger.warning("No demand velocity found, using 0.5 default")
        df_result['nasri_velocity'] = 0.5
    
    # Component 5: Compliance Debt (20%)
    # Lower debt is better (invert for scoring)
    if 'compliance_debt_cumulative' in df.columns:
        debt = df['compliance_debt_cumulative']
        
        # Normalize and invert (lower debt = higher score)
        # Clip extreme values to avoid outlier domination
        debt_percentile_99 = debt.quantile(0.99)
        debt_clipped = debt.clip(upper=debt_percentile_99)
        
        compliance_component = normalize_to_0_1(debt_clipped, invert=True)
        
        df_result['nasri_compliance'] = compliance_component
        logger.debug("✓ NASRI component: compliance_debt")
    else:
        logger.warning("No compliance debt found, using 0.5 default")
        df_result['nasri_compliance'] = 0.5
    
    # Calculate weighted NASRI score
    nasri_score = (
        df_result['nasri_completion'] * weights['completion_ratio'] +
        df_result['nasri_capacity'] * weights['capacity_utilization'] +
        df_result['nasri_stability'] * weights['stability_score'] +
        df_result['nasri_velocity'] * weights['demand_velocity'] +
        df_result['nasri_compliance'] * weights['compliance_debt']
    ) * 100  # Scale to 0-100
    
    df_result['nasri_score'] = nasri_score.clip(0, 100)
    
    # Assign NASRI categories
    df_result['nasri_category'] = pd.cut(
        df_result['nasri_score'],
        bins=[-np.inf, 20, 40, 60, 80, np.inf],
        labels=['critical', 'poor', 'fair', 'good', 'excellent']
    )
    
    # Log summary statistics
    mean_score = df_result['nasri_score'].mean()
    category_dist = df_result['nasri_category'].value_counts().to_dict()
    
    logger.info(
        f"✓ NASRI computed: mean={mean_score:.1f}, "
        f"distribution={category_dist}"
    )
    
    return df_result


def compute_asrs(
    df: pd.DataFrame,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Compute ASRS (Aadhaar Service Risk Score) probability.
    
    ASRS Score Methodology:
    =======================
    
    ASRS is a risk probability (0-1) measuring likelihood of service disruption
    or operational failure. Higher scores indicate higher risk.
    
    Components (4 risk dimensions):
    
    1. Capacity Stress (30%):
       - Risk: Over-utilization leading to service degradation
       - Formula: Sigmoid(utilization - optimal_range)
       - Interpretation: How much capacity is strained
       - Risk increases exponentially above 90% utilization
    
    2. Instability (25%):
       - Risk: Unpredictable demand causing planning failures
       - Formula: 1 - stability_score
       - Interpretation: Operational volatility
       - High instability = hard to plan, likely shortfalls
    
    3. Compliance Gap (25%):
       - Risk: Growing backlog becoming unmanageable
       - Formula: Normalized cumulative debt
       - Interpretation: Service delivery gap accumulation
       - Positive debt = risk, negative = buffer
    
    4. Negative Velocity (20%):
       - Risk: Declining demand indicating system failure
       - Formula: Sigmoid(-velocity) for negative growth
       - Interpretation: Service degradation or data quality issues
       - Sharp declines (>10%) are high risk
    
    Final ASRS Formula:
    ASRS = Σ(risk_component_i * weight_i)
    
    Probability Interpretation:
    - 0.75-1.0:  Critical Risk - Immediate intervention required
    - 0.50-0.74: High Risk - Urgent attention needed
    - 0.30-0.49: Medium Risk - Monitor closely
    - 0.15-0.29: Low Risk - Routine monitoring
    - 0.0-0.14:  Minimal Risk - Healthy operations
    
    Args:
        df: DataFrame with feature-engineered district-month data
        weights: Custom weights dict (default: uses DEFAULT_ASRS_WEIGHTS)
        
    Returns:
        pd.DataFrame: Input DataFrame with ASRS score and risk category added
        
    Example:
        >>> df_with_asrs = compute_asrs(features_df)
        >>> high_risk = df_with_asrs[df_with_asrs['asrs_risk_category'] == 'high']
    """
    logger.info(f"Computing ASRS scores for {len(df)} records")
    
    if df.empty:
        logger.warning("Empty DataFrame provided to compute_asrs")
        df['asrs_score'] = 0
        df['asrs_risk_category'] = 'unknown'
        return df
    
    # Use default weights if not provided
    if weights is None:
        weights = DEFAULT_ASRS_WEIGHTS
        logger.debug("Using default ASRS weights")
    
    # Validate weights sum to 1.0
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        logger.warning(f"ASRS weights sum to {total_weight}, normalizing to 1.0")
        weights = {k: v / total_weight for k, v in weights.items()}
    
    df_result = df.copy()
    
    # Risk Component 1: Capacity Stress (30%)
    # Risk increases as utilization exceeds optimal range (>90%)
    if 'capacity_utilization_pct' in df.columns:
        utilization = df['capacity_utilization_pct'] / 100.0
        
        # Risk calculation:
        # <70%: Low risk (0.1)
        # 70-90%: Minimal risk (0.1-0.3)
        # 90-100%: Moderate risk (0.3-0.6)
        # >100%: High risk (0.6-0.9)
        # >120%: Critical risk (0.9-1.0)
        
        capacity_risk = pd.Series(0.1, index=df.index)
        
        # Optimal range: minimal risk
        optimal_mask = (utilization >= 0.7) & (utilization <= 0.9)
        capacity_risk[optimal_mask] = 0.1 + 0.2 * ((utilization[optimal_mask] - 0.7) / 0.2)
        
        # Approaching capacity: moderate risk
        moderate_mask = (utilization > 0.9) & (utilization <= 1.0)
        capacity_risk[moderate_mask] = 0.3 + 0.3 * ((utilization[moderate_mask] - 0.9) / 0.1)
        
        # Over capacity: high risk
        high_mask = (utilization > 1.0) & (utilization <= 1.2)
        capacity_risk[high_mask] = 0.6 + 0.3 * ((utilization[high_mask] - 1.0) / 0.2)
        
        # Severely over capacity: critical risk
        critical_mask = utilization > 1.2
        capacity_risk[critical_mask] = 0.9 + 0.1 * ((utilization[critical_mask] - 1.2) / 0.3).clip(0, 1)
        
        # Under-utilization: low risk but not zero (waste concern)
        under_mask = utilization < 0.7
        capacity_risk[under_mask] = 0.1
        
        df_result['asrs_capacity_stress'] = capacity_risk.clip(0, 1)
        logger.debug("✓ ASRS component: capacity_stress")
    else:
        logger.warning("No capacity utilization found, using 0.3 default")
        df_result['asrs_capacity_stress'] = 0.3
    
    # Risk Component 2: Instability (25%)
    # Higher instability = higher risk
    if 'stability_score' in df.columns:
        # Invert stability to get instability risk
        instability_risk = 1 - df['stability_score']
        df_result['asrs_instability'] = instability_risk.clip(0, 1)
        logger.debug("✓ ASRS component: instability")
    else:
        logger.warning("No stability score found, using 0.5 default")
        df_result['asrs_instability'] = 0.5
    
    # Risk Component 3: Compliance Gap (25%)
    # Positive debt = risk, higher debt = higher risk
    if 'compliance_debt_cumulative' in df.columns:
        debt = df['compliance_debt_cumulative']
        
        # Normalize debt to risk score
        # Negative debt (ahead) = low risk (0.0-0.1)
        # Zero debt = minimal risk (0.1)
        # Positive debt = increasing risk (0.1-1.0)
        
        # Clip extreme values
        debt_percentile_99 = debt.quantile(0.99)
        debt_clipped = debt.clip(upper=debt_percentile_99)
        
        # Normalize to [0, 1] (higher debt = higher risk)
        compliance_risk = normalize_to_0_1(debt_clipped, invert=False)
        
        # Adjust: negative debt (good) should have minimal risk
        negative_debt_mask = debt < 0
        compliance_risk[negative_debt_mask] = 0.1
        
        df_result['asrs_compliance_gap'] = compliance_risk.clip(0, 1)
        logger.debug("✓ ASRS component: compliance_gap")
    else:
        logger.warning("No compliance debt found, using 0.3 default")
        df_result['asrs_compliance_gap'] = 0.3
    
    # Risk Component 4: Negative Velocity (20%)
    # Declining demand = risk (service degradation or errors)
    if 'demand_velocity_pct' in df.columns:
        velocity = df['demand_velocity_pct']
        
        # Risk calculation:
        # Positive velocity: minimal risk (0.1)
        # 0 to -5%: Low risk (0.1-0.3) - slight decline
        # -5% to -10%: Moderate risk (0.3-0.6) - concerning decline
        # -10% to -20%: High risk (0.6-0.8) - serious decline
        # < -20%: Critical risk (0.8-1.0) - severe decline
        
        velocity_risk = pd.Series(0.1, index=df.index)
        
        # Positive velocity: minimal risk
        positive_mask = velocity >= 0
        velocity_risk[positive_mask] = 0.1
        
        # Slight decline: low risk
        slight_mask = (velocity < 0) & (velocity >= -5)
        velocity_risk[slight_mask] = 0.1 + 0.2 * (abs(velocity[slight_mask]) / 5)
        
        # Concerning decline: moderate risk
        moderate_mask = (velocity < -5) & (velocity >= -10)
        velocity_risk[moderate_mask] = 0.3 + 0.3 * ((abs(velocity[moderate_mask]) - 5) / 5)
        
        # Serious decline: high risk
        high_mask = (velocity < -10) & (velocity >= -20)
        velocity_risk[high_mask] = 0.6 + 0.2 * ((abs(velocity[high_mask]) - 10) / 10)
        
        # Severe decline: critical risk
        critical_mask = velocity < -20
        velocity_risk[critical_mask] = 0.8 + 0.2 * ((abs(velocity[critical_mask]) - 20) / 30).clip(0, 1)
        
        df_result['asrs_negative_velocity'] = velocity_risk.clip(0, 1)
        logger.debug("✓ ASRS component: negative_velocity")
    else:
        logger.warning("No demand velocity found, using 0.2 default")
        df_result['asrs_negative_velocity'] = 0.2
    
    # Calculate weighted ASRS score
    asrs_score = (
        df_result['asrs_capacity_stress'] * weights['capacity_stress'] +
        df_result['asrs_instability'] * weights['instability'] +
        df_result['asrs_compliance_gap'] * weights['compliance_gap'] +
        df_result['asrs_negative_velocity'] * weights['negative_velocity']
    )
    
    df_result['asrs_score'] = asrs_score.clip(0, 1)
    
    # Assign ASRS risk categories
    df_result['asrs_risk_category'] = pd.cut(
        df_result['asrs_score'],
        bins=[-np.inf, 0.15, 0.30, 0.50, 0.75, np.inf],
        labels=['minimal', 'low', 'medium', 'high', 'critical']
    )
    
    # Log summary statistics
    mean_score = df_result['asrs_score'].mean()
    category_dist = df_result['asrs_risk_category'].value_counts().to_dict()
    
    logger.info(
        f"✓ ASRS computed: mean={mean_score:.3f}, "
        f"distribution={category_dist}"
    )
    
    return df_result


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Scoring module - Example usage")
    
    # Example: Create sample feature data
    logger.info("\n=== Example: NASRI and ASRS Scoring ===")
    
    sample_features = pd.DataFrame({
        'district': ['A', 'B', 'C', 'D', 'E'],
        'month': ['2024-01', '2024-01', '2024-01', '2024-01', '2024-01'],
        'completion_ratio_overall': [0.85, 0.60, 0.40, 0.90, 0.75],
        'capacity_utilization_pct': [75, 95, 50, 85, 110],
        'stability_score': [0.85, 0.60, 0.40, 0.90, 0.50],
        'demand_velocity_pct': [8, 25, -5, 7, -15],
        'compliance_debt_cumulative': [100, 500, 1000, -50, 2000]
    })
    
    logger.info("Input feature data:")
    logger.info(f"\n{sample_features}")
    
    # Compute NASRI scores
    df_with_nasri = compute_nasri(sample_features)
    
    logger.info("\n=== NASRI Scores ===")
    logger.info(f"\n{df_with_nasri[['district', 'nasri_score', 'nasri_category']]}")
    
    # Compute ASRS scores
    df_with_asrs = compute_asrs(df_with_nasri)
    
    logger.info("\n=== ASRS Scores ===")
    logger.info(f"\n{df_with_asrs[['district', 'asrs_score', 'asrs_risk_category']]}")
    
    # Combined view
    logger.info("\n=== Combined Scores ===")
    logger.info(f"\n{df_with_asrs[['district', 'nasri_score', 'nasri_category', 'asrs_score', 'asrs_risk_category']]}")
