"""
Rules engine for automated intervention recommendations.

This module implements a rule-based system that maps risk patterns and operational
metrics to specific actionable interventions. Each recommendation includes expected
impact estimates and prioritization.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


# Intervention playbook definitions
# Each intervention has: ID, name, description, conditions, impact estimates
INTERVENTION_PLAYBOOK = {
    'INT_001': {
        'name': 'Emergency Capacity Expansion',
        'description': 'Deploy mobile enrollment units and extend operating hours',
        'conditions': {
            'capacity_utilization_pct': ('>', 100),
            'asrs_risk_category': ('in', ['high', 'critical'])
        },
        'impact': {
            'capacity_increase_pct': 30,
            'implementation_days': 14,
            'cost_level': 'high',
            'effectiveness_probability': 0.85
        },
        'priority': 1  # Critical
    },
    
    'INT_002': {
        'name': 'Staff Augmentation',
        'description': 'Deploy additional staff from neighboring districts',
        'conditions': {
            'capacity_utilization_pct': ('>', 90),
            'compliance_debt_cumulative': ('>', 500)
        },
        'impact': {
            'capacity_increase_pct': 20,
            'implementation_days': 7,
            'cost_level': 'medium',
            'effectiveness_probability': 0.75
        },
        'priority': 2  # High
    },
    
    'INT_003': {
        'name': 'Backlog Clearance Campaign',
        'description': 'Special drive to clear accumulated compliance debt',
        'conditions': {
            'compliance_debt_cumulative': ('>', 1000),
            'completion_ratio_overall': ('<', 0.6)
        },
        'impact': {
            'debt_reduction_pct': 40,
            'implementation_days': 30,
            'cost_level': 'medium',
            'effectiveness_probability': 0.80
        },
        'priority': 2  # High
    },
    
    'INT_004': {
        'name': 'Public Awareness Campaign',
        'description': 'Increase awareness through local media and community outreach',
        'conditions': {
            'demand_velocity_pct': ('<', -10),
            'completion_ratio_overall': ('<', 0.5)
        },
        'impact': {
            'demand_increase_pct': 15,
            'completion_increase_pct': 10,
            'implementation_days': 21,
            'cost_level': 'low',
            'effectiveness_probability': 0.65
        },
        'priority': 3  # Medium
    },
    
    'INT_005': {
        'name': 'Process Optimization',
        'description': 'Streamline workflows and reduce processing time',
        'conditions': {
            'stability_score': ('<', 0.5),
            'capacity_utilization_pct': ('>', 80)
        },
        'impact': {
            'efficiency_increase_pct': 15,
            'stability_improvement': 0.2,
            'implementation_days': 45,
            'cost_level': 'low',
            'effectiveness_probability': 0.70
        },
        'priority': 3  # Medium
    },
    
    'INT_006': {
        'name': 'Technology Upgrade',
        'description': 'Upgrade enrollment systems and biometric devices',
        'conditions': {
            'stability_score': ('<', 0.4),
            'asrs_risk_category': ('in', ['medium', 'high'])
        },
        'impact': {
            'efficiency_increase_pct': 25,
            'stability_improvement': 0.3,
            'implementation_days': 60,
            'cost_level': 'high',
            'effectiveness_probability': 0.80
        },
        'priority': 3  # Medium
    },
    
    'INT_007': {
        'name': 'Demand Forecasting Review',
        'description': 'Update demand models and adjust resource allocation',
        'conditions': {
            'stability_score': ('<', 0.6),
            'demand_velocity_pct': ('>', 20)
        },
        'impact': {
            'planning_improvement_pct': 30,
            'stability_improvement': 0.15,
            'implementation_days': 14,
            'cost_level': 'low',
            'effectiveness_probability': 0.75
        },
        'priority': 4  # Low
    },
    
    'INT_008': {
        'name': 'Capacity Right-Sizing',
        'description': 'Reduce excess capacity to optimize resource utilization',
        'conditions': {
            'capacity_utilization_pct': ('<', 50),
            'demand_velocity_pct': ('<', 0)
        },
        'impact': {
            'cost_reduction_pct': 20,
            'efficiency_increase_pct': 10,
            'implementation_days': 30,
            'cost_level': 'low',
            'effectiveness_probability': 0.85
        },
        'priority': 4  # Low
    },
    
    'INT_009': {
        'name': 'Quality Assurance Audit',
        'description': 'Comprehensive audit to identify and fix data quality issues',
        'conditions': {
            'anomaly_flag': ('==', True),
            'anomaly_severity': ('>', 2)
        },
        'impact': {
            'quality_improvement_pct': 35,
            'anomaly_reduction_pct': 60,
            'implementation_days': 21,
            'cost_level': 'medium',
            'effectiveness_probability': 0.90
        },
        'priority': 1  # Critical (data quality is foundational)
    },
    
    'INT_010': {
        'name': 'Peer Benchmarking Workshop',
        'description': 'Learn from high-performing peer districts',
        'conditions': {
            'nasri_category': ('in', ['poor', 'critical']),
            'nasri_score': ('<', 40)
        },
        'impact': {
            'performance_increase_pct': 15,
            'best_practices_adopted': 5,
            'implementation_days': 10,
            'cost_level': 'low',
            'effectiveness_probability': 0.70
        },
        'priority': 4  # Low
    }
}


def evaluate_condition(value: Any, operator: str, threshold: Any) -> bool:
    """
    Evaluate a single rule condition.
    
    Args:
        value: Actual value from data
        operator: Comparison operator ('>', '<', '==', '>=', '<=', 'in', 'not in')
        threshold: Threshold value to compare against
        
    Returns:
        bool: True if condition is met, False otherwise
    """
    # Handle missing values
    if pd.isna(value):
        return False
    
    # Evaluate based on operator
    if operator == '>':
        return value > threshold
    elif operator == '<':
        return value < threshold
    elif operator == '==':
        return value == threshold
    elif operator == '>=':
        return value >= threshold
    elif operator == '<=':
        return value <= threshold
    elif operator == 'in':
        return value in threshold
    elif operator == 'not in':
        return value not in threshold
    else:
        logger.warning(f"Unknown operator: {operator}")
        return False


def check_intervention_conditions(
    row: pd.Series,
    conditions: Dict[str, tuple]
) -> bool:
    """
    Check if all conditions for an intervention are met.
    
    Args:
        row: Data row containing district-month metrics
        conditions: Dictionary of field -> (operator, threshold) conditions
        
    Returns:
        bool: True if all conditions are met (AND logic)
    """
    for field, (operator, threshold) in conditions.items():
        # Check if field exists in row
        if field not in row.index:
            logger.debug(f"Field {field} not found in data, skipping condition")
            return False
        
        # Evaluate condition
        if not evaluate_condition(row[field], operator, threshold):
            return False
    
    # All conditions met
    return True


def estimate_impact(
    row: pd.Series,
    intervention: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Estimate the expected impact of an intervention on key metrics.
    
    This function projects how the intervention would improve metrics
    based on baseline values and intervention impact parameters.
    
    Args:
        row: Current district-month metrics
        intervention: Intervention definition with impact parameters
        
    Returns:
        dict: Estimated impact including before/after projections
    """
    impact_config = intervention['impact']
    
    estimated_impact = {
        'implementation_days': impact_config['implementation_days'],
        'cost_level': impact_config['cost_level'],
        'effectiveness_probability': impact_config['effectiveness_probability'],
        'projected_improvements': {}
    }
    
    # Project NASRI score improvement
    current_nasri = row.get('nasri_score', 50)
    
    # Capacity improvements
    if 'capacity_increase_pct' in impact_config:
        capacity_impact = impact_config['capacity_increase_pct']
        estimated_impact['projected_improvements']['capacity_increase_pct'] = capacity_impact
        
        # NASRI improvement from capacity (rough estimate)
        # Capacity contributes 20% to NASRI, so improvement scales accordingly
        nasri_improvement = (capacity_impact / 100) * 20 * 0.5  # Partial effect
        current_nasri += nasri_improvement
    
    # Efficiency improvements
    if 'efficiency_increase_pct' in impact_config:
        efficiency_impact = impact_config['efficiency_increase_pct']
        estimated_impact['projected_improvements']['efficiency_increase_pct'] = efficiency_impact
        
        # NASRI improvement from efficiency
        nasri_improvement = (efficiency_impact / 100) * 15
        current_nasri += nasri_improvement
    
    # Debt reduction
    if 'debt_reduction_pct' in impact_config:
        debt_reduction = impact_config['debt_reduction_pct']
        current_debt = row.get('compliance_debt_cumulative', 0)
        estimated_new_debt = current_debt * (1 - debt_reduction / 100)
        
        estimated_impact['projected_improvements']['debt_reduction_absolute'] = current_debt - estimated_new_debt
        estimated_impact['projected_improvements']['debt_reduction_pct'] = debt_reduction
        
        # NASRI improvement from debt reduction
        # Debt contributes 20% to NASRI
        if current_debt > 0:
            nasri_improvement = (debt_reduction / 100) * 20 * 0.7
            current_nasri += nasri_improvement
    
    # Stability improvements
    if 'stability_improvement' in impact_config:
        stability_gain = impact_config['stability_improvement']
        current_stability = row.get('stability_score', 0.5)
        estimated_new_stability = min(current_stability + stability_gain, 1.0)
        
        estimated_impact['projected_improvements']['stability_score_gain'] = stability_gain
        estimated_impact['projected_improvements']['stability_score_new'] = estimated_new_stability
        
        # NASRI improvement from stability
        nasri_improvement = stability_gain * 20  # Stability is 20% of NASRI
        current_nasri += nasri_improvement
    
    # Completion ratio improvements
    if 'completion_increase_pct' in impact_config:
        completion_increase = impact_config['completion_increase_pct']
        current_completion = row.get('completion_ratio_overall', 0.5)
        estimated_new_completion = min(current_completion * (1 + completion_increase / 100), 1.0)
        
        estimated_impact['projected_improvements']['completion_ratio_new'] = estimated_new_completion
        
        # NASRI improvement from completion
        nasri_improvement = (completion_increase / 100) * 25
        current_nasri += nasri_improvement
    
    # Project final NASRI score (capped at 100)
    estimated_impact['projected_nasri_score'] = min(current_nasri, 100)
    estimated_impact['nasri_improvement'] = estimated_impact['projected_nasri_score'] - row.get('nasri_score', 50)
    
    # Project ASRS (risk) reduction
    current_asrs = row.get('asrs_score', 0.5)
    
    # Risk reduction from interventions (inverse of improvements)
    risk_reduction = 0
    
    if 'capacity_increase_pct' in impact_config:
        risk_reduction += (impact_config['capacity_increase_pct'] / 100) * 0.3  # Capacity stress is 30% of ASRS
    
    if 'stability_improvement' in impact_config:
        risk_reduction += impact_config['stability_improvement'] * 0.25  # Instability is 25% of ASRS
    
    if 'debt_reduction_pct' in impact_config:
        risk_reduction += (impact_config['debt_reduction_pct'] / 100) * 0.25  # Compliance gap is 25% of ASRS
    
    estimated_new_asrs = max(current_asrs - risk_reduction, 0)
    estimated_impact['projected_asrs_score'] = estimated_new_asrs
    estimated_impact['asrs_reduction'] = current_asrs - estimated_new_asrs
    
    return estimated_impact


def calculate_priority_score(
    intervention_priority: int,
    impact_estimate: Dict[str, Any],
    current_metrics: pd.Series
) -> float:
    """
    Calculate a numerical priority score for ranking interventions.
    
    Factors considered:
    - Base priority (from playbook)
    - Expected NASRI improvement
    - Expected ASRS reduction
    - Cost-effectiveness (impact / cost)
    - Implementation speed
    - Effectiveness probability
    
    Higher score = higher priority
    
    Args:
        intervention_priority: Base priority level (1-4, lower is higher priority)
        impact_estimate: Estimated impact dictionary
        current_metrics: Current district metrics
        
    Returns:
        float: Priority score for ranking (higher = more urgent)
    """
    # Base priority score (invert so 1=highest gets highest score)
    base_score = (5 - intervention_priority) * 20  # Max 80 points
    
    # Impact score (max 40 points)
    nasri_improvement = impact_estimate.get('nasri_improvement', 0)
    asrs_reduction = impact_estimate.get('asrs_reduction', 0)
    
    impact_score = (nasri_improvement * 0.2) + (asrs_reduction * 100 * 0.2)  # Max ~40 points
    
    # Cost-effectiveness score (max 20 points)
    cost_map = {'low': 1.5, 'medium': 1.0, 'high': 0.6}
    cost_factor = cost_map.get(impact_estimate['cost_level'], 1.0)
    cost_effectiveness_score = impact_score * cost_factor  # Max ~30 points
    
    # Implementation speed bonus (max 10 points)
    impl_days = impact_estimate['implementation_days']
    speed_score = max(10 - (impl_days / 10), 0)  # Faster is better
    
    # Effectiveness probability multiplier
    probability = impact_estimate['effectiveness_probability']
    
    # Combine scores
    total_score = (base_score + impact_score + cost_effectiveness_score + speed_score) * probability
    
    return total_score


def recommend_actions(
    row: pd.Series,
    max_recommendations: int = 5,
    min_priority_score: float = 20.0
) -> List[Dict[str, Any]]:
    """
    Recommend interventions for a district based on current metrics.
    
    Process:
    1. Evaluate all intervention conditions against current metrics
    2. For matching interventions, estimate expected impact
    3. Calculate priority scores for ranking
    4. Return top N recommendations with impact estimates
    
    Recommendation Logic:
    - Rule-based matching (conditions must be met)
    - Impact-driven prioritization (expected improvement)
    - Cost-effectiveness consideration
    - Multiple interventions may be recommended (non-exclusive)
    
    Args:
        row: Series containing district-month metrics
        max_recommendations: Maximum number of interventions to recommend
        min_priority_score: Minimum priority score to include recommendation
        
    Returns:
        list: List of recommended interventions with details
        
    Example:
        >>> metrics = df.iloc[0]  # Get one district-month row
        >>> recommendations = recommend_actions(metrics)
        >>> for rec in recommendations:
        ...     print(f"{rec['name']}: {rec['impact_estimate']['nasri_improvement']:.1f} NASRI points")
    """
    logger.debug(f"Generating recommendations for district: {row.get('district', 'Unknown')}")
    
    recommendations = []
    
    # Evaluate each intervention in playbook
    for intervention_id, intervention in INTERVENTION_PLAYBOOK.items():
        # Check if conditions are met
        conditions_met = check_intervention_conditions(row, intervention['conditions'])
        
        if conditions_met:
            logger.debug(f"Intervention {intervention_id} conditions met")
            
            # Estimate impact
            impact_estimate = estimate_impact(row, intervention)
            
            # Calculate priority score
            priority_score = calculate_priority_score(
                intervention['priority'],
                impact_estimate,
                row
            )
            
            # Create recommendation object
            recommendation = {
                'intervention_id': intervention_id,
                'name': intervention['name'],
                'description': intervention['description'],
                'priority': intervention['priority'],
                'priority_score': priority_score,
                'impact_estimate': impact_estimate,
                'conditions_met': intervention['conditions']
            }
            
            recommendations.append(recommendation)
    
    # Filter by minimum priority score
    recommendations = [r for r in recommendations if r['priority_score'] >= min_priority_score]
    
    # Sort by priority score (descending)
    recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Limit to max recommendations
    recommendations = recommendations[:max_recommendations]
    
    # Log summary
    if recommendations:
        logger.info(
            f"Generated {len(recommendations)} recommendation(s) for "
            f"district {row.get('district', 'Unknown')}"
        )
        for rec in recommendations:
            logger.debug(
                f"  {rec['intervention_id']}: {rec['name']} "
                f"(priority score: {rec['priority_score']:.1f})"
            )
    else:
        logger.info(
            f"No interventions recommended for district {row.get('district', 'Unknown')} "
            f"(metrics within acceptable ranges)"
        )
    
    return recommendations


def generate_action_report(
    df: pd.DataFrame,
    max_recommendations_per_district: int = 3
) -> pd.DataFrame:
    """
    Generate action recommendations for all districts in DataFrame.
    
    Args:
        df: DataFrame with district-month metrics
        max_recommendations_per_district: Max interventions per district
        
    Returns:
        pd.DataFrame: Recommendations with district, intervention, and impact details
    """
    logger.info(f"Generating action report for {len(df)} records")
    
    all_recommendations = []
    
    for idx, row in df.iterrows():
        recommendations = recommend_actions(row, max_recommendations=max_recommendations_per_district)
        
        for rec in recommendations:
            # Flatten recommendation for DataFrame
            rec_flat = {
                'district': row.get('district', 'Unknown'),
                'month': row.get('month', 'Unknown'),
                'current_nasri_score': row.get('nasri_score', np.nan),
                'current_asrs_score': row.get('asrs_score', np.nan),
                'intervention_id': rec['intervention_id'],
                'intervention_name': rec['name'],
                'intervention_description': rec['description'],
                'priority': rec['priority'],
                'priority_score': rec['priority_score'],
                'implementation_days': rec['impact_estimate']['implementation_days'],
                'cost_level': rec['impact_estimate']['cost_level'],
                'effectiveness_probability': rec['impact_estimate']['effectiveness_probability'],
                'projected_nasri_score': rec['impact_estimate']['projected_nasri_score'],
                'projected_asrs_score': rec['impact_estimate']['projected_asrs_score'],
                'nasri_improvement': rec['impact_estimate']['nasri_improvement'],
                'asrs_reduction': rec['impact_estimate']['asrs_reduction']
            }
            
            all_recommendations.append(rec_flat)
    
    if all_recommendations:
        recommendations_df = pd.DataFrame(all_recommendations)
        logger.info(f"Generated {len(recommendations_df)} total recommendations")
        return recommendations_df
    else:
        logger.warning("No recommendations generated for any district")
        return pd.DataFrame()


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Rules engine module - Example usage")
    
    # Example: Create sample district metrics
    logger.info("\n=== Example: Intervention Recommendations ===")
    
    sample_metrics = pd.DataFrame({
        'district': ['HIGH_RISK', 'UNDERPERFORMING', 'DECLINING', 'HEALTHY'],
        'month': ['2024-01'] * 4,
        'capacity_utilization_pct': [110, 75, 45, 80],
        'completion_ratio_overall': [0.55, 0.45, 0.70, 0.85],
        'stability_score': [0.45, 0.35, 0.65, 0.80],
        'demand_velocity_pct': [5, 8, -15, 7],
        'compliance_debt_cumulative': [800, 1500, 200, -50],
        'nasri_score': [35, 30, 50, 75],
        'nasri_category': ['poor', 'critical', 'fair', 'good'],
        'asrs_score': [0.75, 0.60, 0.35, 0.20],
        'asrs_risk_category': ['critical', 'high', 'medium', 'low'],
        'anomaly_flag': [False, True, False, False],
        'anomaly_severity': [0, 3, 0, 0]
    })
    
    logger.info("Sample district metrics:")
    logger.info(f"\n{sample_metrics[['district', 'nasri_score', 'asrs_score', 'nasri_category', 'asrs_risk_category']]}")
    
    # Generate recommendations for each district
    logger.info("\n=== Intervention Recommendations by District ===")
    
    for idx, row in sample_metrics.iterrows():
        logger.info(f"\n--- District: {row['district']} ---")
        logger.info(f"Current State: NASRI={row['nasri_score']:.1f}, ASRS={row['asrs_score']:.2f}")
        
        recommendations = recommend_actions(row, max_recommendations=3)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"\nRecommendation {i}:")
                logger.info(f"  Action: {rec['name']}")
                logger.info(f"  Description: {rec['description']}")
                logger.info(f"  Priority: {rec['priority']} (score: {rec['priority_score']:.1f})")
                logger.info(f"  Implementation: {rec['impact_estimate']['implementation_days']} days")
                logger.info(f"  Cost: {rec['impact_estimate']['cost_level']}")
                logger.info(f"  Effectiveness: {rec['impact_estimate']['effectiveness_probability']:.0%}")
                logger.info(f"  Expected Impact:")
                logger.info(f"    - NASRI: {row['nasri_score']:.1f} → {rec['impact_estimate']['projected_nasri_score']:.1f} (+{rec['impact_estimate']['nasri_improvement']:.1f})")
                logger.info(f"    - ASRS: {row['asrs_score']:.2f} → {rec['impact_estimate']['projected_asrs_score']:.2f} (-{rec['impact_estimate']['asrs_reduction']:.2f})")
        else:
            logger.info("  No interventions recommended (metrics within acceptable ranges)")
    
    # Generate full action report
    logger.info("\n=== Full Action Report ===")
    action_report = generate_action_report(sample_metrics, max_recommendations_per_district=2)
    
    if not action_report.empty:
        logger.info(f"\n{action_report[['district', 'intervention_name', 'priority_score', 'nasri_improvement', 'cost_level']]}")
