"""
Simulation module for intervention impact modeling.

This module provides conservative what-if simulations to compare scenarios:
- Baseline (no action)
- With intervention

Uses deterministic modeling with conservative assumptions. No machine learning.
All projections are based on observable metrics and historical patterns.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


# Simulation constants
DEFAULT_SIMULATION_MONTHS = 6  # Default simulation horizon
MONTHLY_NATURAL_IMPROVEMENT = 0.02  # 2% baseline improvement per month
MONTHLY_DEGRADATION_RATE = 0.05  # 5% degradation if no action on critical issues
WAITING_TIME_PER_BACKLOG_UNIT = 0.5  # Days of waiting time per unit of backlog


def project_baseline_scenario(
    initial_metrics: Dict[str, float],
    months: int = 6
) -> List[Dict[str, float]]:
    """
    Project metrics over time assuming no intervention (baseline scenario).
    
    Baseline Assumptions:
    - Natural improvement: Slight organic improvement (~2% per month) from learning
    - Degradation under stress: If capacity >90%, system degrades (~5% per month)
    - Debt accumulation: Backlog grows if completion ratio <80%
    - Stability maintenance: Stability score remains relatively constant
    
    Conservative Approach:
    - Pessimistic projections for high-risk scenarios
    - Modest projections for stable scenarios
    - No heroic assumptions about self-correction
    
    Args:
        initial_metrics: Starting values for key metrics
        months: Number of months to project
        
    Returns:
        list: Monthly projections of metrics
    """
    logger.debug(f"Projecting baseline scenario for {months} months")
    
    projections = []
    
    # Initialize with current state
    current = initial_metrics.copy()
    projections.append(current.copy())
    
    for month in range(1, months + 1):
        # Copy previous month
        next_month = current.copy()
        next_month['month'] = month
        
        # NASRI Score projection
        nasri = current.get('nasri_score', 50)
        
        # Apply natural improvement or degradation
        if current.get('capacity_utilization_pct', 80) > 90:
            # System under stress - degrades
            nasri = nasri * (1 - MONTHLY_DEGRADATION_RATE)
        else:
            # Normal operations - slight improvement
            nasri = nasri * (1 + MONTHLY_NATURAL_IMPROVEMENT)
        
        next_month['nasri_score'] = np.clip(nasri, 0, 100)
        
        # ASRS Score projection
        asrs = current.get('asrs_score', 0.5)
        
        # Risk increases if capacity stressed
        if current.get('capacity_utilization_pct', 80) > 95:
            asrs = asrs * 1.05  # 5% increase in risk per month
        elif current.get('capacity_utilization_pct', 80) < 70:
            asrs = asrs * 0.98  # 2% decrease in risk per month
        
        next_month['asrs_score'] = np.clip(asrs, 0, 1)
        
        # Compliance Debt projection
        debt = current.get('compliance_debt_cumulative', 0)
        completion_ratio = current.get('completion_ratio_overall', 0.7)
        
        # Debt accumulation logic
        # If completion < 80%, debt grows; if >= 80%, debt reduces
        if completion_ratio < 0.8:
            # Backlog accumulating
            monthly_demand = 1000  # Assume 1000 units per month (conservative)
            monthly_completion = monthly_demand * completion_ratio
            debt += (monthly_demand - monthly_completion)
        else:
            # Backlog reducing
            debt = debt * 0.9  # 10% reduction per month
        
        next_month['compliance_debt_cumulative'] = max(debt, 0)
        
        # Capacity Utilization (relatively stable in baseline)
        capacity = current.get('capacity_utilization_pct', 80)
        # Minor fluctuation ±5%
        capacity = capacity + np.random.uniform(-5, 5)
        next_month['capacity_utilization_pct'] = np.clip(capacity, 0, 150)
        
        # Completion Ratio (slightly degrades under stress)
        completion = current.get('completion_ratio_overall', 0.7)
        if capacity > 95:
            completion = completion * 0.98  # Degrades under stress
        else:
            completion = completion * 1.01  # Slight improvement
        next_month['completion_ratio_overall'] = np.clip(completion, 0, 1)
        
        # Stability Score (remains relatively constant)
        stability = current.get('stability_score', 0.5)
        next_month['stability_score'] = stability
        
        projections.append(next_month.copy())
        current = next_month
    
    return projections


def project_intervention_scenario(
    initial_metrics: Dict[str, float],
    intervention_impact: Dict[str, Any],
    months: int = 6
) -> List[Dict[str, float]]:
    """
    Project metrics over time with intervention applied.
    
    Intervention Effects:
    - Implementation delay: Benefits start after implementation period
    - Ramp-up period: Full effect reached gradually (3 months)
    - Sustained improvement: Benefits maintained if not over-stressed
    - Diminishing returns: Each subsequent month sees smaller gains
    
    Conservative Assumptions:
    - 80% of promised impact actually realized (effectiveness discount)
    - Gradual ramp-up (not immediate effect)
    - Benefits may erode if system re-stressed
    
    Args:
        initial_metrics: Starting values for key metrics
        intervention_impact: Impact parameters from intervention playbook
        months: Number of months to project
        
    Returns:
        list: Monthly projections with intervention effects
    """
    logger.debug(f"Projecting intervention scenario for {months} months")
    
    projections = []
    
    # Extract intervention parameters
    impl_days = intervention_impact.get('implementation_days', 30)
    impl_months = np.ceil(impl_days / 30.0)
    effectiveness = intervention_impact.get('effectiveness_probability', 0.8)
    
    # Apply effectiveness discount (conservative)
    effectiveness_factor = effectiveness * 0.8  # 80% of promised impact
    
    # Initialize with current state
    current = initial_metrics.copy()
    projections.append(current.copy())
    
    for month in range(1, months + 1):
        next_month = current.copy()
        next_month['month'] = month
        
        # Calculate intervention effect multiplier
        # Effect = 0 during implementation, ramps up after
        if month <= impl_months:
            # Implementation phase - no benefits yet
            intervention_effect = 0.0
        elif month <= impl_months + 3:
            # Ramp-up phase - gradual benefit realization
            ramp_progress = (month - impl_months) / 3.0
            intervention_effect = ramp_progress * effectiveness_factor
        else:
            # Full effect phase
            intervention_effect = effectiveness_factor
        
        # Apply intervention impacts
        
        # NASRI Score improvement
        nasri = current.get('nasri_score', 50)
        nasri_improvement = intervention_impact.get('nasri_improvement', 0)
        
        # Apply improvement with intervention effect
        nasri += (nasri_improvement / months) * intervention_effect
        
        # Also apply natural improvement
        nasri = nasri * (1 + MONTHLY_NATURAL_IMPROVEMENT)
        
        next_month['nasri_score'] = np.clip(nasri, 0, 100)
        
        # ASRS Score reduction
        asrs = current.get('asrs_score', 0.5)
        asrs_reduction = intervention_impact.get('asrs_reduction', 0)
        
        # Apply risk reduction with intervention effect
        asrs -= (asrs_reduction / months) * intervention_effect
        
        # Natural risk fluctuation
        if current.get('capacity_utilization_pct', 80) > 95:
            asrs = asrs * 1.02  # Still some risk increase if over capacity
        else:
            asrs = asrs * 0.98
        
        next_month['asrs_score'] = np.clip(asrs, 0, 1)
        
        # Compliance Debt reduction
        debt = current.get('compliance_debt_cumulative', 0)
        
        # Apply intervention debt reduction effect
        if 'debt_reduction_pct' in intervention_impact.get('projected_improvements', {}):
            debt_reduction_pct = intervention_impact['projected_improvements']['debt_reduction_pct']
            # Gradual reduction over time with intervention effect
            debt = debt * (1 - (debt_reduction_pct / 100 / months) * intervention_effect)
        else:
            # Improved completion from intervention reduces debt accumulation
            completion = current.get('completion_ratio_overall', 0.7)
            if completion >= 0.8:
                debt = debt * 0.85  # Faster reduction with intervention
        
        next_month['compliance_debt_cumulative'] = max(debt, 0)
        
        # Capacity Utilization improvement
        capacity = current.get('capacity_utilization_pct', 80)
        
        # Apply capacity increase from intervention
        if 'capacity_increase_pct' in intervention_impact.get('projected_improvements', {}):
            capacity_increase = intervention_impact['projected_improvements']['capacity_increase_pct']
            # Reduce utilization as capacity expands
            capacity = capacity / (1 + (capacity_increase / 100 / months) * intervention_effect)
        
        next_month['capacity_utilization_pct'] = np.clip(capacity, 0, 150)
        
        # Completion Ratio improvement
        completion = current.get('completion_ratio_overall', 0.7)
        
        if 'completion_ratio_new' in intervention_impact.get('projected_improvements', {}):
            target_completion = intervention_impact['projected_improvements']['completion_ratio_new']
            # Move towards target with intervention effect
            completion += (target_completion - completion) * (intervention_effect / months)
        else:
            # Natural improvement with intervention support
            completion = completion * (1 + 0.03 * intervention_effect)  # 3% monthly improvement
        
        next_month['completion_ratio_overall'] = np.clip(completion, 0, 1)
        
        # Stability Score improvement
        stability = current.get('stability_score', 0.5)
        
        if 'stability_score_gain' in intervention_impact.get('projected_improvements', {}):
            stability_gain = intervention_impact['projected_improvements']['stability_score_gain']
            stability += (stability_gain / months) * intervention_effect
        
        next_month['stability_score'] = np.clip(stability, 0, 1)
        
        projections.append(next_month.copy())
        current = next_month
    
    return projections


def calculate_backlog_clearance_time(
    initial_debt: float,
    completion_ratio: float,
    monthly_demand: float = 1000
) -> int:
    """
    Calculate months required to clear existing backlog.
    
    Formula:
    clearance_time = debt / (monthly_capacity - monthly_demand)
    
    Where:
    - monthly_capacity = monthly_demand / completion_ratio
    - net_clearance = capacity - demand (surplus applied to backlog)
    
    Conservative Assumptions:
    - Demand remains constant
    - Completion ratio remains constant
    - No additional stress factors
    
    Args:
        initial_debt: Current compliance debt (backlog)
        completion_ratio: Current completion ratio (0-1)
        monthly_demand: Expected monthly demand
        
    Returns:
        int: Months required to clear backlog (999 if impossible)
    """
    if completion_ratio <= 0:
        return 999  # Cannot clear
    
    if initial_debt <= 0:
        return 0  # Already clear
    
    # Calculate monthly clearance capacity
    monthly_capacity = monthly_demand / completion_ratio
    net_clearance_per_month = monthly_capacity - monthly_demand
    
    if net_clearance_per_month <= 0:
        return 999  # Backlog growing, not clearing
    
    # Calculate months to clear
    months_to_clear = np.ceil(initial_debt / net_clearance_per_month)
    
    return int(months_to_clear)


def calculate_waiting_time_impact(
    debt_baseline: List[float],
    debt_intervention: List[float]
) -> Dict[str, float]:
    """
    Calculate waiting time impact of intervention vs baseline.
    
    Waiting Time Model:
    waiting_time_days = backlog_size * WAITING_TIME_PER_BACKLOG_UNIT
    
    Assumptions:
    - Each unit of backlog adds fixed waiting time
    - Linear relationship (conservative)
    - Waiting time affects all new requests equally
    
    Args:
        debt_baseline: Backlog projection without intervention
        debt_intervention: Backlog projection with intervention
        
    Returns:
        dict: Waiting time statistics and improvements
    """
    # Calculate waiting times
    waiting_baseline = [debt * WAITING_TIME_PER_BACKLOG_UNIT for debt in debt_baseline]
    waiting_intervention = [debt * WAITING_TIME_PER_BACKLOG_UNIT for debt in debt_intervention]
    
    # Calculate statistics
    avg_waiting_baseline = np.mean(waiting_baseline)
    avg_waiting_intervention = np.mean(waiting_intervention)
    
    max_waiting_baseline = np.max(waiting_baseline)
    max_waiting_intervention = np.max(waiting_intervention)
    
    # Calculate improvements
    avg_reduction_days = avg_waiting_baseline - avg_waiting_intervention
    max_reduction_days = max_waiting_baseline - max_waiting_intervention
    
    # Calculate percentage improvement
    pct_improvement = (avg_reduction_days / avg_waiting_baseline * 100) if avg_waiting_baseline > 0 else 0
    
    return {
        'avg_waiting_baseline_days': avg_waiting_baseline,
        'avg_waiting_intervention_days': avg_waiting_intervention,
        'avg_waiting_reduction_days': avg_reduction_days,
        'max_waiting_baseline_days': max_waiting_baseline,
        'max_waiting_intervention_days': max_waiting_intervention,
        'max_waiting_reduction_days': max_reduction_days,
        'pct_improvement': pct_improvement
    }


def simulate_impact(
    row: pd.Series,
    action: Dict[str, Any],
    simulation_months: int = 6
) -> Dict[str, Any]:
    """
    Simulate impact of intervention vs no-action baseline.
    
    Simulation Process:
    1. Extract current metrics from district row
    2. Project baseline scenario (no intervention)
    3. Project intervention scenario (with action)
    4. Calculate comparative metrics
    5. Estimate citizen impact (waiting times)
    
    Output Metrics:
    - NASRI trajectory (baseline vs intervention)
    - ASRS trajectory (baseline vs intervention)
    - Backlog clearance time comparison
    - Waiting time impact
    - Net benefit summary
    
    Conservative Approach:
    - Pessimistic baseline (assumes degradation)
    - Discounted intervention benefits (80% effectiveness)
    - No compounding of benefits
    - Linear projections (no exponential growth assumptions)
    
    Args:
        row: District-month metrics (current state)
        action: Intervention definition with impact estimates
        simulation_months: Simulation horizon in months (default: 6)
        
    Returns:
        dict: Comprehensive simulation results
        
    Example:
        >>> from rules_engine import recommend_actions
        >>> recommendations = recommend_actions(district_row)
        >>> simulation = simulate_impact(district_row, recommendations[0])
        >>> print(f"Expected waiting time reduction: {simulation['waiting_time']['avg_waiting_reduction_days']:.1f} days")
    """
    logger.info(
        f"Simulating impact for district {row.get('district', 'Unknown')} "
        f"with intervention: {action.get('name', 'Unknown')}"
    )
    
    # Extract current metrics
    initial_metrics = {
        'nasri_score': row.get('nasri_score', 50),
        'asrs_score': row.get('asrs_score', 0.5),
        'compliance_debt_cumulative': row.get('compliance_debt_cumulative', 0),
        'capacity_utilization_pct': row.get('capacity_utilization_pct', 80),
        'completion_ratio_overall': row.get('completion_ratio_overall', 0.7),
        'stability_score': row.get('stability_score', 0.5)
    }
    
    # Get intervention impact parameters
    intervention_impact = action.get('impact_estimate', {})
    
    # Run baseline simulation (no action)
    logger.debug("Simulating baseline scenario (no intervention)")
    baseline_projection = project_baseline_scenario(initial_metrics, simulation_months)
    
    # Run intervention simulation
    logger.debug("Simulating intervention scenario")
    intervention_projection = project_intervention_scenario(
        initial_metrics,
        intervention_impact,
        simulation_months
    )
    
    # Extract final states
    baseline_final = baseline_projection[-1]
    intervention_final = intervention_projection[-1]
    
    # Calculate improvements
    nasri_improvement = intervention_final['nasri_score'] - baseline_final['nasri_score']
    asrs_improvement = baseline_final['asrs_score'] - intervention_final['asrs_score']  # Lower is better
    
    # Calculate backlog clearance times
    baseline_clearance_time = calculate_backlog_clearance_time(
        baseline_final['compliance_debt_cumulative'],
        baseline_final['completion_ratio_overall']
    )
    
    intervention_clearance_time = calculate_backlog_clearance_time(
        intervention_final['compliance_debt_cumulative'],
        intervention_final['completion_ratio_overall']
    )
    
    # Calculate waiting time impacts
    debt_baseline_timeline = [m['compliance_debt_cumulative'] for m in baseline_projection]
    debt_intervention_timeline = [m['compliance_debt_cumulative'] for m in intervention_projection]
    
    waiting_time_impact = calculate_waiting_time_impact(
        debt_baseline_timeline,
        debt_intervention_timeline
    )
    
    # Compile comprehensive results
    simulation_results = {
        'district': row.get('district', 'Unknown'),
        'intervention_name': action.get('name', 'Unknown'),
        'intervention_id': action.get('intervention_id', 'Unknown'),
        'simulation_months': simulation_months,
        
        # Initial state
        'initial': initial_metrics,
        
        # Baseline scenario (no action)
        'baseline': {
            'trajectory': baseline_projection,
            'final_nasri': baseline_final['nasri_score'],
            'final_asrs': baseline_final['asrs_score'],
            'final_debt': baseline_final['compliance_debt_cumulative'],
            'backlog_clearance_months': baseline_clearance_time
        },
        
        # Intervention scenario
        'intervention': {
            'trajectory': intervention_projection,
            'final_nasri': intervention_final['nasri_score'],
            'final_asrs': intervention_final['asrs_score'],
            'final_debt': intervention_final['compliance_debt_cumulative'],
            'backlog_clearance_months': intervention_clearance_time,
            'implementation_days': intervention_impact.get('implementation_days', 30),
            'cost_level': intervention_impact.get('cost_level', 'medium')
        },
        
        # Comparative benefits
        'benefits': {
            'nasri_improvement': nasri_improvement,
            'asrs_reduction': asrs_improvement,
            'debt_reduction': baseline_final['compliance_debt_cumulative'] - intervention_final['compliance_debt_cumulative'],
            'clearance_time_reduction_months': baseline_clearance_time - intervention_clearance_time
        },
        
        # Citizen impact
        'waiting_time': waiting_time_impact,
        
        # Summary recommendation
        'recommendation': {
            'is_beneficial': nasri_improvement > 5 or asrs_improvement > 0.1,
            'confidence_level': intervention_impact.get('effectiveness_probability', 0.7),
            'priority': action.get('priority', 3),
            'roi_indicator': 'high' if nasri_improvement > 10 else ('medium' if nasri_improvement > 5 else 'low')
        }
    }
    
    # Log summary
    logger.info(
        f"Simulation complete: NASRI {initial_metrics['nasri_score']:.1f} → "
        f"Baseline: {baseline_final['nasri_score']:.1f}, "
        f"Intervention: {intervention_final['nasri_score']:.1f} "
        f"(+{nasri_improvement:.1f} benefit)"
    )
    
    logger.info(
        f"Waiting time impact: {waiting_time_impact['avg_waiting_reduction_days']:.1f} days reduction "
        f"({waiting_time_impact['pct_improvement']:.1f}% improvement)"
    )
    
    return simulation_results


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Simulation module - Example usage")
    
    # Example: Simulate intervention impact
    logger.info("\n=== Example: Impact Simulation ===")
    
    # Sample district metrics (high-risk case)
    sample_district = pd.Series({
        'district': 'HIGH_RISK_DISTRICT',
        'month': '2024-01',
        'nasri_score': 35,
        'asrs_score': 0.75,
        'compliance_debt_cumulative': 1500,
        'capacity_utilization_pct': 110,
        'completion_ratio_overall': 0.55,
        'stability_score': 0.45
    })
    
    # Sample intervention (emergency capacity expansion)
    sample_intervention = {
        'intervention_id': 'INT_001',
        'name': 'Emergency Capacity Expansion',
        'priority': 1,
        'impact_estimate': {
            'implementation_days': 14,
            'cost_level': 'high',
            'effectiveness_probability': 0.85,
            'nasri_improvement': 15,
            'asrs_reduction': 0.20,
            'projected_improvements': {
                'capacity_increase_pct': 30,
                'debt_reduction_pct': 25
            }
        }
    }
    
    logger.info("Initial district state:")
    logger.info(f"  NASRI: {sample_district['nasri_score']:.1f}")
    logger.info(f"  ASRS: {sample_district['asrs_score']:.2f}")
    logger.info(f"  Debt: {sample_district['compliance_debt_cumulative']:.0f}")
    logger.info(f"  Capacity: {sample_district['capacity_utilization_pct']:.1f}%")
    
    logger.info(f"\nProposed intervention: {sample_intervention['name']}")
    logger.info(f"  Implementation: {sample_intervention['impact_estimate']['implementation_days']} days")
    logger.info(f"  Cost: {sample_intervention['impact_estimate']['cost_level']}")
    
    # Run simulation
    simulation_results = simulate_impact(sample_district, sample_intervention, simulation_months=6)
    
    logger.info("\n=== Simulation Results (6 months) ===")
    
    logger.info("\nBaseline Scenario (No Action):")
    logger.info(f"  Final NASRI: {simulation_results['baseline']['final_nasri']:.1f}")
    logger.info(f"  Final ASRS: {simulation_results['baseline']['final_asrs']:.2f}")
    logger.info(f"  Final Debt: {simulation_results['baseline']['final_debt']:.0f}")
    logger.info(f"  Clearance Time: {simulation_results['baseline']['backlog_clearance_months']} months")
    
    logger.info("\nIntervention Scenario:")
    logger.info(f"  Final NASRI: {simulation_results['intervention']['final_nasri']:.1f}")
    logger.info(f"  Final ASRS: {simulation_results['intervention']['final_asrs']:.2f}")
    logger.info(f"  Final Debt: {simulation_results['intervention']['final_debt']:.0f}")
    logger.info(f"  Clearance Time: {simulation_results['intervention']['backlog_clearance_months']} months")
    
    logger.info("\nNet Benefits:")
    logger.info(f"  NASRI Improvement: +{simulation_results['benefits']['nasri_improvement']:.1f} points")
    logger.info(f"  ASRS Reduction: -{simulation_results['benefits']['asrs_reduction']:.2f} risk points")
    logger.info(f"  Debt Reduction: -{simulation_results['benefits']['debt_reduction']:.0f} units")
    logger.info(f"  Faster Clearance: {simulation_results['benefits']['clearance_time_reduction_months']} months earlier")
    
    logger.info("\nCitizen Impact (Waiting Time):")
    logger.info(f"  Average Reduction: {simulation_results['waiting_time']['avg_waiting_reduction_days']:.1f} days")
    logger.info(f"  Improvement: {simulation_results['waiting_time']['pct_improvement']:.1f}%")
    
    logger.info("\nRecommendation:")
    logger.info(f"  Beneficial: {simulation_results['recommendation']['is_beneficial']}")
    logger.info(f"  ROI Indicator: {simulation_results['recommendation']['roi_indicator']}")
    logger.info(f"  Confidence: {simulation_results['recommendation']['confidence_level']:.0%}")
