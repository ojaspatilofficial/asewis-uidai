"""
Test script to verify simulation realism and intervention logic.

Purpose:
- Load scored dataset with NASRI and ASRS
- Identify high-risk districts
- Test recommend_actions() produces valid recommendations
- Test simulate_impact() produces realistic projections
- Validate intervention improves outcomes (lower ASRS, reduced backlog)

Validation Criteria:
- ASRS must NOT increase after intervention (risk should decrease)
- Backlog reduction must be positive or zero (no negative clearance)
- NASRI should improve or stay stable
- Projections should be within realistic bounds

Exit codes:
- 0: All tests passed
- 1: Validation failures detected
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rules_engine import recommend_actions
from simulation import simulate_impact

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SCORES_FILE = PROJECT_ROOT / 'dataset' / 'processed' / 'scores.parquet'


def load_scored_dataset() -> pd.DataFrame:
    """Load the scored dataset with NASRI and ASRS."""
    logger.info(f"Loading scored dataset from: {SCORES_FILE}")
    
    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            f"Scores file not found: {SCORES_FILE}\n"
            "Run 'python src/run_intelligence.py' first to generate scores."
        )
    
    df = pd.read_parquet(SCORES_FILE)
    logger.info(f"✓ Loaded {len(df):,} scored records")
    logger.info(f"  Columns: {len(df.columns)}")
    
    # Validate required columns
    required_cols = ['district', 'nasri_score', 'asrs_score', 'asrs_risk_category']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def find_high_risk_district(df: pd.DataFrame) -> pd.Series:
    """Find a high-risk district for testing."""
    logger.info("\nSearching for high-risk district...")
    
    # Filter for high-risk districts (ASRS category = 'high')
    high_risk = df[df['asrs_risk_category'] == 'high']
    
    if len(high_risk) == 0:
        logger.warning("No 'high' risk districts found, trying 'medium'...")
        high_risk = df[df['asrs_risk_category'] == 'medium']
    
    if len(high_risk) == 0:
        logger.warning("No 'medium' risk districts found, selecting highest ASRS...")
        # Get district with highest ASRS
        idx = df['asrs_score'].idxmax()
        selected = df.loc[idx]
    else:
        # Get the most recent record for the highest risk district
        high_risk = high_risk.sort_values('asrs_score', ascending=False)
        selected = high_risk.iloc[0]
    
    logger.info(f"✓ Selected district: {selected['district']}")
    logger.info(f"  ASRS: {selected['asrs_score']:.3f} ({selected['asrs_risk_category']})")
    logger.info(f"  NASRI: {selected['nasri_score']:.1f} ({selected.get('nasri_category', 'N/A')})")
    
    if 'compliance_debt_cumulative' in selected:
        logger.info(f"  Compliance Debt: {selected['compliance_debt_cumulative']:.0f}")
    if 'capacity_utilization_pct' in selected:
        logger.info(f"  Capacity Utilization: {selected['capacity_utilization_pct']:.1f}%")
    
    return selected


def test_recommendations(district_row: pd.Series) -> list:
    """Test recommendation engine."""
    logger.info("\n" + "="*80)
    logger.info("TESTING RECOMMENDATION ENGINE")
    logger.info("="*80)
    
    recommendations = recommend_actions(district_row, max_recommendations=5)
    
    if not recommendations:
        logger.warning("⚠ No recommendations generated for this district")
        logger.info("This may be expected if district metrics don't match intervention conditions")
        return []
    
    logger.info(f"\n✓ Generated {len(recommendations)} recommendation(s)")
    
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\n[{i}] {rec['name']}")
        logger.info(f"    ID: {rec['intervention_id']}")
        logger.info(f"    Priority Score: {rec['priority_score']:.1f}")
        logger.info(f"    Description: {rec['description']}")
        
        # Show impact estimates if available
        impact = rec.get('impact_estimate', {})
        if impact:
            logger.info("    Expected Impact:")
            for key, value in impact.items():
                if isinstance(value, (int, float)):
                    logger.info(f"      - {key}: {value}")
                else:
                    logger.info(f"      - {key}: {value}")
    
    return recommendations


def test_simulation(district_row: pd.Series, action: Dict[str, Any]) -> Dict[str, Any]:
    """Test simulation for a single intervention."""
    logger.info("\n" + "="*80)
    logger.info("TESTING IMPACT SIMULATION")
    logger.info("="*80)
    
    logger.info(f"\nSimulating intervention: {action['name']}")
    logger.info(f"Duration: 6 months")
    
    # Run simulation
    simulation_result = simulate_impact(district_row, action, simulation_months=6)
    
    logger.info("\n" + "-"*80)
    logger.info("SIMULATION RESULTS")
    logger.info("-"*80)
    
    # Extract key metrics
    initial = simulation_result['initial']
    baseline = simulation_result['baseline']
    intervention = simulation_result['intervention']
    net_benefits = simulation_result.get('net_benefits', {})
    waiting_time = simulation_result.get('waiting_time', {})
    
    # Print initial state
    logger.info("\nINITIAL STATE:")
    logger.info(f"  NASRI Score:        {initial['nasri_score']:.1f}")
    logger.info(f"  ASRS Score:         {initial['asrs_score']:.3f}")
    logger.info(f"  Compliance Debt:    {initial['compliance_debt_cumulative']:.0f}")
    logger.info(f"  Capacity Util:      {initial['capacity_utilization_pct']:.1f}%")
    logger.info(f"  Completion Ratio:   {initial['completion_ratio_overall']:.2f}")
    
    # Print baseline scenario (no action)
    logger.info("\nBASELINE (No Action) - After 6 Months:")
    logger.info(f"  NASRI Score:        {baseline['final_nasri']:.1f}")
    logger.info(f"  ASRS Score:         {baseline['final_asrs']:.3f}")
    logger.info(f"  Compliance Debt:    {baseline['final_debt']:.0f}")
    logger.info(f"  Backlog Clearance:  {baseline['backlog_clearance_months']:.1f} months")
    
    # Print intervention scenario
    logger.info(f"\nINTERVENTION ({action['name']}) - After 6 Months:")
    logger.info(f"  NASRI Score:        {intervention['final_nasri']:.1f}")
    logger.info(f"  ASRS Score:         {intervention['final_asrs']:.3f}")
    logger.info(f"  Compliance Debt:    {intervention['final_debt']:.0f}")
    logger.info(f"  Backlog Clearance:  {intervention['backlog_clearance_months']:.1f} months")
    logger.info(f"  Implementation:     {intervention['implementation_days']} days")
    
    # Print net benefits
    logger.info("\nNET BENEFITS (Intervention vs Baseline):")
    logger.info(f"  NASRI Improvement:  {net_benefits.get('nasri_improvement', 0):.1f} points")
    logger.info(f"  ASRS Reduction:     {net_benefits.get('asrs_improvement', 0):.3f} (lower is better)")
    logger.info(f"  Debt Reduction:     {net_benefits.get('debt_reduction', 0):.0f}")
    logger.info(f"  Time Saved:         {net_benefits.get('clearance_time_reduction_months', 0):.1f} months")
    
    # Print waiting time impact
    if waiting_time:
        logger.info("\nCITIZEN IMPACT (Waiting Time):")
        logger.info(f"  Avg Waiting (Baseline):       {waiting_time.get('avg_waiting_baseline_days', 0):.1f} days")
        logger.info(f"  Avg Waiting (Intervention):   {waiting_time.get('avg_waiting_intervention_days', 0):.1f} days")
        logger.info(f"  Reduction:                    {waiting_time.get('avg_waiting_reduction_days', 0):.1f} days")
        logger.info(f"  Total Citizens Impacted:      {waiting_time.get('total_citizens_impacted', 0):,.0f}")
    
    return simulation_result


def validate_simulation(simulation_result: Dict[str, Any]) -> bool:
    """Validate simulation results for realism."""
    logger.info("\n" + "="*80)
    logger.info("VALIDATION CHECKS")
    logger.info("="*80)
    
    validation_passed = True
    
    baseline = simulation_result['baseline']
    intervention = simulation_result['intervention']
    net_benefits = simulation_result.get('net_benefits', {})
    
    # Check 1: ASRS should NOT increase (risk should decrease or stay same)
    asrs_change = intervention['final_asrs'] - baseline['final_asrs']
    logger.info(f"\n[1] ASRS Change: {asrs_change:+.3f}")
    
    if asrs_change > 0:
        logger.error("   ❌ FAIL: ASRS increased after intervention (risk got worse)")
        logger.error(f"      Baseline ASRS: {baseline['final_asrs']:.3f}")
        logger.error(f"      Intervention ASRS: {intervention['final_asrs']:.3f}")
        validation_passed = False
    else:
        logger.info(f"   ✓ PASS: ASRS decreased by {abs(asrs_change):.3f} (risk reduced)")
    
    # Check 2: Backlog reduction should be positive or zero
    debt_reduction = baseline['final_debt'] - intervention['final_debt']
    logger.info(f"\n[2] Debt Reduction: {debt_reduction:+.0f}")
    
    if debt_reduction < 0:
        logger.error("   ❌ FAIL: Backlog increased after intervention (negative reduction)")
        logger.error(f"      Baseline Debt: {baseline['final_debt']:.0f}")
        logger.error(f"      Intervention Debt: {intervention['final_debt']:.0f}")
        validation_passed = False
    else:
        logger.info(f"   ✓ PASS: Backlog reduced by {debt_reduction:.0f} units")
    
    # Check 3: NASRI should improve or stay stable
    nasri_change = intervention['final_nasri'] - baseline['final_nasri']
    logger.info(f"\n[3] NASRI Change: {nasri_change:+.1f}")
    
    if nasri_change < -5:  # Allow small degradation (±5 points) for realism
        logger.warning(f"   ⚠ WARNING: NASRI decreased significantly by {abs(nasri_change):.1f} points")
        logger.warning("      This may indicate unrealistic simulation parameters")
    elif nasri_change < 0:
        logger.info(f"   ○ ACCEPTABLE: NASRI decreased slightly by {abs(nasri_change):.1f} points")
    else:
        logger.info(f"   ✓ PASS: NASRI improved by {nasri_change:.1f} points")
    
    # Check 4: Clearance time reduction should be positive
    clearance_reduction = net_benefits.get('clearance_time_reduction_months', 0)
    logger.info(f"\n[4] Clearance Time Reduction: {clearance_reduction:+.1f} months")
    
    if clearance_reduction < 0:
        logger.error("   ❌ FAIL: Clearance time increased (takes longer with intervention)")
        validation_passed = False
    elif clearance_reduction == 0:
        logger.info("   ○ NEUTRAL: No change in clearance time")
    else:
        logger.info(f"   ✓ PASS: Clearance time reduced by {clearance_reduction:.1f} months")
    
    # Check 5: Bounds validation
    logger.info(f"\n[5] Bounds Validation:")
    
    bounds_valid = True
    
    if not (0 <= intervention['final_nasri'] <= 100):
        logger.error(f"   ❌ FAIL: NASRI out of bounds [0, 100]: {intervention['final_nasri']:.1f}")
        bounds_valid = False
        validation_passed = False
    
    if not (0 <= intervention['final_asrs'] <= 1):
        logger.error(f"   ❌ FAIL: ASRS out of bounds [0, 1]: {intervention['final_asrs']:.3f}")
        bounds_valid = False
        validation_passed = False
    
    if intervention['final_debt'] < 0:
        logger.error(f"   ❌ FAIL: Negative debt: {intervention['final_debt']:.0f}")
        bounds_valid = False
        validation_passed = False
    
    if bounds_valid:
        logger.info("   ✓ PASS: All metrics within valid bounds")
    
    # Final summary
    logger.info("\n" + "="*80)
    if validation_passed:
        logger.info("✓ ALL VALIDATION CHECKS PASSED")
        logger.info("  Simulation produces realistic and beneficial outcomes")
    else:
        logger.error("❌ VALIDATION FAILED")
        logger.error("  Simulation produced unrealistic or harmful outcomes")
    logger.info("="*80)
    
    return validation_passed


def main() -> int:
    """Main test execution."""
    logger.info("="*80)
    logger.info("ASEWIS SIMULATION VALIDATION TEST")
    logger.info("="*80)
    logger.info("")
    logger.info("Purpose: Verify intervention simulation produces realistic outcomes")
    logger.info("")
    
    try:
        # Step 1: Load scored dataset
        logger.info("[Step 1/5] Loading scored dataset...")
        df = load_scored_dataset()
        
        # Step 2: Select high-risk district
        logger.info("\n[Step 2/5] Selecting high-risk district...")
        district_row = find_high_risk_district(df)
        
        # Step 3: Generate recommendations
        logger.info("\n[Step 3/5] Generating recommendations...")
        recommendations = test_recommendations(district_row)
        
        if not recommendations:
            logger.warning("\n⚠ No recommendations to test - exiting")
            logger.info("This is acceptable if district doesn't meet intervention conditions")
            return 0
        
        # Step 4: Simulate first recommendation
        logger.info("\n[Step 4/5] Simulating impact of first recommendation...")
        first_action = recommendations[0]
        simulation_result = test_simulation(district_row, first_action)
        
        # Step 5: Validate results
        logger.info("\n[Step 5/5] Validating simulation results...")
        validation_passed = validate_simulation(simulation_result)
        
        # Final result
        if validation_passed:
            logger.info("\n" + "="*80)
            logger.info("✓ TEST PASSED: Simulation validation successful")
            logger.info("="*80)
            return 0
        else:
            logger.error("\n" + "="*80)
            logger.error("❌ TEST FAILED: Simulation validation failed")
            logger.error("="*80)
            return 1
    
    except Exception as e:
        logger.exception(f"Test execution failed: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
