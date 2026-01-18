"""
Forecasting module for operational demand planning.

This module provides lightweight statistical forecasting methods suitable for
short-term operational planning (30-90 days). Uses classical time series methods
that are interpretable, fast, and require minimal data.

No deep learning - focuses on simple, explainable models that work well with
limited historical data typical in operational contexts.
"""

import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


def calculate_rolling_mean_forecast(
    series: pd.Series,
    window_size: int = 3,
    forecast_periods: int = 30
) -> np.ndarray:
    """
    Calculate forecast using rolling mean (moving average).
    
    Method: Simple Moving Average (SMA)
    The forecast for all future periods equals the average of the last N observations.
    
    Assumptions:
    - Recent past is a good indicator of near future
    - Seasonal patterns are weak or already accounted for
    - Works best for stable demand with minor fluctuations
    
    Pros:
    - Simple to understand and explain
    - Robust to outliers (with appropriate window)
    - Fast computation
    
    Cons:
    - Assumes flat future (no trend)
    - All future periods get same forecast
    - Lags behind actual changes
    
    Args:
        series: Historical time series data (e.g., monthly demand)
        window_size: Number of recent periods to average (default: 3 months)
        forecast_periods: Number of periods to forecast (default: 30 days)
        
    Returns:
        np.ndarray: Forecast values for all forecast_periods
        
    Example:
        >>> history = pd.Series([100, 110, 105, 115, 108])
        >>> forecast = calculate_rolling_mean_forecast(history, window_size=3, forecast_periods=30)
        >>> # All 30 days will have same value: (105 + 115 + 108) / 3 ≈ 109.3
    """
    if len(series) < window_size:
        logger.warning(
            f"Insufficient data for rolling mean: {len(series)} points, "
            f"need {window_size}. Using all available data."
        )
        window_size = len(series)
    
    # Calculate rolling mean of last N periods
    # This is our constant forecast for all future periods
    rolling_mean = series.iloc[-window_size:].mean()
    
    # Repeat this forecast for all forecast periods
    # Assumption: demand stays at recent average level
    forecast = np.full(forecast_periods, rolling_mean)
    
    logger.debug(
        f"Rolling mean forecast: {rolling_mean:.2f} "
        f"(based on last {window_size} periods)"
    )
    
    return forecast


def calculate_exponential_smoothing_forecast(
    series: pd.Series,
    alpha: float = 0.3,
    forecast_periods: int = 30
) -> np.ndarray:
    """
    Calculate forecast using exponential smoothing.
    
    Method: Single Exponential Smoothing (SES)
    Weighted average where recent observations have exponentially higher weights.
    
    Formula:
    S[t] = α * Y[t] + (1-α) * S[t-1]
    Where:
    - S[t] = Smoothed value at time t
    - Y[t] = Actual observation at time t
    - α (alpha) = Smoothing parameter (0 < α < 1)
    
    Assumptions:
    - Recent data more relevant than older data
    - No trend or seasonality (level model only)
    - Stationary demand patterns
    
    Alpha parameter interpretation:
    - α = 0.1 : Heavy smoothing, slow response to changes (stable demand)
    - α = 0.3 : Moderate smoothing, balanced (DEFAULT)
    - α = 0.7 : Light smoothing, fast response (volatile demand)
    
    Pros:
    - Adapts to recent changes faster than simple average
    - Single parameter easy to tune
    - Works well for short-term forecasts
    
    Cons:
    - Still assumes flat future (no trend)
    - Requires tuning alpha for each series
    - May lag behind sustained trends
    
    Args:
        series: Historical time series data
        alpha: Smoothing parameter (0-1), higher = more weight to recent data
        forecast_periods: Number of periods to forecast
        
    Returns:
        np.ndarray: Forecast values for all forecast_periods
    """
    if len(series) == 0:
        logger.error("Empty series provided for exponential smoothing")
        return np.zeros(forecast_periods)
    
    # Initialize with first observation
    smoothed_value = series.iloc[0]
    
    # Apply exponential smoothing to all historical points
    # Each iteration updates the smoothed value based on actual observation
    for actual_value in series.iloc[1:]:
        smoothed_value = alpha * actual_value + (1 - alpha) * smoothed_value
    
    # The final smoothed value is our forecast for all future periods
    # Assumption: demand will continue at this smoothed level
    forecast = np.full(forecast_periods, smoothed_value)
    
    logger.debug(
        f"Exponential smoothing forecast: {smoothed_value:.2f} "
        f"(alpha={alpha})"
    )
    
    return forecast


def calculate_trend_adjusted_forecast(
    series: pd.Series,
    forecast_periods: int = 30
) -> np.ndarray:
    """
    Calculate forecast with linear trend adjustment.
    
    Method: Linear Regression Trend
    Fits a straight line through historical data and projects it forward.
    
    Formula:
    Y[t] = a + b*t
    Where:
    - a = intercept (starting level)
    - b = slope (trend per period)
    - t = time period
    
    Assumptions:
    - Linear trend continues into future
    - No structural breaks or regime changes
    - Trend observed in past is stable
    
    Use cases:
    - Growing or declining demand patterns
    - Capacity planning with sustained trends
    - Medium-term forecasts (30-90 days)
    
    Pros:
    - Captures trend direction and magnitude
    - Simple to interpret and explain
    - Works well when clear trend exists
    
    Cons:
    - Assumes trend continues unchanged
    - Can produce unrealistic forecasts for long horizons
    - Sensitive to outliers at endpoints
    
    Args:
        series: Historical time series data
        forecast_periods: Number of periods to forecast
        
    Returns:
        np.ndarray: Forecast values with trend projection
    """
    if len(series) < 2:
        logger.warning("Insufficient data for trend calculation, using last value")
        return np.full(forecast_periods, series.iloc[-1] if len(series) > 0 else 0)
    
    # Create time index (0, 1, 2, ..., n-1)
    x = np.arange(len(series))
    y = series.values
    
    # Fit linear regression: y = a + b*x
    # Using least squares method
    slope, intercept = np.polyfit(x, y, deg=1)
    
    # Project trend forward
    # Future time indices: n, n+1, n+2, ..., n+forecast_periods-1
    future_x = np.arange(len(series), len(series) + forecast_periods)
    forecast = intercept + slope * future_x
    
    # Ensure non-negative forecasts (demand cannot be negative)
    forecast = np.maximum(forecast, 0)
    
    logger.debug(
        f"Trend-adjusted forecast: slope={slope:.2f}/period, "
        f"intercept={intercept:.2f}"
    )
    
    return forecast


def calculate_confidence_bounds(
    series: pd.Series,
    forecast: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence bounds for forecast using historical volatility.
    
    Method: Prediction Intervals based on Historical Standard Deviation
    
    Formula:
    Upper Bound = Forecast + z * σ * √h
    Lower Bound = Forecast - z * σ * √h
    
    Where:
    - z = z-score for confidence level (1.96 for 95%)
    - σ = historical standard deviation (volatility)
    - h = forecast horizon (1, 2, 3, ... periods ahead)
    
    Assumptions:
    - Forecast errors are normally distributed
    - Historical volatility represents future uncertainty
    - Error variance increases with forecast horizon (√h adjustment)
    
    Interpretation:
    - 95% confidence: True value expected to fall within bounds 95% of time
    - Wider bounds = more uncertainty
    - Bounds widen as we forecast further into future
    
    Args:
        series: Historical time series data
        forecast: Point forecast values
        confidence_level: Confidence level (default: 0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound) arrays
    """
    # Calculate historical standard deviation (measure of volatility)
    historical_std = series.std()
    
    if pd.isna(historical_std) or historical_std == 0:
        logger.warning("Cannot calculate standard deviation, using 10% of forecast")
        # Fallback: use 10% of forecast as uncertainty
        margin = forecast * 0.1
        return forecast - margin, forecast + margin
    
    # Get z-score for desired confidence level
    # 95% confidence -> z = 1.96
    # 99% confidence -> z = 2.58
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate forecast horizon adjustment
    # Uncertainty grows with √horizon (random walk assumption)
    # Period 1: √1 = 1.0x, Period 30: √30 = 5.5x
    horizons = np.arange(1, len(forecast) + 1)
    horizon_adjustment = np.sqrt(horizons)
    
    # Calculate margins (prediction intervals)
    # Margins increase as we forecast further into future
    margins = z_score * historical_std * horizon_adjustment
    
    # Calculate bounds
    lower_bound = forecast - margins
    upper_bound = forecast + margins
    
    # Ensure non-negative lower bound (demand cannot be negative)
    lower_bound = np.maximum(lower_bound, 0)
    
    logger.debug(
        f"Confidence bounds: ±{z_score:.2f} std devs ({confidence_level:.0%} confidence), "
        f"historical σ={historical_std:.2f}"
    )
    
    return lower_bound, upper_bound


def select_best_method(
    series: pd.Series,
    methods_forecasts: Dict[str, np.ndarray]
) -> str:
    """
    Select best forecasting method based on historical accuracy.
    
    Method: Cross-validation on recent history
    Simulates forecast accuracy by predicting last period and comparing to actual.
    
    Process:
    1. Hold out last observation as test
    2. Forecast it using each method on remaining data
    3. Calculate error (absolute difference)
    4. Select method with lowest error
    
    Assumptions:
    - Recent accuracy indicates future accuracy
    - Single-period validation sufficient for method selection
    - All methods evaluated on same data
    
    Args:
        series: Historical time series data
        methods_forecasts: Dictionary of method names to forecast arrays
        
    Returns:
        str: Name of best-performing method
    """
    if len(series) < 2:
        logger.warning("Insufficient data for method selection, defaulting to exponential_smoothing")
        return 'exponential_smoothing'
    
    # Hold out last observation for validation
    train_series = series.iloc[:-1]
    actual_value = series.iloc[-1]
    
    errors = {}
    
    for method_name in methods_forecasts.keys():
        try:
            # Generate forecast on training data
            if method_name == 'rolling_mean':
                test_forecast = calculate_rolling_mean_forecast(train_series, window_size=3, forecast_periods=1)[0]
            elif method_name == 'exponential_smoothing':
                test_forecast = calculate_exponential_smoothing_forecast(train_series, alpha=0.3, forecast_periods=1)[0]
            elif method_name == 'trend_adjusted':
                test_forecast = calculate_trend_adjusted_forecast(train_series, forecast_periods=1)[0]
            else:
                continue
            
            # Calculate absolute error
            error = abs(actual_value - test_forecast)
            errors[method_name] = error
            
        except Exception as e:
            logger.warning(f"Error validating {method_name}: {e}")
            errors[method_name] = float('inf')
    
    if not errors:
        logger.warning("No methods validated successfully, defaulting to exponential_smoothing")
        return 'exponential_smoothing'
    
    # Select method with minimum error
    best_method = min(errors, key=errors.get)
    
    logger.debug(
        f"Method selection errors: {errors}, selected: {best_method}"
    )
    
    return best_method


def forecast_demand(
    district_df: pd.DataFrame,
    forecast_horizons: List[int] = [30, 60, 90],
    confidence_level: float = 0.95,
    auto_select_method: bool = True
) -> Dict:
    """
    Forecast demand for a district using multiple statistical methods.
    
    This is the main forecasting entry point that:
    1. Prepares time series data
    2. Applies multiple forecasting methods
    3. Selects best method (optional)
    4. Calculates confidence bounds
    5. Returns structured forecast output
    
    Forecasting Strategy:
    - Use last 6-12 months of data (if available)
    - Apply 3 methods: rolling mean, exponential smoothing, trend-adjusted
    - Auto-select best method based on recent accuracy
    - Provide point forecast + uncertainty bounds
    
    Assumptions:
    - Input is district-level time series (sorted by date)
    - Demand column exists (enrolment_count or similar)
    - Data is at monthly granularity
    - No major structural breaks in recent past
    
    Args:
        district_df: DataFrame with time series data for one district
        forecast_horizons: List of forecast horizons in days (default: [30, 60, 90])
        confidence_level: Confidence level for bounds (default: 0.95)
        auto_select_method: Whether to auto-select best method (default: True)
        
    Returns:
        dict: Forecast results containing:
            - district: District name
            - forecast_date: When forecast was generated
            - method: Forecasting method used
            - forecasts: Dict of horizon -> {point, lower, upper}
            - historical_stats: Historical mean, std, trend
            
    Example:
        >>> district_data = df[df['district'] == 'BANGALORE']
        >>> forecast = forecast_demand(district_data, forecast_horizons=[30, 60, 90])
        >>> print(forecast['forecasts'][30]['point'])  # 30-day point forecast
        >>> print(forecast['forecasts'][30]['lower'])  # 30-day lower bound
    """
    logger.info(f"Generating demand forecast for district data ({len(district_df)} records)")
    
    # Validate input
    if district_df.empty:
        logger.error("Empty DataFrame provided for forecasting")
        return {
            'error': 'Empty DataFrame',
            'forecasts': {}
        }
    
    # Identify district name
    district_col = [col for col in district_df.columns if 'district' in col.lower()]
    district_name = district_df[district_col[0]].iloc[0] if district_col else 'Unknown'
    
    # Identify demand column
    demand_cols = [col for col in district_df.columns if 'count' in col.lower()]
    if not demand_cols:
        logger.error("No demand/count column found in data")
        return {
            'error': 'No demand column found',
            'district': district_name,
            'forecasts': {}
        }
    
    demand_col = demand_cols[0]
    
    # Identify date column and sort
    date_cols = [col for col in district_df.columns if any(kw in col.lower() for kw in ['date', 'month'])]
    if date_cols:
        district_df = district_df.sort_values(date_cols[0])
    
    # Extract time series
    # Use last 12 months of data for forecasting (if available)
    # Assumption: Recent 12 months best represents current demand patterns
    time_series = district_df[demand_col].tail(12)
    
    if len(time_series) == 0:
        logger.error("No valid demand data for forecasting")
        return {
            'error': 'No valid demand data',
            'district': district_name,
            'forecasts': {}
        }
    
    logger.info(
        f"Forecasting for {district_name} using {len(time_series)} historical points "
        f"(mean={time_series.mean():.1f}, std={time_series.std():.1f})"
    )
    
    # Generate forecasts using multiple methods
    # We compute all methods and compare them
    max_horizon = max(forecast_horizons)
    
    forecasts_by_method = {
        'rolling_mean': calculate_rolling_mean_forecast(
            time_series, 
            window_size=3, 
            forecast_periods=max_horizon
        ),
        'exponential_smoothing': calculate_exponential_smoothing_forecast(
            time_series, 
            alpha=0.3, 
            forecast_periods=max_horizon
        ),
        'trend_adjusted': calculate_trend_adjusted_forecast(
            time_series, 
            forecast_periods=max_horizon
        )
    }
    
    # Select best method based on validation
    if auto_select_method:
        selected_method = select_best_method(time_series, forecasts_by_method)
        logger.info(f"Auto-selected method: {selected_method}")
    else:
        # Default to exponential smoothing
        selected_method = 'exponential_smoothing'
        logger.info(f"Using default method: {selected_method}")
    
    # Get forecast from selected method
    point_forecast = forecasts_by_method[selected_method]
    
    # Calculate confidence bounds
    lower_bound, upper_bound = calculate_confidence_bounds(
        time_series,
        point_forecast,
        confidence_level=confidence_level
    )
    
    # Structure output for requested horizons
    forecast_results = {}
    
    for horizon in forecast_horizons:
        if horizon > max_horizon:
            logger.warning(f"Horizon {horizon} exceeds computed forecast, using max")
            idx = max_horizon - 1
        else:
            idx = horizon - 1
        
        forecast_results[horizon] = {
            'point': float(point_forecast[idx]),
            'lower': float(lower_bound[idx]),
            'upper': float(upper_bound[idx]),
            'confidence_level': confidence_level
        }
        
        logger.info(
            f"  {horizon}-day forecast: {point_forecast[idx]:.1f} "
            f"[{lower_bound[idx]:.1f}, {upper_bound[idx]:.1f}]"
        )
    
    # Calculate historical statistics for context
    historical_stats = {
        'mean': float(time_series.mean()),
        'std': float(time_series.std()),
        'min': float(time_series.min()),
        'max': float(time_series.max()),
        'trend': float(time_series.diff().mean()),  # Average change per period
        'cv': float(time_series.std() / time_series.mean()) if time_series.mean() > 0 else 0
    }
    
    # Compile final output
    output = {
        'district': district_name,
        'forecast_date': datetime.now().strftime('%Y-%m-%d'),
        'method': selected_method,
        'forecasts': forecast_results,
        'historical_stats': historical_stats,
        'data_points_used': len(time_series),
        'all_methods': {
            method: {
                horizon: float(forecasts_by_method[method][horizon-1])
                for horizon in forecast_horizons
            }
            for method in forecasts_by_method.keys()
        }
    }
    
    return output


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Forecasting module - Example usage")
    
    # Example: Create sample district time series data
    logger.info("\n=== Example: Demand Forecasting ===")
    
    sample_district_data = pd.DataFrame({
        'district': ['BANGALORE'] * 12,
        'month': pd.date_range('2023-01', periods=12, freq='MS').strftime('%Y-%m'),
        'enrolment_count': [1000, 1050, 1100, 1080, 1150, 1200, 1180, 1250, 1300, 1280, 1350, 1400]
    })
    
    logger.info("Historical demand data:")
    logger.info(f"\n{sample_district_data}")
    
    # Generate forecast
    forecast_result = forecast_demand(
        sample_district_data,
        forecast_horizons=[30, 60, 90],
        confidence_level=0.95,
        auto_select_method=True
    )
    
    logger.info("\n=== Forecast Results ===")
    logger.info(f"District: {forecast_result['district']}")
    logger.info(f"Method: {forecast_result['method']}")
    logger.info(f"Forecast date: {forecast_result['forecast_date']}")
    logger.info(f"Data points used: {forecast_result['data_points_used']}")
    
    logger.info("\n=== Historical Statistics ===")
    for stat, value in forecast_result['historical_stats'].items():
        logger.info(f"  {stat}: {value:.2f}")
    
    logger.info("\n=== Demand Forecasts ===")
    for horizon, forecast in forecast_result['forecasts'].items():
        logger.info(
            f"  {horizon} days: {forecast['point']:.1f} "
            f"[{forecast['lower']:.1f} - {forecast['upper']:.1f}] "
            f"({forecast['confidence_level']:.0%} confidence)"
        )
    
    logger.info("\n=== All Methods Comparison ===")
    for method, forecasts in forecast_result['all_methods'].items():
        logger.info(f"  {method}:")
        for horizon, value in forecasts.items():
            logger.info(f"    {horizon} days: {value:.1f}")
