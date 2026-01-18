"""
ASEWIS Streamlit Application

A two-page dashboard for Aadhar System Engineering & Workflow Intelligence:
1. NASRI Readiness Dashboard - Visual analytics and monitoring
2. Orchestration Control Panel - Intervention management and simulation

Design Principles:
- Load pre-computed data from Parquet (no heavy computation)
- Clear visual explanations with context
- Modular page structure
- Production-ready error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoring import compute_nasri, compute_asrs
from rules_engine import recommend_actions
from simulation import simulate_impact
from data_cleaning.location_cleaner import clean_location_columns
from geo_utils import (
    load_india_districts_geojson,
    build_district_lookup,
    match_district_name,
    normalize_district_name,
    get_unmatched_districts
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data_cache"
PROCESSED_DIR = Path(__file__).parent.parent / "dataset" / "processed"


# ============================================================================
# Data Loading Functions
# ============================================================================

@st.cache_data(ttl=300)
def load_aggregated_data() -> Optional[pd.DataFrame]:
    """
    Load pre-aggregated district-month data from Parquet cache.
    
    Returns:
        DataFrame with district-month metrics or None if not found
    """
    parquet_path = PROCESSED_DIR / "aggregated_metrics.parquet"
    
    if not parquet_path.exists():
        logger.error(f"Aggregated data not found at {parquet_path}")
        return None
    
    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} district-month records from cache")
        
        # Apply location cleaning as safety net
        pre_clean_rows = len(df)
        df = clean_location_columns(df)
        post_clean_rows = len(df)
        
        if pre_clean_rows != post_clean_rows:
            logger.warning(
                f"Location cleaning removed {pre_clean_rows - post_clean_rows} rows with invalid districts"
            )
        
        return df
    except Exception as e:
        logger.error(f"Error loading aggregated data: {e}")
        return None


@st.cache_data(ttl=300)
def load_features_data() -> Optional[pd.DataFrame]:
    """
    Load pre-computed features from Parquet cache.
    
    Returns:
        DataFrame with engineered features or None if not found
    """
    parquet_path = PROCESSED_DIR / "features.parquet"
    
    if not parquet_path.exists():
        logger.warning(f"Features data not found at {parquet_path}")
        return None
    
    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} feature records from cache")
        
        # Apply location cleaning as safety net
        pre_clean_rows = len(df)
        df = clean_location_columns(df)
        post_clean_rows = len(df)
        
        if pre_clean_rows != post_clean_rows:
            logger.warning(
                f"Location cleaning removed {pre_clean_rows - post_clean_rows} rows with invalid districts"
            )
        
        return df
    except Exception as e:
        logger.error(f"Error loading features data: {e}")
        return None


@st.cache_data(ttl=300)
def load_scores_data() -> Optional[pd.DataFrame]:
    """
    Load pre-computed NASRI/ASRS scores from Parquet cache.
    
    Returns:
        DataFrame with scores or None if not found
    """
    parquet_path = PROCESSED_DIR / "scores.parquet"
    
    if not parquet_path.exists():
        logger.warning(f"Scores data not found at {parquet_path}")
        return None
    
    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} score records from cache")
        
        # Apply location cleaning as safety net
        pre_clean_rows = len(df)
        df = clean_location_columns(df)
        post_clean_rows = len(df)
        
        if pre_clean_rows != post_clean_rows:
            logger.warning(
                f"Location cleaning removed {pre_clean_rows - post_clean_rows} rows with invalid districts"
            )
        
        return df
    except Exception as e:
        logger.error(f"Error loading scores data: {e}")
        return None


def get_latest_month_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract latest month data for each district.
    
    Args:
        df: DataFrame with date or month column
        
    Returns:
        DataFrame filtered to latest month per district
    """
    # Check for date column (preferred) or month column
    date_col = 'date' if 'date' in df.columns else 'month' if 'month' in df.columns else None
    
    if date_col is None:
        return df
    
    # Convert to datetime if string
    if df[date_col].dtype == 'object':
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Get latest month for each district
    latest_data = df.sort_values(date_col).groupby('district').tail(1)
    
    return latest_data


@st.cache_data(ttl=3600, show_spinner="Loading India district boundaries...")
def load_india_geojson_cached() -> Optional[Dict[str, Any]]:
    """
    Load and cache India district GeoJSON.
    
    Cached for 1 hour to avoid repeated downloads.
    
    Returns:
        GeoJSON dict with district boundaries
    """
    cache_dir = Path(__file__).parent.parent / "data_cache"
    geojson = load_india_districts_geojson(cache_dir)
    
    if geojson:
        logger.info(f"GeoJSON loaded successfully with {len(geojson.get('features', []))} features")
    else:
        logger.error("GeoJSON loading failed - returning None")
    
    return geojson


@st.cache_data(show_spinner=False, ttl=3600)
def load_prebuilt_map(metric_name: str):
    """
    ⚡ INSTANT LOAD: Load pre-built map from disk (no computation).
    If map doesn't exist, automatically generates it.
    
    Args:
        metric_name: 'ASRS' or 'NASRI'
    
    Returns:
        Tuple of (plotly figure, stats dict) or (None, None) if failed
    """
    import plotly.io as pio
    import json
    
    map_cache_dir = Path(__file__).parent.parent / "data_cache" / "maps"
    map_cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = map_cache_dir / f"{metric_name.lower()}_map.json"
    stats_path = map_cache_dir / "map_stats.json"
    
    # If pre-built map doesn't exist, generate it automatically
    if not json_path.exists():
        logger.info(f"Pre-built {metric_name} map not found. Generating automatically...")
        success = generate_prebuilt_maps()
        if not success:
            return None, None
    
    # Load map using plotly.io.read_json (handles binary format correctly)
    try:
        fig = pio.read_json(json_path)
    except Exception as e:
        logger.error(f"Error loading map JSON: {e}")
        return None, None
    
    # Load stats
    stats = {}
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            all_stats = json.load(f)
            stats = all_stats.get(metric_name, {})
    
    logger.info(f"⚡ Loaded pre-built {metric_name} map from disk (instant)")
    return fig, stats


def generate_prebuilt_maps():
    """
    Auto-generate pre-built map files if they don't exist.
    Called automatically when maps are missing.
    
    Returns:
        bool: True if successful, False otherwise
    """
    import json
    import plotly.express as px
    
    logger.info("🗺️ Auto-generating pre-built maps...")
    
    # Load GeoJSON
    cache_dir = Path(__file__).parent.parent / "data_cache"
    geojson_data = load_india_geojson_cached()
    if not geojson_data:
        logger.error("Failed to load GeoJSON - cannot generate maps")
        return False
    
    # Load scores
    scores_df = load_scores_data()
    if scores_df is None:
        logger.error("Failed to load scores - cannot generate maps")
        return False
    
    # Get latest month data
    latest_scores = get_latest_month_data(scores_df)
    
    # Build district lookup
    district_lookup = build_district_lookup(geojson_data.get('features', []))
    geojson_districts = set(district_lookup.keys())
    
    # Match districts
    matched_data = []
    for _, row in latest_scores.iterrows():
        district_name = row['district']
        matched_name = match_district_name(district_name, geojson_districts, verbose=False)
        if matched_name:
            matched_row = row.copy()
            matched_row['geojson_district'] = matched_name
            matched_data.append(matched_row)
    
    if not matched_data:
        logger.error("No districts matched - cannot generate maps")
        return False
    
    matched_df = pd.DataFrame(matched_data)
    match_rate = len(matched_df) / len(latest_scores) * 100
    logger.info(f"Matched {len(matched_df)}/{len(latest_scores)} districts ({match_rate:.1f}%)")
    
    # Update GeoJSON properties for both metrics
    for metric_col in ['asrs_score', 'nasri_score']:
        metric_name = 'ASRS' if metric_col == 'asrs_score' else 'NASRI'
        district_metric_map = matched_df.set_index('geojson_district')[metric_col].to_dict()
        
        for feature in geojson_data['features']:
            props = feature.get('properties', {})
            district_name = None
            for key in ['NAME_2', 'district', 'dtname', 'DISTRICT', 'NAME', 'name']:
                if key in props:
                    district_name = props[key]
                    break
            if district_name:
                normalized = normalize_district_name(district_name)
                props['normalized_district'] = normalized
    
    # Generate maps for both metrics
    map_cache_dir = cache_dir / "maps"
    map_cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    
    for metric_col, metric_name in [('asrs_score', 'ASRS'), ('nasri_score', 'NASRI')]:
        # Color scale
        if metric_name == 'NASRI':
            color_scale = [
                (0.0, '#D32F2F'), (0.25, '#F44336'), (0.5, '#FF9800'),
                (0.75, '#FFC107'), (1.0, '#4CAF50')
            ]
        else:
            color_scale = [
                (0.0, '#4CAF50'), (0.25, '#FFC107'), (0.5, '#FF9800'),
                (0.75, '#F44336'), (1.0, '#D32F2F')
            ]
        
        # Create choropleth
        fig = px.choropleth_mapbox(
            matched_df,
            geojson=geojson_data,
            locations='geojson_district',
            featureidkey='properties.normalized_district',
            color=metric_col,
            color_continuous_scale=color_scale,
            range_color=[matched_df[metric_col].min(), matched_df[metric_col].max()],
            mapbox_style="carto-positron",
            zoom=4,
            center={"lat": 23.5, "lon": 78.9},
            opacity=0.8,
            labels={metric_col: metric_name},
            hover_data={
                'geojson_district': False,
                'district': True,
                metric_col: ':.3f' if metric_name == 'ASRS' else ':.1f'
            }
        )
        
        fig.update_layout(
            height=750,
            margin=dict(l=0, r=0, t=30, b=0),
            title=None,
            showlegend=False,
            coloraxis_colorbar=dict(
                title=metric_name,
                thickness=12,
                len=0.6,
                x=1.02,
                bgcolor='rgba(255,255,255,0.9)',
                borderwidth=0
            )
        )
        
        # Save map
        json_path = map_cache_dir / f"{metric_name.lower()}_map.json"
        fig.write_json(json_path)
        logger.info(f"✅ Generated {metric_name} map: {json_path}")
        
        # Collect stats
        high_risk_count = len(matched_df[matched_df['asrs_score'] >= 0.5]) if 'asrs_risk_category' not in matched_df.columns else len(matched_df[matched_df['asrs_risk_category'].isin(['high', 'critical'])])
        stats[metric_name] = {
            'districts_mapped': len(matched_df),
            'match_rate': match_rate,
            'high_risk_count': high_risk_count,
            'avg_nasri': matched_df['nasri_score'].mean(),
            'avg_asrs': matched_df['asrs_score'].mean()
        }
    
    # Save stats
    stats_path = map_cache_dir / "map_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("✅ Pre-built maps generated successfully!")
    return True


def OLD_DEPRECATED_get_district_coordinates() -> Dict[str, tuple]:
    """
    Comprehensive Indian district coordinates for map visualization.
    Contains approximate centroids for 500+ districts across all states.
    
    Returns:
        dict: District name -> (lat, lon)
    """
    return {
        # Andhra Pradesh
        'ANANTAPUR': (14.6819, 77.6006), 'CHITTOOR': (13.2172, 79.1003), 'EAST GODAVARI': (17.2403, 81.7869),
        'GUNTUR': (16.3067, 80.4365), 'KRISHNA': (16.5193, 80.6305), 'KURNOOL': (15.8281, 78.0373),
        'PRAKASAM': (15.4991, 79.5616), 'SRIKAKULAM': (18.2949, 83.8938), 'VISAKHAPATNAM': (17.6868, 83.2185),
        'VIZIANAGARAM': (18.1167, 83.4000), 'WEST GODAVARI': (16.9891, 81.2809), 'YSR': (14.4673, 78.8242),
        
        # Arunachal Pradesh
        'TAWANG': (27.5861, 91.8586), 'WEST KAMENG': (27.2402, 92.3939), 'EAST KAMENG': (27.2746, 93.1090),
        'PAPUM PARE': (27.1130, 93.6158), 'LOWER SUBANSIRI': (27.6073, 93.8984), 'UPPER SUBANSIRI': (28.2742, 93.8830),
        'WEST SIANG': (28.3350, 94.5353), 'EAST SIANG': (28.0667, 95.0333), 'UPPER SIANG': (28.8833, 95.2500),
        
        # Assam
        'BAKSA': (26.7100, 91.2600), 'BARPETA': (26.3209, 91.0079), 'BONGAIGAON': (26.4832, 90.5579),
        'CACHAR': (24.8333, 92.7789), 'CHIRANG': (26.5387, 90.4525), 'DARRANG': (26.4545, 92.0290),
        'DHEMAJI': (27.4832, 94.5763), 'DHUBRI': (26.0172, 89.9812), 'DIBRUGARH': (27.4728, 94.9120),
        'DIMA HASAO': (25.5500, 93.0167), 'GOALPARA': (26.1667, 90.6167), 'GOLAGHAT': (26.5201, 93.9580),
        'GUWAHATI': (26.1445, 91.7362), 'HAILAKANDI': (24.6847, 92.5688), 'JORHAT': (26.7509, 94.2037),
        'KAMRUP': (26.0802, 91.5815), 'KARBI ANGLONG': (26.0019, 93.4417), 'KARIMGANJ': (24.8699, 92.3507),
        'KOKRAJHAR': (26.4018, 90.2717), 'LAKHIMPUR': (27.2333, 94.1000), 'MAJULI': (26.9544, 94.1630),
        'MORIGAON': (26.2527, 92.3438), 'NAGAON': (26.3484, 92.6856), 'NALBARI': (26.4462, 91.4346),
        'SIVASAGAR': (26.9845, 94.6378), 'SONITPUR': (26.6338, 92.7875), 'TINSUKIA': (27.4900, 95.3600),
        'UDALGURI': (26.7536, 92.1022),
        
        # Bihar
        'ARARIA': (26.1497, 87.5151), 'ARWAL': (25.2561, 84.6811), 'AURANGABAD': (24.7521, 84.3742),
        'AURANGABAD(BH)': (24.7521, 84.3742), 'BANKA': (24.8893, 86.9230), 'BEGUSARAI': (25.4182, 86.1272),
        'BHAGALPUR': (25.2425, 86.9842), 'BHOJPUR': (25.5626, 84.4597), 'BUXAR': (25.5649, 83.9777),
        'DARBHANGA': (26.1542, 85.8918), 'GAYA': (24.7955, 85.0002), 'GOPALGANJ': (26.4670, 84.4381),
        'JAMUI': (24.9275, 86.2248), 'JEHANABAD': (25.2072, 84.9880), 'KAIMUR': (25.0456, 83.6153),
        'KATIHAR': (25.5353, 87.5676), 'KHAGARIA': (25.5022, 86.4667), 'KISHANGANJ': (26.1058, 87.9506),
        'LAKHISARAI': (25.1730, 86.0938), 'MADHEPURA': (25.9211, 86.7939), 'MADHUBANI': (26.3538, 86.0746),
        'MUNGER': (25.3753, 86.4733), 'MUZAFFARPUR': (26.1225, 85.3906), 'NALANDA': (25.2083, 85.4608),
        'NAWADA': (24.8838, 85.5394), 'PATNA': (25.5941, 85.1376), 'PURNIA': (25.7771, 87.4753),
        'ROHTAS': (24.9563, 83.9928), 'SAHARSA': (25.8810, 86.5964), 'SAMASTIPUR': (25.8658, 85.7822),
        'SARAN': (25.8704, 84.7597), 'SHEIKHPURA': (25.1391, 85.8415), 'SHEOHAR': (26.5167, 85.2833),
        'SITAMARHI': (26.5958, 85.4826), 'SIWAN': (26.2190, 84.3538), 'SUPAUL': (26.1260, 86.6050),
        'VAISHALI': (25.9820, 85.1315), 'WEST CHAMPARAN': (27.0333, 84.5500),
        
        # Chhattisgarh
        'BALOD': (20.7307, 81.2061), 'BALODA BAZAR': (21.6575, 82.1605), 'BALRAMPUR': (23.1124, 83.0577),
        'BASTAR': (19.0688, 81.9528), 'BEMETARA': (21.7105, 81.5353), 'BIJAPUR': (18.8736, 81.0389),
        'BILASPUR': (22.0900, 82.1500), 'DANTEWADA': (18.8922, 81.3497), 'DHAMTARI': (20.7064, 81.5491),
        'DURG': (21.1901, 81.2849), 'GARIABAND': (20.6293, 82.0659), 'JANJGIR-CHAMPA': (22.0124, 82.5775),
        'JASHPUR': (22.8833, 84.1500), 'KABIRDHAM': (22.0964, 81.2396), 'KANKER': (20.2730, 81.4924),
        'KONDAGAON': (19.5909, 81.6639), 'KORBA': (22.3595, 82.6894), 'KORIYA': (23.2676, 82.3507),
        'MAHASAMUND': (21.1081, 82.0959), 'MUNGELI': (22.0656, 81.6855), 'NARAYANPUR': (19.2149, 81.0061),
        'RAIGARH': (21.8974, 83.3950), 'RAIPUR': (21.2514, 81.6296), 'RAJNANDGAON': (21.0978, 81.0371),
        'SUKMA': (18.3859, 81.6635), 'SURAJPUR': (23.2234, 82.8724), 'SURGUJA': (23.1102, 83.1976),
        # Additional Chhattisgarh name variants
        'MANENDRAGARHCHIRMIRIBHARATPUR': (23.2234, 82.8724), 'MANENDRAGARH-CHIRMIRI-BHARATPUR': (23.2234, 82.8724),
        'MANENDRAGARH CHIRMIRI BHARATPUR': (23.2234, 82.8724),
        
        # Delhi
        'CENTRAL DELHI': (28.6562, 77.2410), 'EAST DELHI': (28.6517, 77.3050), 'NEW DELHI': (28.6139, 77.2090),
        'NORTH DELHI': (28.7167, 77.2000), 'NORTH EAST DELHI': (28.7167, 77.2833), 'NORTH WEST DELHI': (28.7167, 77.1000),
        'SHAHDARA': (28.6753, 77.2885), 'SOUTH DELHI': (28.5244, 77.2066), 'SOUTH WEST DELHI': (28.6050, 77.0000),
        'WEST DELHI': (28.6500, 77.1000), 'DELHI': (28.6139, 77.2090),
        
        # Gujarat
        'AHMADABAD': (23.0225, 72.5714), 'AHMADNAGAR': (23.0225, 72.5714), 'AMRELI': (21.6005, 71.2184),
        'ANAND': (22.5645, 72.9289), 'ARAVALLI': (23.2500, 73.1500), 'BANASKANTHA': (24.1719, 72.4388),
        'BHARUCH': (21.7051, 72.9959), 'BHAVNAGAR': (21.7645, 72.1519), 'BOTAD': (22.1693, 71.6672),
        'CHHOTA UDAIPUR': (22.3050, 74.0103), 'DAHOD': (22.8387, 74.2542), 'DANG': (20.7476, 73.7089),
        'DEVBHOOMI DWARKA': (22.2394, 69.0908), 'GANDHINAGAR': (23.2156, 72.6369), 'GIR SOMNATH': (20.8973, 70.4005),
        'JAMNAGAR': (22.4707, 70.0577), 'JUNAGADH': (21.5222, 70.4579), 'KHEDA': (22.7507, 72.6834),
        'KUTCH': (23.7337, 69.8597), 'MAHISAGAR': (23.1100, 73.6000), 'MEHSANA': (23.5880, 72.3693),
        'MORBI': (22.8172, 70.8372), 'NARMADA': (21.8718, 73.5024), 'NAVSARI': (20.8507, 72.9236),
        'PANCHMAHAL': (22.8291, 73.6008), 'PATAN': (23.8514, 72.1228), 'PORBANDAR': (21.6417, 69.6293),
        'RAJKOT': (22.3039, 70.8022), 'SABARKANTHA': (23.5450, 73.0500), 'SURAT': (21.1702, 72.8311),
        'SURENDRANAGAR': (22.7039, 71.6369), 'TAPI': (21.1333, 73.4167), 'VADODARA': (22.3072, 73.1812),
        'VALSAD': (20.6337, 72.9342),
        
        # Haryana
        'AMBALA': (30.3782, 76.7767), 'BHIWANI': (28.7930, 76.1395), 'CHARKHI DADRI': (28.5917, 76.2711),
        'FARIDABAD': (28.4089, 77.3178), 'FATEHABAD': (29.5151, 75.4539), 'GURUGRAM': (28.4595, 77.0266),
        'GURGAON': (28.4595, 77.0266), 'HISAR': (29.1492, 75.7217), 'JHAJJAR': (28.6063, 76.6565),
        'JIND': (29.3156, 76.3159), 'KAITHAL': (29.8013, 76.3999), 'KARNAL': (29.6857, 76.9905),
        'KURUKSHETRA': (29.9695, 76.8783), 'MAHENDRAGARH': (28.2832, 76.1498), 'MEWAT': (27.7833, 77.0167),
        'NUHH': (28.1033, 77.0058), 'PALWAL': (28.1442, 77.3264), 'PANCHKULA': (30.6942, 76.8534),
        'PANIPAT': (29.3909, 76.9635), 'REWARI': (28.1989, 76.6194), 'ROHTAK': (28.8955, 76.5897),
        'SIRSA': (29.5353, 75.0289), 'SONIPAT': (28.9931, 77.0151), 'YAMUNANAGAR': (30.1290, 77.2674),
        
        # Himachal Pradesh
        'BILASPUR': (31.3390, 76.7568), 'CHAMBA': (32.5562, 76.1262), 'HAMIRPUR': (31.6838, 76.5179),
        'KANGRA': (32.0998, 76.2691), 'KINNAUR': (31.5833, 78.2333), 'KULLU': (31.9576, 77.1093),
        'LAHAUL AND SPITI': (32.5597, 77.4278), 'MANDI': (31.7077, 76.9318), 'SHIMLA': (31.1048, 77.1734),
        'SIRMAUR': (30.5621, 77.2960), 'SOLAN': (30.9045, 77.0967), 'UNA': (31.4649, 76.2708),
        
        # Jharkhand
        'BOKARO': (23.7871, 85.9565), 'CHATRA': (24.2074, 84.8706), 'DEOGHAR': (24.4833, 86.7000),
        'DHANBAD': (23.7957, 86.4304), 'DUMKA': (24.2667, 87.2500), 'EAST SINGHBHUM': (22.8046, 86.2029),
        'GARHWA': (24.1667, 83.8000), 'GIRIDIH': (24.1894, 86.3054), 'GODDA': (24.8333, 87.2167),
        'GUMLA': (23.0445, 84.5388), 'HAZARIBAGH': (23.9929, 85.3605), 'JAMTARA': (23.9631, 86.8021),
        'KHUNTI': (23.0715, 85.2789), 'KODERMA': (24.4677, 85.5956), 'LATEHAR': (23.7441, 84.4999),
        'LOHARDAGA': (23.4333, 84.6833), 'PAKUR': (24.6333, 87.8500), 'PALAMU': (24.0334, 84.0667),
        'RAMGARH': (23.6323, 85.5191), 'RANCHI': (23.3441, 85.3096), 'SAHEBGANJ': (25.2500, 87.6167),
        'SERAIKELA-KHARSAWAN': (22.6979, 85.9498), 'SIMDEGA': (22.6189, 84.5012), 'WEST SINGHBHUM': (22.5645, 85.3800),
        
        # Karnataka
        'BAGALKOT': (16.1691, 75.6962), 'BALLARI': (15.1394, 76.9214), 'BELAGAVI': (15.8497, 74.4977),
        'BELGAUM': (15.8497, 74.4977), 'BENGALURU RURAL': (13.2257, 77.3657), 'BENGALURU URBAN': (12.9716, 77.5946),
        'BANGALORE': (12.9716, 77.5946), 'BIDAR': (17.9130, 77.5302), 'CHAMARAJANAGAR': (11.9256, 76.9393),
        'CHIKBALLAPUR': (13.4355, 77.7315), 'CHIKKAMAGALURU': (13.3161, 75.7720), 'CHITRADURGA': (14.2226, 76.4020),
        'DAKSHINA KANNADA': (12.8438, 75.2479), 'DAVANAGERE': (14.4644, 75.9218), 'DHARWAD': (15.4589, 75.0078),
        'GADAG': (15.4315, 75.6193), 'HASSAN': (13.0072, 76.0962), 'HAVERI': (14.7951, 75.4005),
        'KALABURAGI': (17.3297, 76.8343), 'KODAGU': (12.3375, 75.8069), 'KOLAR': (13.1358, 78.1297),
        'KOPPAL': (15.3500, 76.1500), 'MANDYA': (12.5218, 76.8951), 'MYSURU': (12.2958, 76.6394),
        'MYSORE': (12.2958, 76.6394), 'RAICHUR': (16.2072, 77.3566), 'RAMANAGARA': (12.7173, 77.2806),
        'SHIVAMOGGA': (13.9299, 75.5681), 'TUMAKURU': (13.3392, 77.1006), 'UDUPI': (13.3409, 74.7421),
        'UTTARA KANNADA': (14.5237, 74.6869), 'VIJAYAPURA': (16.8302, 75.7100), 'VIJAYANAGARA': (15.1704, 76.3788),
        'YADGIR': (16.7700, 77.1383),
        
        # Kerala
        'ALAPPUZHA': (9.4981, 76.3388), 'ERNAKULAM': (9.9816, 76.2999), 'IDUKKI': (9.9189, 77.1025),
        'KANNUR': (11.8745, 75.3704), 'KASARAGOD': (12.4996, 74.9869), 'KOLLAM': (8.8909, 76.6144),
        'KOTTAYAM': (9.5916, 76.5222), 'KOZHIKODE': (11.2588, 75.7804), 'MALAPPURAM': (11.0510, 76.0711),
        'PALAKKAD': (10.7867, 76.6548), 'PATHANAMTHITTA': (9.2648, 76.7870), 'THIRUVANANTHAPURAM': (8.5241, 76.9366),
        'THRISSUR': (10.5276, 76.2144), 'WAYANAD': (11.6854, 76.1320), 'KOCHI': (9.9312, 76.2673),
        
        # Madhya Pradesh
        'AGAR MALWA': (23.7121, 76.0159), 'ALIRAJPUR': (22.3100, 74.3601), 'ANUPPUR': (23.1057, 81.6897),
        'ASHOKNAGAR': (24.5810, 77.7299), 'BALAGHAT': (21.8054, 80.1870), 'BARWANI': (22.0326, 74.9020),
        'BETUL': (21.9045, 77.8988), 'BHIND': (26.5645, 78.7802), 'BHOPAL': (23.2599, 77.4126),
        'BURHANPUR': (21.3086, 76.2309), 'CHHATARPUR': (24.9108, 79.5884), 'CHHINDWARA': (22.0576, 78.9382),
        'DAMOH': (23.8315, 79.4421), 'DATIA': (25.6651, 78.4631), 'DEWAS': (22.9659, 76.0534),
        'DHAR': (22.5979, 75.2970), 'DINDORI': (22.9418, 81.0791), 'GUNA': (24.6469, 77.3119),
        'GWALIOR': (26.2183, 78.1828), 'HARDA': (22.3442, 77.0954), 'HOSHANGABAD': (22.7542, 77.7266),
        'INDORE': (22.7196, 75.8577), 'JABALPUR': (23.1815, 79.9864), 'JHABUA': (22.7676, 74.5910),
        'KATNI': (23.8346, 80.3950), 'KHANDWA': (21.8333, 76.3500), 'KHARGONE': (21.8234, 75.6147),
        'MANDLA': (22.5996, 80.3714), 'MANDSAUR': (24.0736, 75.0722), 'MORENA': (26.4989, 78.0014),
        'NARSINGHPUR': (22.9493, 79.1926), 'NEEMUCH': (24.4708, 74.8694), 'PANNA': (24.7166, 80.1944),
        'RAISEN': (23.3315, 77.7833), 'RAJGARH': (24.0073, 76.7292), 'RATLAM': (23.3315, 75.0367),
        'REWA': (24.5364, 81.2961), 'SAGAR': (23.8388, 78.7378), 'SATNA': (24.5924, 80.8200),
        'SEHORE': (23.2016, 77.0832), 'SEONI': (22.0853, 79.5508), 'SHAHDOL': (23.2965, 81.3612),
        'SHAJAPUR': (23.4273, 76.2732), 'SHEOPUR': (25.6681, 76.6966), 'SHIVPURI': (25.4231, 77.6613),
        'SIDHI': (24.4166, 81.8874), 'SINGRAULI': (24.1997, 82.6753), 'TIKAMGARH': (24.7433, 78.8298),
        'UJJAIN': (23.1765, 75.7885), 'UMARIA': (23.5247, 80.8372), 'VIDISHA': (23.5251, 77.8081),
        
        # Maharashtra
        'AHMADNAGAR': (19.0948, 74.7480), 'AHILYANAGAR': (19.0948, 74.7480), 'AKOLA': (20.7002, 77.0082),
        'AMRAVATI': (20.9374, 77.7796), 'AURANGABAD': (19.8762, 75.3433), 'BEED': (18.9892, 75.7613),
        'BHANDARA': (21.1703, 79.6521), 'BULDHANA': (20.5315, 76.1843), 'CHANDRAPUR': (19.9615, 79.2961),
        'DHULE': (20.9042, 74.7749), 'GADCHIROLI': (20.1809, 80.0066), 'GONDIA': (21.4560, 80.1995),
        'HINGOLI': (19.7154, 77.1544), 'JALGAON': (21.0077, 75.5626), 'JALNA': (19.8348, 75.8800),
        'KOLHAPUR': (16.7050, 74.2433), 'LATUR': (18.3984, 76.5604), 'MUMBAI': (19.0760, 72.8777),
        'MUMBAI SUBURBAN': (19.1136, 72.9083), 'NAGPUR': (21.1458, 79.0882), 'NANDED': (19.1383, 77.3210),
        'NANDURBAR': (21.3669, 74.2443), 'NASHIK': (19.9975, 73.7898), 'OSMANABAD': (18.1757, 76.0407),
        'PALGHAR': (19.6966, 72.7655), 'PARBHANI': (19.2704, 76.7695), 'PUNE': (18.5204, 73.8567),
        'RAIGAD': (18.3557, 73.1641), 'RATNAGIRI': (16.9944, 73.3000), 'SANGLI': (16.8524, 74.5815),
        'SATARA': (17.6805, 73.9903), 'SINDHUDURG': (16.0000, 73.6667), 'SOLAPUR': (17.6599, 75.9064),
        'THANE': (19.2183, 72.9781), 'WARDHA': (20.7453, 78.5970), 'WASHIM': (20.1106, 77.1367),
        'YAVATMAL': (20.3897, 78.1307),
        
        # Manipur
        'BISHNUPUR': (24.6160, 93.7783), 'CHANDEL': (24.3167, 94.0000), 'CHURACHANDPUR': (24.3333, 93.6833),
        'IMPHAL EAST': (24.7833, 93.9667), 'IMPHAL WEST': (24.6833, 93.9500), 'JIRIBAM': (24.8000, 93.1167),
        'KAKCHING': (24.4980, 93.9815), 'KAMJONG': (24.8000, 94.2500), 'KANGPOKPI': (25.0500, 93.9833),
        'NONEY': (24.7667, 93.5000), 'PHERZAWL': (24.1667, 93.0833), 'SENAPATI': (25.2667, 94.0167),
        'TAMENGLONG': (24.9833, 93.5000), 'TENGNOUPAL': (24.3167, 94.1167), 'THOUBAL': (24.6333, 94.0167),
        'UKHRUL': (25.0667, 94.3500),
        
        # Meghalaya
        'EAST GARO HILLS': (25.5000, 90.6300), 'EAST JAINTIA HILLS': (25.4667, 92.3500), 'EAST KHASI HILLS': (25.4667, 91.8833),
        'NORTH GARO HILLS': (25.8000, 90.7667), 'RI BHOI': (25.7833, 91.8500), 'SOUTH GARO HILLS': (25.2000, 90.5000),
        'SOUTH WEST GARO HILLS': (25.4333, 90.0167), 'SOUTH WEST KHASI HILLS': (25.3000, 91.2667), 'WEST GARO HILLS': (25.5667, 90.2167),
        'WEST JAINTIA HILLS': (25.4500, 92.2000), 'WEST KHASI HILLS': (25.5333, 91.2833),
        
        # Mizoram
        'AIZAWL': (23.7307, 92.7173), 'CHAMPHAI': (23.4500, 93.3167), 'KOLASIB': (24.2167, 92.6833),
        'LAWNGTLAI': (22.5333, 92.8833), 'LUNGLEI': (22.8833, 92.7333), 'MAMIT': (23.9333, 92.4833),
        'SAIHA': (22.4833, 92.9833), 'SERCHHIP': (23.3000, 92.8333),
        
        # Nagaland
        'DIMAPUR': (25.9042, 93.7275), 'KIPHIRE': (25.8333, 94.8167), 'KOHIMA': (25.6701, 94.1077),
        'LONGLENG': (26.5333, 94.5833), 'MOKOKCHUNG': (26.3167, 94.5167), 'MON': (26.7167, 95.0000),
        'PEREN': (25.5000, 93.7333), 'PHEK': (25.6667, 94.5000), 'TUENSANG': (26.2667, 94.8167),
        'WOKHA': (26.0833, 94.2667), 'ZUNHEBOTO': (25.9667, 94.5167),
        
        # Odisha
        'ANGUL': (20.8400, 85.1018), 'ANUGUL': (20.8400, 85.1018), 'BALANGIR': (20.7099, 83.4879),
        'BALASORE': (21.4934, 86.9336), 'BARGARH': (21.3344, 83.6190), 'BHADRAK': (21.0542, 86.4913),
        'BOUDH': (20.8333, 84.3333), 'CUTTACK': (20.5124, 85.8829), 'DEOGARH': (21.5333, 84.7333),
        'DHENKANAL': (20.6667, 85.6000), 'GAJAPATI': (19.3500, 84.1167), 'GANJAM': (19.3858, 84.8000),
        'JAGATSINGHPUR': (20.2628, 86.1711), 'JAJPUR': (20.8500, 86.3333), 'JHARSUGUDA': (21.8535, 84.0070),
        'KALAHANDI': (19.9139, 83.1656), 'KANDHAMAL': (20.1667, 84.0833), 'KENDRAPARA': (20.4986, 86.4248),
        'KENDUJHAR': (21.6297, 85.5888), 'KHORDHA': (20.1826, 85.6187), 'KORAPUT': (18.8104, 82.7110),
        'MALKANGIRI': (18.3500, 81.8833), 'MAYURBHANJ': (21.9333, 86.7333), 'NABARANGPUR': (19.2311, 82.5431),
        'NAYAGARH': (20.1333, 85.0833), 'NUAPADA': (20.8000, 82.5333), 'PURI': (19.8135, 85.8312),
        'RAYAGADA': (19.1750, 83.4167), 'SAMBALPUR': (21.4669, 83.9812), 'SONEPUR': (20.8333, 83.9167),
        'SUNDARGARH': (22.1167, 84.0333),
        
        # Punjab
        'AMRITSAR': (31.6340, 74.8723), 'BARNALA': (30.3788, 75.5488), 'BATHINDA': (30.2110, 74.9455),
        'FARIDKOT': (30.6705, 74.7573), 'FATEHGARH SAHIB': (30.6445, 76.3948), 'FAZILKA': (30.4028, 74.0286),
        'FEROZEPUR': (30.9180, 74.6132), 'GURDASPUR': (32.0396, 75.4058), 'HOSHIARPUR': (31.5332, 75.9119),
        'JALANDHAR': (31.3260, 75.5762), 'KAPURTHALA': (31.3800, 75.3802), 'LUDHIANA': (30.9010, 75.8573),
        'MANSA': (29.9988, 75.3932), 'MOGA': (30.8152, 75.1697), 'MOHALI': (30.7046, 76.7179),
        'MUKTSAR': (30.4761, 74.5162), 'PATHANKOT': (32.2743, 75.6527), 'PATIALA': (30.3398, 76.3869),
        'RUPNAGAR': (30.9699, 76.5316), 'SANGRUR': (30.2447, 75.8417), 'SHAHEED BHAGAT SINGH NAGAR': (31.1048, 76.1183),
        'TARN TARAN': (31.4519, 74.9278),
        
        # Rajasthan
        'AJMER': (26.4499, 74.6399), 'ALWAR': (27.5530, 76.6346), 'BANSWARA': (23.5411, 74.4416),
        'BARAN': (25.1000, 76.5167), 'BARMER': (25.7521, 71.3962), 'BHARATPUR': (27.2152, 77.4897),
        'BHILWARA': (25.3467, 74.6406), 'BIKANER': (28.0229, 73.3119), 'BUNDI': (25.4305, 75.6499),
        'CHITTORGARH': (24.8887, 74.6269), 'CHURU': (28.2972, 74.9647), 'DAUSA': (26.8942, 76.5619),
        'DHOLPUR': (26.7009, 77.8937), 'DUNGARPUR': (23.8420, 73.7147), 'GANGANAGAR': (29.9038, 73.8772),
        'HANUMANGARH': (29.5819, 74.3220), 'JAIPUR': (26.9124, 75.7873), 'JAISALMER': (26.9157, 70.9083),
        'JALORE': (25.3458, 72.6156), 'JHALAWAR': (24.5979, 76.1613), 'JHUNJHUNU': (28.1308, 75.3982),
        'JODHPUR': (26.2389, 73.0243), 'KARAULI': (26.5025, 77.0152), 'KOTA': (25.2138, 75.8648),
        'NAGAUR': (27.2023, 73.7340), 'PALI': (25.7711, 73.3234), 'PRATAPGARH': (24.0311, 74.7789),
        'RAJSAMAND': (25.0715, 73.8803), 'SAWAI MADHOPUR': (26.0173, 76.3527), 'SIKAR': (27.6119, 75.1397),
        'SIROHI': (24.8867, 72.8581), 'TONK': (26.1544, 75.7860), 'UDAIPUR': (24.5854, 73.7125),
        
        # Tamil Nadu
        'ARIYALUR': (11.1401, 79.0766), 'CHENGALPATTU': (12.6918, 79.9763), 'CHENNAI': (13.0827, 80.2707),
        'COIMBATORE': (11.0168, 76.9558), 'CUDDALORE': (11.7480, 79.7714), 'DHARMAPURI': (12.1211, 78.1582),
        'DINDIGUL': (10.3624, 77.9694), 'ERODE': (11.3410, 77.7172), 'KALLAKURICHI': (11.7390, 78.9597),
        'KANCHIPURAM': (12.8185, 79.7047), 'KANYAKUMARI': (8.0883, 77.5385), 'KARUR': (10.9601, 78.0766),
        'KRISHNAGIRI': (12.5266, 78.2150), 'MADURAI': (9.9252, 78.1198), 'NAGAPATTINAM': (10.7667, 79.8448),
        'NAMAKKAL': (11.2189, 78.1677), 'NILGIRIS': (11.4102, 76.6950), 'PERAMBALUR': (11.2321, 78.8796),
        'PUDUKKOTTAI': (10.3833, 78.8209), 'RAMANATHAPURAM': (9.3639, 78.8377), 'RANIPET': (12.9243, 79.3335),
        'SALEM': (11.6643, 78.1460), 'SIVAGANGA': (9.8433, 78.4809), 'TENKASI': (8.9597, 77.3152),
        'THANJAVUR': (10.7870, 79.1378), 'THENI': (10.0104, 77.4977), 'THOOTHUKUDI': (8.7642, 78.1348),
        'TIRUCHIRAPPALLI': (10.7905, 78.7047), 'TIRUNELVELI': (8.7139, 77.7567), 'TIRUPATHUR': (12.4986, 78.5733),
        'TIRUPPUR': (11.1085, 77.3411), 'TIRUVALLUR': (13.1438, 79.9079), 'TIRUVANNAMALAI': (12.2253, 79.0747),
        'TIRUVARUR': (10.7725, 79.6345), 'VELLORE': (12.9165, 79.1325), 'VILUPPURAM': (11.9401, 79.4861),
        'VILLUPURAM': (11.9401, 79.4861), 'VIRUDHUNAGAR': (9.5681, 77.9624),
        
        # Telangana
        'ADILABAD': (19.6637, 78.5311), 'BHADRADRI KOTHAGUDEM': (17.5509, 80.6145), 'HYDERABAD': (17.3850, 78.4867),
        'JAGTIAL': (18.7937, 78.9177), 'JANGAON': (17.7244, 79.1514), 'JAYASHANKAR': (18.3333, 80.1500),
        'JOGULAMBA': (16.4667, 78.2833), 'KAMAREDDY': (18.3220, 78.3385), 'KARIMNAGAR': (18.4386, 79.1288),
        'KHAMMAM': (17.2473, 80.1514), 'KUMURAM BHEEM': (18.8333, 79.5833), 'MAHABUBABAD': (17.5981, 80.0017),
        'MAHBUBNAGAR': (16.7488, 77.9963), 'MANCHERIAL': (18.8696, 79.4738), 'MEDAK': (18.0499, 78.2646),
        'MEDCHAL': (17.6253, 78.4813), 'MULUGU': (18.1906, 79.9472), 'NAGARKURNOOL': (16.4892, 78.3119),
        'NALGONDA': (17.0577, 79.2604), 'NARAYANPET': (16.7452, 77.4932), 'NIRMAL': (19.0969, 78.3428),
        'NIZAMABAD': (18.6725, 78.0938), 'PEDDAPALLI': (18.6124, 79.3763), 'RAJANNA SIRCILLA': (18.3920, 78.8147),
        'RANGAREDDY': (17.3061, 78.2267), 'SANGAREDDY': (17.6144, 78.0833), 'SIDDIPET': (18.1018, 78.8529),
        'SURYAPET': (17.1414, 79.6237), 'VIKARABAD': (17.3387, 77.9042), 'WANAPARTHY': (16.3671, 78.0672),
        'WARANGAL RURAL': (17.9784, 79.5941), 'WARANGAL URBAN': (17.9689, 79.5941), 'YADADRI BHUVANAGIRI': (17.4500, 78.9000),
        
        # Tripura
        'DHALAI': (23.8333, 91.9167), 'GOMATI': (23.5167, 91.4833), 'KHOWAI': (24.0667, 91.6000),
        'NORTH TRIPURA': (24.1333, 92.1667), 'SEPAHIJALA': (23.7333, 91.4000), 'SOUTH TRIPURA': (23.1667, 91.7833),
        'UNAKOTI': (24.3167, 92.0000), 'WEST TRIPURA': (23.8333, 91.2833),
        
        # Uttar Pradesh
        'AGRA': (27.1767, 78.0081), 'ALIGARH': (27.8974, 78.0880), 'AMBEDKAR NAGAR': (26.4045, 82.6979),
        'AMETHI': (26.1542, 81.8079), 'AMROHA': (28.9037, 78.4671), 'AURAIYA': (26.4656, 79.5132),
        'AYODHYA': (26.7922, 82.1998), 'AZAMGARH': (26.0686, 83.1840), 'BADAUN': (28.0352, 79.1208),
        'BAGHPAT': (28.9473, 77.2240), 'BAHRAICH': (27.5742, 81.5947), 'BALLIA': (25.7615, 84.1499),
        'BALRAMPUR': (27.4308, 82.1818), 'BANDA': (25.4764, 80.3364), 'BARABANKI': (26.9242, 81.1862),
        'BAREILLY': (28.3670, 79.4304), 'BASTI': (26.8039, 82.7392), 'BIJNOR': (29.3731, 78.1364),
        'BULANDSHAHR': (28.4067, 77.8498), 'CHANDAULI': (25.2667, 83.2667), 'CHITRAKOOT': (25.2000, 80.9000),
        'DEORIA': (26.5024, 83.7791), 'ETAH': (27.5557, 78.6650), 'ETAWAH': (26.7855, 79.0215),
        'FARRUKHABAD': (27.3882, 79.5801), 'FATEHPUR': (25.9302, 80.8130), 'FIROZABAD': (27.1501, 78.3957),
        'GAUTAM BUDDHA NAGAR': (28.4519, 77.5340), 'GHAZIABAD': (28.6692, 77.4538), 'GHAZIPUR': (25.5800, 83.5800),
        'GONDA': (27.1333, 81.9600), 'GORAKHPUR': (26.7606, 83.3732), 'HAMIRPUR': (25.9562, 80.1482),
        'HAPUR': (28.7306, 77.7608), 'HARDOI': (27.3965, 80.1282), 'HATHRAS': (27.5926, 78.0492),
        'JALAUN': (26.1449, 79.3362), 'JAUNPUR': (25.7500, 82.6833), 'JHANSI': (25.4484, 78.5685),
        'KANNAUJ': (27.0514, 79.9196), 'KANPUR DEHAT': (26.4615, 79.8549), 'KANPUR NAGAR': (26.4499, 80.3319),
        'KANPUR': (26.4499, 80.3319), 'KASGANJ': (27.8094, 78.6431), 'KAUSHAMBI': (25.5316, 81.3784),
        'KHERI': (27.9046, 80.7799), 'KUSHINAGAR': (26.7411, 83.8883), 'LALITPUR': (24.6901, 78.4136),
        'LUCKNOW': (26.8467, 80.9462), 'MAHARAJGANJ': (27.1439, 83.5615), 'MAHOBA': (25.2920, 79.8730),
        'MAINPURI': (27.2351, 79.0250), 'MATHURA': (27.4924, 77.6737), 'MAU': (25.9417, 83.5611),
        'MEERUT': (28.9845, 77.7064), 'MIRZAPUR': (25.1462, 82.5650), 'MORADABAD': (28.8389, 78.7767),
        'MUZAFFARNAGAR': (29.4727, 77.7085), 'PILIBHIT': (28.6303, 79.8046), 'PRATAPGARH': (25.8931, 81.9378),
        'PRAYAGRAJ': (25.4358, 81.8463), 'RAE BARELI': (26.2124, 81.2459), 'RAMPUR': (28.8103, 79.0250),
        'SAHARANPUR': (29.9680, 77.5460), 'SAMBHAL': (28.5855, 78.5703), 'SANT KABIR NAGAR': (26.7667, 83.0333),
        'SHAHJAHANPUR': (27.8831, 79.9056), 'SHAMLI': (29.4500, 77.3167), 'SHRAVASTI': (27.5167, 82.0500),
        'SIDDHARTHNAGAR': (27.2519, 83.0661), 'SITAPUR': (27.5667, 80.6833), 'SONBHADRA': (24.6913, 83.0696),
        'SULTANPUR': (26.2646, 82.0736), 'UNNAO': (26.5464, 80.4880), 'VARANASI': (25.3176, 82.9739),
        
        # Uttarakhand
        'ALMORA': (29.5971, 79.6591), 'BAGESHWAR': (29.8390, 79.7710), 'CHAMOLI': (30.4000, 79.3300),
        'CHAMPAWAT': (29.3372, 80.0923), 'DEHRADUN': (30.3165, 78.0322), 'HARIDWAR': (29.9457, 78.1642),
        'NAINITAL': (29.3803, 79.4636), 'PAURI GARHWAL': (30.1533, 78.7812), 'PITHORAGARH': (29.5833, 80.2167),
        'RUDRAPRAYAG': (30.2841, 78.9811), 'TEHRI GARHWAL': (30.3901, 78.4803), 'UDHAM SINGH NAGAR': (28.9753, 79.4017),
        'UTTARKASHI': (30.7268, 78.4354),
        
        # West Bengal
        '24 PARAGANAS NORTH': (22.6157, 88.4332), '24 PARAGANAS SOUTH': (22.1626, 88.4297),
        'ALIPURDUAR': (26.4916, 89.5272), 'BANKURA': (23.2324, 87.0696), 'BIRBHUM': (24.0354, 87.6195),
        'COOCH BEHAR': (26.3157, 89.4485), 'DAKSHIN DINAJPUR': (25.2194, 88.7667), 'DARJEELING': (27.0410, 88.2663),
        'HOOGHLY': (22.9089, 88.3967), 'HOWRAH': (22.5958, 88.2636), 'JALPAIGURI': (26.5167, 88.7181),
        'JHARGRAM': (22.4539, 86.9880), 'KALIMPONG': (27.0644, 88.4753), 'KOLKATA': (22.5726, 88.3639),
        'MALDA': (25.0096, 88.1410), 'MURSHIDABAD': (24.1752, 88.2803), 'NADIA': (23.4712, 88.5567),
        'NORTH 24 PARGANAS': (22.6157, 88.4332), 'PASCHIM BARDHAMAN': (23.5522, 87.2847), 'PASCHIM MEDINIPUR': (22.4291, 87.3219),
        'PURBA BARDHAMAN': (23.2324, 87.8615), 'PURBA MEDINIPUR': (21.9513, 87.7385), 'PURULIA': (23.3421, 86.3660),
        'SOUTH 24 PARGANAS': (22.1626, 88.4297), 'UTTAR DINAJPUR': (26.3157, 88.1184),
        # Additional West Bengal district name variants
        'NORTH TWENTY FOUR PARGANAS': (22.6157, 88.4332), 'SOUTH TWENTY FOUR PARGANAS': (22.1626, 88.4297),
        'PARAGANAS NORTH': (22.6157, 88.4332), 'PARAGANAS SOUTH': (22.1626, 88.4297),
        'MEDINIPUR WEST': (22.4291, 87.3219), 'MEDINIPUR EAST': (21.9513, 87.7385),
        'WEST MEDINIPUR': (22.4291, 87.3219), 'EAST MEDINIPUR': (21.9513, 87.7385),
        'BARDDHAMAN': (23.2324, 87.8615), 'BARDHAMAN': (23.2324, 87.8615),
        
        # Other territories
        'CHANDIGARH': (30.7333, 76.7794), 'DADRA AND NAGAR HAVELI': (20.1809, 73.0169), 'DAMAN': (20.4140, 72.8328),
        'LAKSHADWEEP': (10.5667, 72.6417), 'PUDUCHERRY': (11.9139, 79.8145), 'YANAM': (16.7333, 82.2167),
        'PONDICHERRY': (11.9139, 79.8145), 'KARAIKAL': (10.9254, 79.8380), 'MAHE': (11.7014, 75.5347),
        
        # Jammu & Kashmir and Ladakh
        'ANANTNAG': (33.7311, 75.1486), 'BADGAM': (34.0167, 74.7333), 'BANDIPORA': (34.4167, 74.6500),
        'BARAMULLA': (34.2090, 74.3434), 'DODA': (33.1489, 75.5467), 'GANDERBAL': (34.2167, 74.7833),
        'JAMMU': (32.7266, 74.8570), 'KATHUA': (32.3691, 75.5158), 'KISHTWAR': (33.3119, 75.7681),
        'KULGAM': (33.6400, 75.0189), 'KUPWARA': (34.5267, 74.2567), 'POONCH': (33.7700, 74.0936),
        'PULWAMA': (33.8712, 74.8936), 'RAJOURI': (33.3792, 74.3125), 'RAMBAN': (33.2433, 75.1933),
        'REASI': (33.0833, 74.8333), 'SAMBA': (32.5625, 75.1194), 'SHOPIAN': (33.7081, 74.8306),
        'SRINAGAR': (34.0837, 74.7973), 'UDHAMPUR': (32.9344, 75.1417), 'KARGIL': (34.5539, 76.1311),
        'LEH': (34.1526, 77.5770),
        
        # Additional districts
        'KHAIRTHAL-TIJARA': (27.7667, 76.6000), 'SAITUAL': (23.4833, 92.9667),
        'BALOTRA': (25.8322, 72.2389), 'BEAWAR': (26.1019, 74.3203),
    }


# ============================================================================
# Page 1: NASRI Readiness Dashboard
# ============================================================================

def render_nasri_dashboard():
    """
    NASRI Readiness Dashboard Page
    
    Displays:
    - National summary metrics
    - District-level NASRI scores
    - Risk distribution (ASRS)
    - Trend visualizations
    - Geographic heatmap
    """
    st.title("🎯 NASRI Readiness Dashboard")
    st.markdown("**National Aadhar System Readiness Index** - Real-time monitoring and analytics")
    
    # Load data
    scores_df = load_scores_data()
    features_df = load_features_data()
    
    if scores_df is None:
        st.error("❌ Score data not found. Please run the data pipeline first.")
        st.code("""
# Run data pipeline:
cd src
python aggregation.py
python feature_engineering.py
python scoring.py
        """)
        return
    
    # Get latest data
    latest_scores = get_latest_month_data(scores_df)
    
    # ========================================
    # Section 1: National Summary
    # ========================================
    st.header("📊 National Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_nasri = latest_scores['nasri_score'].mean()
        st.metric(
            label="Avg NASRI Score",
            value=f"{avg_nasri:.1f}",
            delta=None,
            help="National average readiness index (0-100)"
        )
    
    with col2:
        avg_asrs = latest_scores['asrs_score'].mean()
        st.metric(
            label="Avg ASRS Score",
            value=f"{avg_asrs:.3f}",
            delta=None,
            help="National average risk score (0-1, lower is better)"
        )
    
    with col3:
        high_risk_count = (latest_scores['asrs_score'] > 0.7).sum()
        st.metric(
            label="Critical-Risk Districts",
            value=high_risk_count,
            delta=None,
            help="Districts with ASRS > 0.7"
        )
    
    with col4:
        excellent_count = (latest_scores['nasri_score'] >= 80).sum()
        st.metric(
            label="Excellent Readiness",
            value=excellent_count,
            delta=None,
            help="Districts with NASRI ≥ 80"
        )
    
    # ========================================
    # Section 2: District Rankings
    # ========================================
    st.header("🏆 District Rankings")
    
    tab1, tab2 = st.tabs(["Top Performers", "Needs Attention"])
    
    with tab1:
        st.subheader("Top 10 Districts by NASRI")
        top_districts = latest_scores.nlargest(10, 'nasri_score')[
            ['district', 'nasri_score', 'asrs_score']
        ].copy()
        top_districts['rank'] = range(1, len(top_districts) + 1)
        top_districts = top_districts[['rank', 'district', 'nasri_score', 'asrs_score']]
        
        st.dataframe(
            top_districts,
            column_config={
                'rank': st.column_config.NumberColumn('Rank', width='small'),
                'district': st.column_config.TextColumn('District'),
                'nasri_score': st.column_config.ProgressColumn(
                    'NASRI Score',
                    min_value=0,
                    max_value=100,
                    format='%.1f'
                ),
                'asrs_score': st.column_config.ProgressColumn(
                    'ASRS Risk',
                    min_value=0,
                    max_value=1,
                    format='%.3f'
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    with tab2:
        st.subheader("Bottom 10 Districts by NASRI")
        bottom_districts = latest_scores.nsmallest(10, 'nasri_score')[
            ['district', 'nasri_score', 'asrs_score']
        ].copy()
        bottom_districts['rank'] = range(1, len(bottom_districts) + 1)
        bottom_districts = bottom_districts[['rank', 'district', 'nasri_score', 'asrs_score']]
        
        st.dataframe(
            bottom_districts,
            column_config={
                'rank': st.column_config.NumberColumn('Rank', width='small'),
                'district': st.column_config.TextColumn('District'),
                'nasri_score': st.column_config.ProgressColumn(
                    'NASRI Score',
                    min_value=0,
                    max_value=100,
                    format='%.1f'
                ),
                'asrs_score': st.column_config.ProgressColumn(
                    'ASRS Risk',
                    min_value=0,
                    max_value=1,
                    format='%.3f'
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    # ========================================
    # Section 3: Risk Distribution
    # ========================================
    st.header("⚠️ Risk Distribution")
    
    # ASRS risk categories
    latest_scores['risk_category'] = pd.cut(
        latest_scores['asrs_score'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
    )
    
    risk_counts = latest_scores['risk_category'].value_counts().sort_index()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(
            risk_counts.reset_index(),
            column_config={
                'risk_category': st.column_config.TextColumn('Risk Category'),
                'count': st.column_config.NumberColumn('Districts', width='small')
            },
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        # Risk distribution bar chart
        st.bar_chart(risk_counts)
    
    # ========================================
    # Section 4: Detailed District View (RENDERS FIRST - NO BLOCKING)
    # ========================================
    # Performance Note: This section renders instantly before any map loading
    st.header("🔍 District Deep Dive")
    
    selected_district = st.selectbox(
        "Select District",
        options=sorted(latest_scores['district'].unique())
    )
    
    if selected_district:
        district_data = latest_scores[latest_scores['district'] == selected_district].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "NASRI Score",
                f"{district_data['nasri_score']:.1f}",
                help="Overall readiness index"
            )
        
        with col2:
            st.metric(
                "ASRS Score",
                f"{district_data['asrs_score']:.3f}",
                help="System risk probability"
            )
        
        with col3:
            risk_level = 'Critical' if district_data['asrs_score'] > 0.7 else \
                        'High' if district_data['asrs_score'] > 0.5 else \
                        'Medium' if district_data['asrs_score'] > 0.3 else 'Low'
            st.metric("Risk Level", risk_level)
        
        # Show all metrics for selected district
        st.subheader("All Metrics")
        
        # Convert Series to DataFrame for better display
        metrics_df = pd.DataFrame({
            'Metric': district_data.index,
            'Value': district_data.values
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # ========================================
    # Section 5: Trends Over Time
    # ========================================
    st.header("📈 Trends Over Time")
    
    # Check for date or month column
    date_col = 'date' if 'date' in scores_df.columns else 'month' if 'month' in scores_df.columns else None
    
    if date_col:
        # Aggregate national trends
        monthly_avg = scores_df.groupby(date_col).agg({
            'nasri_score': 'mean',
            'asrs_score': 'mean'
        }).reset_index()
        
        # Convert to datetime
        if monthly_avg[date_col].dtype == 'object':
            monthly_avg[date_col] = pd.to_datetime(monthly_avg[date_col])
        
        monthly_avg = monthly_avg.sort_values(date_col).set_index(date_col)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("NASRI Trend")
            st.line_chart(monthly_avg[['nasri_score']], use_container_width=True)
        
        with col2:
            st.subheader("ASRS Trend")
            st.line_chart(monthly_avg[['asrs_score']], use_container_width=True)
    else:
        st.info("Monthly trend data not available")
    
    # ========================================
    # Section 6: India District Choropleth Heatmap (LAZY LOADED)
    # ========================================
    st.header("🗺️ District-wise Aadhaar Service Index (India)")
    
    # Initialize session state for map rendering control
    if 'map_rendered' not in st.session_state:
        st.session_state.map_rendered = False
    if 'last_metric' not in st.session_state:
        st.session_state.last_metric = None
    
    # Metric selector and render button side by side
    col_metric, col_render = st.columns([3, 1])
    with col_metric:
        metric_option = st.selectbox(
            "Select Risk Metric",
            options=["ASRS (Aadhaar Service Risk Score)", "NASRI (National Aadhaar Service Readiness Index)"],
            index=0,
            key="map_metric_selector"
        )
    with col_render:
        st.write("")  # Spacing to align with selectbox
        render_map_btn = st.button("🗺️ Render Map", use_container_width=True)
    
    metric_name = 'ASRS' if 'ASRS' in metric_option else 'NASRI'
    
    # Check if metric changed - reset render state if so
    if st.session_state.last_metric != metric_name:
        st.session_state.map_rendered = False
        st.session_state.last_metric = metric_name
    
    # Render map when button clicked
    if render_map_btn:
        st.session_state.map_rendered = True
    
    if st.session_state.map_rendered:
        # Load GeoJSON (cached)
        geojson_data = load_india_geojson_cached()
        
        if geojson_data is None:
            st.error("❌ Failed to load India district boundaries.")
        else:
            # Load pre-built map
            with st.spinner(f"Loading {metric_name} map..."):
                fig, stats = load_prebuilt_map(metric_name)
            
            if fig is None:
                st.error("❌ Pre-built map not found! Run: `python scripts/prebuild_map.py`")
            else:
                # Display map
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Show stats in compact format
                if stats:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Districts Mapped", stats.get('districts_mapped', 'N/A'))
                    with col2:
                        st.metric("Match Rate", f"{stats.get('match_rate', 0):.1f}%")
                    with col3:
                        if metric_name == 'ASRS':
                            st.metric("Avg ASRS", f"{stats.get('avg_asrs', 0):.3f}")
                        else:
                            st.metric("Avg NASRI", f"{stats.get('avg_nasri', 0):.1f}")
                    with col4:
                        st.metric("High Risk", stats.get('high_risk_count', 0))
                
                # Risk band legend
                if metric_name == 'ASRS':
                    st.caption("**ASRS Risk Levels (0-1, lower is better):** 🟢 Minimal (<0.15) | 🟡 Low (0.15-0.3) | 🟠 Medium (0.3-0.5) | 🔴 High (0.5-0.75) | 🔴 Critical (>0.75)")
                else:
                    st.caption("**NASRI Readiness (0-100, higher is better):** 🔴 Critical (<20) | 🟠 Poor (20-40) | 🟡 Fair (40-60) | 🟢 Good (60-80) | 🟢 Excellent (>80)")


# ============================================================================
# Page 2: Orchestration Control Panel
# ============================================================================

def render_orchestration_panel():
    """
    Orchestration Control Panel Page
    
    Displays:
    - Critical districts requiring intervention
    - Recommended actions with impact estimates
    - Simulation interface (what-if analysis)
    - Action prioritization
    """
    st.title("🎛️ Orchestration Control Panel")
    st.markdown("**Automated intervention recommendations and impact simulation**")
    
    # Load data
    scores_df = load_scores_data()
    features_df = load_features_data()
    
    if scores_df is None:
        st.error("❌ Score data not found. Please run the data pipeline first.")
        return
    
    # Get latest data
    latest_scores = get_latest_month_data(scores_df)
    
    # ========================================
    # Section 1: Critical Districts
    # ========================================
    st.header("🚨 Critical Districts")
    
    # Filter high-risk districts
    critical_districts = latest_scores[latest_scores['asrs_score'] > 0.6].sort_values(
        'asrs_score', ascending=False
    )
    
    st.metric(
        "Districts Requiring Intervention",
        len(critical_districts),
        help="Districts with ASRS > 0.6"
    )
    
    if len(critical_districts) > 0:
        st.dataframe(
            critical_districts[['district', 'nasri_score', 'asrs_score']].head(10),
            column_config={
                'district': st.column_config.TextColumn('District'),
                'nasri_score': st.column_config.ProgressColumn(
                    'NASRI',
                    min_value=0,
                    max_value=100,
                    format='%.1f'
                ),
                'asrs_score': st.column_config.ProgressColumn(
                    'ASRS',
                    min_value=0,
                    max_value=1,
                    format='%.3f'
                )
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success("✅ No critical districts at this time")
    
    # ========================================
    # Section 2: Intervention Recommendations
    # ========================================
    st.header("💡 Intervention Recommendations")
    
    selected_district = st.selectbox(
        "Select District for Recommendations",
        options=sorted(latest_scores['district'].unique()),
        key='orchestration_district'
    )
    
    if selected_district:
        district_row = latest_scores[latest_scores['district'] == selected_district].iloc[0]
        
        # Display current state
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("NASRI", f"{district_row['nasri_score']:.1f}")
        
        with col2:
            st.metric("ASRS", f"{district_row['asrs_score']:.3f}")
        
        with col3:
            if 'compliance_debt_cumulative' in district_row:
                st.metric("Backlog", f"{district_row['compliance_debt_cumulative']:.0f}")
            else:
                st.metric("Backlog", "N/A")
        
        with col4:
            if 'capacity_utilization_pct' in district_row:
                st.metric("Capacity", f"{district_row['capacity_utilization_pct']:.0f}%")
            else:
                st.metric("Capacity", "N/A")
        
        # Get recommendations
        try:
            recommendations = recommend_actions(district_row, max_recommendations=5)
            
            if len(recommendations) > 0:
                st.subheader(f"📋 Top {len(recommendations)} Recommended Actions")
                
                for idx, action in enumerate(recommendations, 1):
                    with st.expander(
                        f"#{idx}: {action['name']} (Priority: {action['priority']})",
                        expanded=(idx == 1)
                    ):
                        st.markdown(f"**Intervention ID:** `{action['intervention_id']}`")
                        st.markdown(f"**Description:** {action.get('description', 'N/A')}")
                        
                        impact = action.get('impact_estimate', {})
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Expected NASRI Improvement",
                                f"+{impact.get('nasri_improvement', 0):.1f}",
                                help="Projected NASRI score increase"
                            )
                        
                        with col2:
                            st.metric(
                                "Expected ASRS Reduction",
                                f"-{impact.get('asrs_reduction', 0):.3f}",
                                help="Projected risk score decrease"
                            )
                        
                        with col3:
                            st.metric(
                                "Implementation Time",
                                f"{impact.get('implementation_days', 0)} days",
                                help="Estimated deployment timeline"
                            )
                        
                        st.markdown(f"**Cost Level:** {impact.get('cost_level', 'medium').upper()}")
                        st.markdown(f"**Effectiveness Probability:** {impact.get('effectiveness_probability', 0.7):.0%}")
                        
                        # Projected improvements details
                        if 'projected_improvements' in impact:
                            st.markdown("**Projected Improvements:**")
                            improvements = impact['projected_improvements']
                            for key, value in improvements.items():
                                st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
                        
                        # Simulation button
                        if st.button(f"🔮 Simulate Impact", key=f"sim_{action['intervention_id']}"):
                            st.session_state[f"simulate_{selected_district}_{action['intervention_id']}"] = True
                        
                        # Run simulation if button clicked
                        if st.session_state.get(f"simulate_{selected_district}_{action['intervention_id']}", False):
                            with st.spinner("Running 6-month impact simulation..."):
                                try:
                                    simulation_results = simulate_impact(district_row, action, simulation_months=6)
                                    
                                    st.success("✅ Simulation Complete")
                                    
                                    st.markdown("### Simulation Results")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**Baseline (No Action)**")
                                        st.metric(
                                            "Final NASRI",
                                            f"{simulation_results['baseline']['final_nasri']:.1f}"
                                        )
                                        st.metric(
                                            "Final ASRS",
                                            f"{simulation_results['baseline']['final_asrs']:.3f}"
                                        )
                                        st.metric(
                                            "Clearance Time",
                                            f"{simulation_results['baseline']['backlog_clearance_months']} months"
                                        )
                                    
                                    with col2:
                                        st.markdown("**With Intervention**")
                                        st.metric(
                                            "Final NASRI",
                                            f"{simulation_results['intervention']['final_nasri']:.1f}",
                                            delta=f"+{simulation_results['benefits']['nasri_improvement']:.1f}"
                                        )
                                        st.metric(
                                            "Final ASRS",
                                            f"{simulation_results['intervention']['final_asrs']:.3f}",
                                            delta=f"-{simulation_results['benefits']['asrs_reduction']:.3f}"
                                        )
                                        st.metric(
                                            "Clearance Time",
                                            f"{simulation_results['intervention']['backlog_clearance_months']} months",
                                            delta=f"-{simulation_results['benefits']['clearance_time_reduction_months']} months"
                                        )
                                    
                                    # Citizen impact
                                    st.markdown("### 👥 Citizen Impact")
                                    waiting_impact = simulation_results['waiting_time']
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            "Avg Waiting Time Reduction",
                                            f"{waiting_impact['avg_waiting_reduction_days']:.1f} days"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            "Improvement",
                                            f"{waiting_impact['pct_improvement']:.1f}%"
                                        )
                                    
                                    with col3:
                                        roi = simulation_results['recommendation']['roi_indicator']
                                        st.metric("ROI Indicator", roi.upper())
                                    
                                    # Trajectory visualization
                                    st.markdown("### 📊 6-Month Trajectory")
                                    
                                    # Prepare trajectory data
                                    baseline_trajectory = pd.DataFrame(simulation_results['baseline']['trajectory'])
                                    intervention_trajectory = pd.DataFrame(simulation_results['intervention']['trajectory'])
                                    
                                    trajectory_chart = pd.DataFrame({
                                        'Month': range(len(baseline_trajectory)),
                                        'Baseline NASRI': baseline_trajectory['nasri_score'],
                                        'Intervention NASRI': intervention_trajectory['nasri_score']
                                    }).set_index('Month')
                                    
                                    st.line_chart(trajectory_chart, use_container_width=True)
                                    
                                    # Recommendation
                                    if simulation_results['recommendation']['is_beneficial']:
                                        st.success(
                                            f"✅ **Recommended Action** - "
                                            f"Expected NASRI improvement: +{simulation_results['benefits']['nasri_improvement']:.1f} points, "
                                            f"Confidence: {simulation_results['recommendation']['confidence_level']:.0%}"
                                        )
                                    else:
                                        st.warning(
                                            "⚠️ Marginal impact - Consider alternative interventions"
                                        )
                                    
                                except Exception as e:
                                    st.error(f"Simulation error: {e}")
                                    logger.error(f"Simulation error: {e}", exc_info=True)
            else:
                st.info("✅ No interventions required - District performing well")
        
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            logger.error(f"Recommendation error: {e}", exc_info=True)
    
    # ========================================
    # Section 3: Batch Operations
    # ========================================
    st.header("⚙️ Batch Operations")
    
    st.markdown("""
    **Automated batch processing for all critical districts:**
    - Generate action reports
    - Prioritize interventions nationally
    - Export deployment plans
    """)
    
    if st.button("🚀 Generate National Action Report"):
        with st.spinner("Processing all districts..."):
            try:
                # Get high-risk districts
                high_risk = latest_scores[latest_scores['asrs_score'] > 0.6]
                
                all_recommendations = []
                
                for idx, row in high_risk.iterrows():
                    recommendations = recommend_actions(row, max_recommendations=3)
                    for rec in recommendations:
                        rec['district'] = row['district']
                        all_recommendations.append(rec)
                
                if len(all_recommendations) > 0:
                    # Sort by priority
                    all_recommendations.sort(key=lambda x: x['priority'])
                    
                    st.success(f"✅ Generated {len(all_recommendations)} recommendations for {len(high_risk)} districts")
                    
                    # Display summary
                    summary_df = pd.DataFrame([
                        {
                            'District': r['district'],
                            'Intervention': r['name'],
                            'Priority': r['priority'],
                            'NASRI Impact': f"+{r['impact_estimate']['nasri_improvement']:.1f}",
                            'Implementation': f"{r['impact_estimate']['implementation_days']} days"
                        }
                        for r in all_recommendations[:20]  # Top 20
                    ])
                    
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    st.download_button(
                        label="📥 Download Full Report (CSV)",
                        data=summary_df.to_csv(index=False),
                        file_name="national_action_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No recommendations needed - All districts performing well")
                
            except Exception as e:
                st.error(f"Error generating report: {e}")
                logger.error(f"Batch report error: {e}", exc_info=True)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """
    Main application entry point.
    
    Configures page layout and navigation.
    """
    # Page configuration
    st.set_page_config(
        page_title="ASEWIS Dashboard",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("🎯 ASEWIS")
    st.sidebar.markdown("**Aadhar System Engineering & Workflow Intelligence**")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["NASRI Readiness Dashboard", "Orchestration Control Panel"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Status")
    
    # Check data availability
    if load_scores_data() is not None:
        st.sidebar.success("✅ Data Loaded")
    else:
        st.sidebar.error("❌ No Data Available")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Quick Guide")
    st.sidebar.markdown("""
    **NASRI Dashboard:**
    - Monitor national readiness
    - Track district performance
    - Analyze trends
    
    **Orchestration Panel:**
    - Get intervention recommendations
    - Simulate impact scenarios
    - Generate action reports
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("UIDAI Hackathon 2026 | ASEWIS v1.0")
    
    # Route to selected page
    if page == "NASRI Readiness Dashboard":
        render_nasri_dashboard()
    elif page == "Orchestration Control Panel":
        render_orchestration_panel()


if __name__ == "__main__":
    main()
