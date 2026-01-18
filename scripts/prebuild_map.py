#!/usr/bin/env python3
"""
Pre-build and save India district heatmaps to disk.
Run this script after data pipeline to generate static map files.

Usage:
    python scripts/prebuild_map.py
"""

import sys
import json
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import plotly.express as px
from geo_utils import (
    load_india_districts_geojson,
    build_district_lookup,
    normalize_district_name,
    match_district_name
)

def load_latest_scores():
    """Load latest district scores."""
    # Try parquet first (preferred)
    scores_path = Path(__file__).parent.parent / 'dataset' / 'processed' / 'scores.parquet'
    
    if not scores_path.exists():
        # Fallback to CSV
        scores_path = Path(__file__).parent.parent / 'dataset' / 'processed' / 'api_data_aadhar_enrolment' / 'scores.csv'
    
    if not scores_path.exists():
        print(f"❌ Scores file not found: {scores_path}")
        print("Run data pipeline first: python src/run_pipeline.py")
        return None
    
    # Load based on file type
    if scores_path.suffix == '.parquet':
        df = pd.read_parquet(scores_path)
    else:
        df = pd.read_csv(scores_path)
    
    # Get latest month data
    if 'date' in df.columns or 'month' in df.columns:
        date_col = 'date' if 'date' in df.columns else 'month'
        df[date_col] = pd.to_datetime(df[date_col])
        latest = df.sort_values(date_col).groupby('district').tail(1)
    else:
        latest = df
    
    print(f"✅ Loaded {len(latest)} districts")
    return latest


def build_map_for_metric(latest_scores, geojson_data, metric_col, metric_name):
    """Build complete map for a metric."""
    print(f"Building {metric_name} map...")
    
    # Match districts
    district_lookup = build_district_lookup(geojson_data.get('features', []))
    geojson_districts = set(district_lookup.keys())
    
    matched_data = []
    for _, row in latest_scores.iterrows():
        district_name = row['district']
        matched_name = match_district_name(district_name, geojson_districts, verbose=False)
        
        if matched_name:
            matched_row = row.copy()
            matched_row['geojson_district'] = matched_name
            matched_data.append(matched_row)
    
    if not matched_data:
        print(f"❌ No districts matched for {metric_name}")
        return None
    
    matched_df = pd.DataFrame(matched_data)
    match_rate = len(matched_df) / len(latest_scores) * 100
    print(f"  Matched: {len(matched_df)}/{len(latest_scores)} districts ({match_rate:.1f}%)")
    
    # Update GeoJSON properties
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
            
            if normalized in district_metric_map:
                props[metric_name] = district_metric_map[normalized]
                props['matched'] = True
            else:
                props[metric_name] = None
                props['matched'] = False
    
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
    
    # Create choropleth - INDIA ONLY (zoom and center on India)
    fig = px.choropleth_mapbox(
        matched_df,
        geojson=geojson_data,
        locations='geojson_district',
        featureidkey='properties.normalized_district',
        color=metric_col,
        color_continuous_scale=color_scale,
        range_color=[matched_df[metric_col].min(), matched_df[metric_col].max()],
        mapbox_style="carto-positron",
        zoom=4,  # Closer zoom on India
        center={"lat": 23.5, "lon": 78.9},  # India center
        opacity=0.8,
        labels={metric_col: metric_name},
        hover_data={
            'geojson_district': False,
            'district': True,
            metric_col: ':.3f' if metric_name == 'ASRS' else ':.1f'
        }
    )
    
    # Clean layout - India focused
    fig.update_layout(
        height=750,
        margin=dict(l=0, r=0, t=30, b=0),
        title=None,  # Remove title for cleaner look
        showlegend=False,
        coloraxis_colorbar=dict(
            title=metric_name,
            thickness=12,
            len=0.6,
            x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            borderwidth=0
        ),
        mapbox=dict(
            bearing=0,
            pitch=0
        )
    )
    
    # Hide mapbox attribution for cleaner look
    fig.update_layout(
        mapbox_accesstoken=None,
        mapbox_style="carto-positron"
    )
    
    print(f"  ✅ {metric_name} map built")
    return fig, matched_df, match_rate


def main():
    """Build and save all maps."""
    print("="*60)
    print("PRE-BUILDING INDIA DISTRICT HEATMAPS")
    print("="*60)
    
    # Load data
    cache_dir = Path(__file__).parent.parent / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    
    print("\n1. Loading GeoJSON...")
    geojson_data = load_india_districts_geojson(cache_dir)
    if not geojson_data:
        print("❌ Failed to load GeoJSON")
        return
    print(f"✅ Loaded {len(geojson_data.get('features', []))} polygons")
    
    print("\n2. Loading latest scores...")
    latest_scores = load_latest_scores()
    if latest_scores is None:
        return
    
    # Build maps for both metrics
    print("\n3. Building maps...")
    
    maps = {}
    stats = {}
    
    for metric_col, metric_name in [('asrs_score', 'ASRS'), ('nasri_score', 'NASRI')]:
        fig, matched_df, match_rate = build_map_for_metric(
            latest_scores, 
            geojson_data.copy(),  # Copy to avoid mutation
            metric_col, 
            metric_name
        )
        
        if fig:
            maps[metric_name] = fig
            stats[metric_name] = {
                'districts_mapped': len(matched_df),
                'match_rate': match_rate,
                'high_risk_count': len(matched_df[matched_df['asrs_risk_category'].isin(['high', 'critical'])]),
                'avg_nasri': matched_df['nasri_score'].mean(),
                'avg_asrs': matched_df['asrs_score'].mean()
            }
    
    # Save to disk
    print("\n4. Saving pre-built maps...")
    
    map_cache_dir = cache_dir / "maps"
    map_cache_dir.mkdir(exist_ok=True)
    
    for metric_name, fig in maps.items():
        # Save as JSON (Plotly native format - fastest to load)
        # Use write_json instead of to_json() + json.dump to avoid double-encoding
        json_path = map_cache_dir / f"{metric_name.lower()}_map.json"
        fig.write_json(json_path)
        print(f"  ✅ Saved {json_path}")
    
    # Save stats
    stats_path = map_cache_dir / "map_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✅ Saved {stats_path}")
    
    print("\n" + "="*60)
    print("✅ PRE-BUILT MAPS READY!")
    print("="*60)
    print(f"\nMaps saved in: {map_cache_dir}")
    print("\nTo use in Streamlit:")
    print("  - Maps load instantly from disk")
    print("  - No runtime computation needed")
    print("  - Zoom focused on India only")
    print("\nRe-run this script after data updates.")


if __name__ == "__main__":
    main()
