import pandas as pd
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from geo_utils import normalize_district_name, match_district_name, build_district_lookup

# Load data
data_path = Path(__file__).parent.parent / "dataset" / "processed" / "aggregated_metrics.parquet"
df = pd.read_parquet(data_path)
data_districts = df['district'].unique()

# Load GeoJSON
with open('data_cache/india_districts.geojson', 'r', encoding='utf-8') as f:
    geojson = json.load(f)

# Build lookup
district_lookup = build_district_lookup(geojson['features'])
geojson_districts = set(district_lookup.keys())

print("=" * 70)
print("District Matching Analysis")
print("=" * 70)
print(f"\nData districts: {len(data_districts)}")
print(f"GeoJSON districts: {len(geojson_districts)}")

# Test matching
matched = []
unmatched = []

for district in sorted(data_districts)[:30]:  # Test first 30
    normalized_data = normalize_district_name(str(district))
    matched_name = match_district_name(str(district), geojson_districts, verbose=False)
    
    if matched_name:
        matched.append((district, matched_name))
        print(f"✓ '{district}' → '{matched_name}'")
    else:
        unmatched.append(district)
        print(f"✗ '{district}' (normalized: '{normalized_data}') - NO MATCH")

print(f"\n\nMatched: {len(matched)}/{len(data_districts[:30])}")
print(f"Unmatched: {len(unmatched)}")

if unmatched:
    print(f"\nUnmatched districts:")
    for d in unmatched[:10]:
        print(f"  - {d}")
        # Find similar GeoJSON districts
        normalized = normalize_district_name(str(d))
        similar = [g for g in geojson_districts if normalized[:4] in g or g[:4] in normalized]
        if similar:
            print(f"    Similar in GeoJSON: {similar[:3]}")
