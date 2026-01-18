"""
Manual GeoJSON download script for India districts.

Run this if automatic download fails in the Streamlit app.
"""

import json
import urllib.request
from pathlib import Path

# GeoJSON sources (try in order)
GEOJSON_URLS = [
    "https://raw.githubusercontent.com/datameet/maps/master/Districts/districts.geojson",
    "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_Districts.json",
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
]

def download_geojson():
    """Download India districts GeoJSON and save to cache."""
    
    # Determine cache path
    cache_file = Path(__file__).parent.parent / "data_cache" / "india_districts.geojson"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Target location: {cache_file}\n")
    
    for idx, url in enumerate(GEOJSON_URLS, 1):
        print(f"[{idx}/{len(GEOJSON_URLS)}] Trying: {url}")
        
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (ASEWIS Dashboard)'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
                geojson = json.loads(data.decode('utf-8'))
            
            # Validate
            if 'features' not in geojson or len(geojson['features']) == 0:
                print(f"  ✗ Invalid GeoJSON (no features)\n")
                continue
            
            # Save to file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=2)
            
            print(f"  ✓ SUCCESS! Downloaded {len(geojson['features'])} districts")
            print(f"  ✓ Saved to: {cache_file}\n")
            print("🎉 GeoJSON ready! Refresh your Streamlit app.")
            return True
            
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
    
    print("❌ All download attempts failed.")
    print("\n📥 Manual download instructions:")
    print("1. Visit one of these URLs in your browser:")
    for url in GEOJSON_URLS:
        print(f"   - {url}")
    print(f"2. Save the file as: {cache_file}")
    print("3. Refresh your Streamlit app")
    
    return False


if __name__ == "__main__":
    print("=" * 70)
    print("India Districts GeoJSON Downloader")
    print("=" * 70 + "\n")
    download_geojson()
