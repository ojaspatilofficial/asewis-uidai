"""
Download latest (2023-2024) India district GeoJSON from authoritative sources.

Priority order:
1. Datameet India - Latest District Boundaries
2. ML InfoMap India (2023 Districts)
"""

import requests
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Authoritative sources for India district boundaries (2023-2024)
GEOJSON_SOURCES = [
    {
        "name": "Datameet India Districts (Latest)",
        "url": "https://raw.githubusercontent.com/datameet/maps/master/Districts/districts.geojson",
        "year": "2023+",
        "priority": 1
    },
    {
        "name": "Datameet India Districts (Alternative)",
        "url": "https://raw.githubusercontent.com/datameet/maps/master/Districts/india_district.geojson",
        "year": "2023+",
        "priority": 2
    },
    {
        "name": "India GeoJSON (District Level)",
        "url": "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_Districts.json",
        "year": "2022+",
        "priority": 3
    },
    {
        "name": "Datameet Census 2011 (Fallback - DO NOT USE)",
        "url": "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/india_district.geojson",
        "year": "2011",
        "priority": 99  # Avoid this - outdated
    }
]

def download_geojson(url: str, output_path: Path) -> bool:
    """Download GeoJSON from URL and save to file."""
    try:
        logger.info(f"Downloading from: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Parse to validate JSON
        geojson_data = response.json()
        
        # Basic validation
        if 'features' not in geojson_data:
            logger.error("Invalid GeoJSON: missing 'features' key")
            return False
        
        features_count = len(geojson_data['features'])
        logger.info(f"✓ Downloaded {features_count} features")
        
        # Sample district names
        sample_districts = []
        for feature in geojson_data['features'][:5]:
            props = feature.get('properties', {})
            # Try common district name keys
            for key in ['dtname', 'district', 'DISTRICT', 'NAME_2', 'NAME', 'name']:
                if key in props and props[key]:
                    sample_districts.append(props[key])
                    break
        
        logger.info(f"Sample districts: {sample_districts}")
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, ensure_ascii=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Saved to {output_path} ({file_size_mb:.1f} MB)")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def main():
    # Output path
    output_file = Path(__file__).parent.parent / "data_cache" / "india_districts_2023.geojson"
    
    logger.info("=" * 80)
    logger.info("DOWNLOADING LATEST INDIA DISTRICT GEOJSON (2023-2024)")
    logger.info("=" * 80)
    
    # Try sources in priority order
    for source in sorted(GEOJSON_SOURCES, key=lambda x: x['priority']):
        if source['priority'] >= 99:
            logger.warning(f"⚠️ Skipping outdated source: {source['name']} ({source['year']})")
            continue
        
        logger.info(f"\n📥 Trying source #{source['priority']}: {source['name']} ({source['year']})")
        
        if download_geojson(source['url'], output_file):
            logger.info(f"\n✅ SUCCESS! Downloaded from: {source['name']}")
            logger.info(f"File saved: {output_file}")
            return True
        else:
            logger.warning(f"❌ Failed to download from: {source['name']}")
    
    logger.error("\n❌ All download attempts failed!")
    return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
