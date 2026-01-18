"""
Geospatial utilities for India district-level mapping.

Handles:
- District name normalization and matching
- India GeoJSON loading and caching
- District alias resolution
"""

import re
import unicodedata
import json
import urllib.request
from typing import Dict, Optional, Set, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# India District GeoJSON sources (multiple fallbacks)
INDIA_DISTRICTS_GEOJSON_URLS = [
    # Primary: Datameet India boundaries
    "https://raw.githubusercontent.com/datameet/maps/master/Districts/districts.geojson",
    # Fallback 1: Alternative repository
    "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_Districts.json",
    # Fallback 2: Another source
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
]


# District name aliases for matching
# Format: DATA_NAME -> GEOJSON_NAME
# Only explicit, verified mappings (no fuzzy guessing)
DISTRICT_ALIASES = {
    # ==== SPELLING VARIANTS (Official GeoJSON uses older/alternate spellings) ====
    
    # Karnataka - Renamed districts
    'BENGALURU': 'BANGALORE URBAN',
    'BENGALURU URBAN': 'BANGALORE URBAN',
    'BENGALURU RURAL': 'BANGALORE RURAL',
    'BENGALURU SOUTH': 'BANGALORE URBAN',
    'BELAGAVI': 'BELGAUM',
    'BALLARI': 'BELLARY',
    'MYSURU': 'MYSORE',
    'TUMAKURU': 'TUMKUR',
    'SHIVAMOGGA': 'SHIMOGA',
    'VIJAYAPURA': 'BIJAPUR',
    'KALABURAGI': 'GULBARGA',
    'CHIKKAMAGALURU': 'CHIKMAGALUR',
    'HOSAPETE': 'HOSPET',
    'VIJAYANAGARA': 'BELLARY',
    
    # Gujarat
    'AHMEDABAD': 'AHMADABAD',
    'BANAS KANTHA': 'BANASKANTHA',
    'BANASKANTHA': 'BANAS KANTHA',
    'DOHAD': 'DAHOD',
    'DANG': 'THE DANGS',  # The Dangs district
    'DIU': 'DAMAN',  # Part of Daman and Diu
    
    # Uttar Pradesh
    'PRAYAGRAJ': 'ALLAHABAD',
    'AYODHYA': 'FAIZABAD',
    'SAMBHAL': 'MORADABAD',
    'AMETHI': 'SULTANPUR',
    'AMROHA': 'JYOTIBA PHULE NAGAR',
    'HAPUR': 'GHAZIABAD',
    'SHAMLI': 'MUZAFFARNAGAR',
    
    # Odisha
    'BALASORE': 'BALESHWAR',
    'BALANGIR': 'BOLANGIR',
    'BARGARH': 'BARAGARH',
    'ANUGUL': 'ANGUL',
    'JAGATSINGHAPUR': 'JAGATSINGHPUR',
    'SONEPUR': 'SUBARNAPUR',
    
    # West Bengal
    'BARDHAMAN': 'BARDDHAMAN',
    'BURDWAN': 'BARDDHAMAN',  # Old English spelling
    'PURBA BARDHAMAN': 'BARDDHAMAN',
    'PASCHIM BARDHAMAN': 'BARDDHAMAN',
    'PURBA MEDINIPUR': 'MIDNAPORE',
    'PASCHIM MEDINIPUR': 'MIDNAPORE',
    'MEDINIPUR WEST': 'MIDNAPORE',
    'MEDINIPUR': 'MIDNAPORE',
    'DINAJPUR DAKSHIN': 'DINAJPUR SOUTH',
    'DINAJPUR UTTAR': 'DINAJPUR NORTH',
    'PARAGANAS NORTH': 'NORTH TWENTY FOUR PARGANAS',
    'PARAGANAS SOUTH': 'SOUTH TWENTY FOUR PARGANAS',
    'NORTH 24 PARGANAS': 'NORTH TWENTY FOUR PARGANAS',
    'SOUTH 24 PARGANAS': 'SOUTH TWENTY FOUR PARGANAS',
    'PURBI CHAMPARAN': 'PURBA CHAMPARAN',  # Eastern Champaran,
    
    # Haryana
    'GURUGRAM': 'GURGAON',
    'CHARKHI DADRI': 'BHIWANI',
    
    # Chhattisgarh
    'KABIRDHAM': 'KAWARDHA',
    'GARIABAND': 'RAIPUR',
    
    # Jammu & Kashmir / Ladakh
    'ANANTNAG': 'ANANTNAG KASHMIR SOUTH',
    'BARAMULA': 'BARAMULA KASHMIR NORTH',
    'BADGAM': 'BAGDAM',
    'SHOPIAN': 'PULWAMA',  # New district from Pulwama (post-2007)
    
    # Andhra Pradesh (old names in GeoJSON)
    'ANANTAPURAMU': 'ANANTAPUR',
    'ANANTHAPURAMU': 'ANANTAPUR',
    'SRI POTTI SRIRAMULU NELLORE': 'NELLORE',
    'SPSR NELLORE': 'NELLORE',
    'YSR KADAPA': 'KADAPA',
    'YSR': 'KADAPA',
    'DR B R AMBEDKAR KONASEEMA': 'EAST GODAVARI',  # New district from East Godavari
    
    # Karnataka additional
    'RAMANAGARA': 'RAMANAGARAM',
    'DAKSHINA KANNADA': 'DAKSHIN KANNAD',
    'HASAN': 'HASSAN',
    'YADGIR': 'YADAGIRI',
    
    # Uttarakhand
    'HARIDWAR': 'HARDWAR',
    
    # Madhya Pradesh
    'BALAGHAT': 'BALGHAT',
    
    # Assam - Directional variants
    'KAMRUP METRO': 'KAMRUP',
    'KAMRUP METROPOLITAN': 'KAMRUP',
    
    # Bihar
    'BARABANKI': 'BARA BANKI',
    'BAGPAT': 'BAGHPAT',
    
    # ==== MODERN DISTRICTS (Post-2014 splits - Map to parent district) ====
    # These are new districts not in old GeoJSON, mapped to their parent/closest match
    
    # Telangana (created 2014, further splits 2016-2019)
    'MEDCHAL MALKAJGIRI': 'RANGA REDDY',  # New district from Ranga Reddy
    'MEDCHALMALKAJGIRI': 'RANGA REDDY',
    'RANGA REDDY': 'RANGAREDDY',  # Spelling variant
    'JANGAON': 'WARANGAL',  # New district from Warangal
    'JAYASHANKAR BHUPALPALLY': 'WARANGAL',
    'JOGULAMBA GADWAL': 'MAHBUBNAGAR',
    'KAMAREDDY': 'NIZAMABAD',
    'MANCHERIAL': 'ADILABAD',
    'NAGARKURNOOL': 'MAHBUBNAGAR',
    'PEDDAPALLI': 'KARIMNAGAR',
    'RAJANNA SIRCILLA': 'KARIMNAGAR',
    'SIDDIPET': 'MEDAK',
    'SANGAREDDY': 'MEDAK',
    'VIKARABAD': 'RANGA REDDY',
    'WANAPARTHY': 'MAHBUBNAGAR',
    'YADADRI BHUVANAGIRI': 'NALGONDA',
    'MULUGU': 'WARANGAL',
    'NARAYANPET': 'MAHBUBNAGAR',
    
    # Note: These mappings allow partial visualization.
    # Actual boundaries may differ. Update GeoJSON when 2023+ data available.
}


def normalize_district_name(name: str) -> str:
    """
    Normalize district name for matching.
    
    Rules:
    1. Unicode normalize (NFKD)
    2. Convert to UPPERCASE
    3. Remove special characters (-, _, extra spaces)
    4. Strip non-alphabetic noise
    
    Args:
        name: Raw district name
        
    Returns:
        Normalized district name
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    
    # Unicode normalization
    name = unicodedata.normalize('NFKD', name)
    
    # Convert to uppercase
    name = name.upper()
    
    # Remove common prefixes/suffixes
    name = re.sub(r'\s+(DISTRICT|DIST|DIV)\s*$', '', name)
    
    # Replace special characters with space
    name = re.sub(r'[-_./()]', ' ', name)
    
    # Remove non-alphanumeric except spaces
    name = re.sub(r'[^A-Z0-9\s]', '', name)
    
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)
    
    # Strip whitespace
    name = name.strip()
    
    return name


def resolve_district_alias(normalized_name: str) -> str:
    """
    Resolve known district aliases to canonical names.
    
    Args:
        normalized_name: Normalized district name
        
    Returns:
        Canonical district name
    """
    return DISTRICT_ALIASES.get(normalized_name, normalized_name)


def match_district_name(
    data_district: str,
    geojson_districts: Set[str],
    verbose: bool = False
) -> Optional[str]:
    """
    Match a district name from data to GeoJSON properties.
    
    Matching strategy:
    1. Normalize both names
    2. Direct match
    3. Alias resolution + match
    4. Partial match (first word, but only if unique)
    
    Args:
        data_district: District name from metrics data
        geojson_districts: Set of normalized district names from GeoJSON
        verbose: Log matching attempts
        
    Returns:
        Matched GeoJSON district name or None
    """
    normalized = normalize_district_name(data_district)
    
    if not normalized:
        return None
    
    # Direct match
    if normalized in geojson_districts:
        if verbose:
            logger.info(f"✓ Direct match: {data_district} → {normalized}")
        return normalized
    
    # Alias resolution
    resolved = resolve_district_alias(normalized)
    if resolved != normalized and resolved in geojson_districts:
        if verbose:
            logger.info(f"✓ Alias match: {data_district} → {resolved}")
        return resolved
    
    # Partial match (only if exact substring and unique)
    # More conservative - require at least 5 chars and unique match
    if len(normalized) >= 5:
        candidates = []
        for geo_dist in geojson_districts:
            # Check if normalized is substring of geo_dist or vice versa
            if normalized in geo_dist or geo_dist in normalized:
                candidates.append(geo_dist)
        
        # Only accept if exactly one match
        if len(candidates) == 1:
            if verbose:
                logger.info(f"⚠ Partial match: {data_district} → {candidates[0]}")
            return candidates[0]
    
    # No match found
    if verbose:
        logger.warning(f"✗ No match: {data_district} (normalized: {normalized})")
    
    return None


def build_district_lookup(geojson_features: list) -> Dict[str, dict]:
    """
    Build a lookup dictionary from GeoJSON features.
    
    Extracts district names from properties and creates a normalized lookup.
    Common property keys: 'district', 'dtname', 'NAME', 'DISTRICT', etc.
    
    Args:
        geojson_features: List of GeoJSON feature objects
        
    Returns:
        Dict mapping normalized district name → feature dict
    """
    district_lookup = {}
    
    logger.info(f"build_district_lookup received {len(geojson_features)} features")
    if geojson_features and len(geojson_features) > 0:
        sample_props = geojson_features[0].get('properties', {})
        logger.info(f"First feature properties keys: {list(sample_props.keys())}")
    
    for feature in geojson_features:
        props = feature.get('properties', {})
        
        # Try multiple property keys for district name
        # NAME_2 is used by standard India GeoJSON files
        district_name = None
        for key in ['NAME_2', 'district', 'dtname', 'DISTRICT', 'NAME', 'name', 'District']:
            if key in props and props[key]:
                district_name = props[key]
                break
        
        if not district_name:
            continue
        
        normalized = normalize_district_name(district_name)
        if normalized:
            district_lookup[normalized] = feature
    
    logger.info(f"Built lookup for {len(district_lookup)} districts from GeoJSON")
    if district_lookup:
        logger.info(f"Sample normalized districts: {list(district_lookup.keys())[:5]}")
    return district_lookup


def get_unmatched_districts(
    data_districts: list,
    geojson_districts: Set[str]
) -> list:
    """
    Find districts in data that cannot be matched to GeoJSON.
    
    Args:
        data_districts: List of district names from metrics data
        geojson_districts: Set of normalized district names from GeoJSON
        
    Returns:
        List of unmatched district names
    """
    unmatched = []
    
    for district in data_districts:
        matched = match_district_name(district, geojson_districts, verbose=False)
        if not matched:
            unmatched.append(district)
    
    return unmatched


def load_india_districts_geojson(cache_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Load India districts GeoJSON from multiple sources with fallbacks.
    
    Downloads and caches the GeoJSON file locally for faster subsequent loads.
    Tries multiple URLs if primary source fails.
    
    Args:
        cache_dir: Directory to cache the GeoJSON file (optional)
        
    Returns:
        GeoJSON dict or None if all sources fail
    """
    # Determine cache path
    if cache_dir:
        cache_file = cache_dir / "india_districts.geojson"
    else:
        cache_file = Path(__file__).parent.parent / "data_cache" / "india_districts.geojson"
    
    # Try loading from cache first
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                geojson = json.load(f)
            
            # Validate GeoJSON structure
            if 'features' in geojson and len(geojson['features']) > 0:
                logger.info(f"✓ Loaded {len(geojson['features'])} districts from cache: {cache_file}")
                return geojson
            else:
                logger.warning(f"Cached GeoJSON is invalid (no features), will re-download")
        except Exception as e:
            logger.warning(f"Failed to load cached GeoJSON: {e}")
    
    # Try downloading from multiple sources
    for idx, url in enumerate(INDIA_DISTRICTS_GEOJSON_URLS, 1):
        try:
            logger.info(f"Attempting download from source {idx}/{len(INDIA_DISTRICTS_GEOJSON_URLS)}: {url}")
            
            # Set timeout and user agent
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (ASEWIS Dashboard)'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                geojson = json.loads(response.read().decode('utf-8'))
            
            # Validate GeoJSON
            if 'features' not in geojson or len(geojson['features']) == 0:
                logger.warning(f"Downloaded GeoJSON from {url} has no features, trying next source")
                continue
            
            # Cache for future use
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(geojson, f, indent=2)
                logger.info(f"✓ Cached GeoJSON to {cache_file}")
            except Exception as cache_err:
                logger.warning(f"Failed to cache GeoJSON: {cache_err}")
            
            logger.info(f"✓ Successfully loaded {len(geojson['features'])} districts from {url}")
            return geojson
            
        except urllib.error.HTTPError as e:
            logger.warning(f"HTTP error downloading from {url}: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            logger.warning(f"URL error downloading from {url}: {e.reason}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from {url}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error downloading from {url}: {e}")
    
    # All sources failed
    logger.error("✗ Failed to download India districts GeoJSON from all sources")
    logger.error(f"Attempted URLs: {', '.join(INDIA_DISTRICTS_GEOJSON_URLS)}")
    
    # Provide helpful error message
    logger.error("Troubleshooting tips:")
    logger.error("1. Check internet connectivity")
    logger.error("2. Verify firewall/proxy settings")
    logger.error("3. Manually download GeoJSON and place in: " + str(cache_file))
    
    return None
