import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from geo_utils import DISTRICT_ALIASES, normalize_district_name, match_district_name

print(f'Total aliases: {len(DISTRICT_ALIASES)}')
print('\nSample DATA→GEO mappings:')
for k, v in list(DISTRICT_ALIASES.items())[:20]:
    print(f'  {k:<35} -> {v}')

# Test some key matchings
test_districts = [
    'Bengaluru',
    'Ahmedabad',
    'Prayagraj',
    'Balasore',
    'Bardhaman',
    'Medchal Malkajgiri',
    'Purba Medinipur',
    'Anantapuramu'
]

print('\n' + '='*80)
print('NORMALIZATION & ALIAS TESTING:')
print('='*80)

for dist in test_districts:
    norm = normalize_district_name(dist)
    alias = DISTRICT_ALIASES.get(norm, norm)
    print(f'{dist:<25} -> {norm:<30} -> {alias}')
