#!/usr/bin/env python3
"""Check actual ChEMBL response structure"""
import requests
import json

# Direct API call to see structure
url = "https://www.ebi.ac.uk/chembl/api/data/drug_indication?limit=2&format=json"
response = requests.get(url)
data = response.json()

print("ChEMBL Drug Indication API Response Structure:")
print("=" * 60)
print(f"Total count: {data.get('page_meta', {}).get('total_count', 0)}")
print(f"\nKeys in response: {list(data.keys())}")

indications = data.get('drug_indications', [])
print(f"\nNumber of indications: {len(indications)}")

if indications:
    print("\nFirst indication structure:")
    first = indications[0]
    for key, value in first.items():
        print(f"  {key}: {value}")
        
    print("\nSecond indication structure:")
    second = indications[1]
    for key, value in second.items():
        print(f"  {key}: {value}")

# Now test with a specific disease
print("\n\nTesting with 'diabetes' search:")
print("=" * 60)
url2 = "https://www.ebi.ac.uk/chembl/api/data/drug_indication?indication__icontains=diabetes&limit=3&format=json"
response2 = requests.get(url2)
data2 = response2.json()

indications2 = data2.get('drug_indications', [])
print(f"Found {len(indications2)} indications")

for i, ind in enumerate(indications2[:3]):
    print(f"\nIndication {i+1}:")
    print(f"  Molecule: {ind.get('molecule_chembl_id')} - {ind.get('parent_molecule_name')}")
    print(f"  Indication: {ind.get('indication')}")
    print(f"  Max phase: {ind.get('max_phase_for_ind')}")
    print(f"  EFO: {ind.get('efo_term')}")
