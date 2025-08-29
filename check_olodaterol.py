#!/usr/bin/env python3
"""Check FDA status details for Olodaterol"""

import asyncio
import sys
sys.path.append('/home/anugraha/agent3')

from thera_agent.data.drug_safety_client import DrugSafetyClient
import json

async def check_drug():
    client = DrugSafetyClient()
    
    # Check Olodaterol (and brand name)
    for name in ["Olodaterol", "Striverdi Respimat", "Striverdi"]:
        print(f"\nChecking: {name}")
        print("-" * 40)
        
        result = await client.get_regulatory_status(name)
        if result:
            print(f"Is Approved: {result.get('is_approved')}")
            print(f"Approval Details: {len(result.get('approval_details', []))} sources")
            
            for detail in result.get('approval_details', []):
                print(f"\n  Source: {detail.get('source')}")
                if detail.get('source') == 'orange_book':
                    ob_data = detail.get('data', {})
                    print(f"    Active Products: {ob_data.get('active_products', [])}")
                    print(f"    Discontinued: {ob_data.get('discontinued_products', [])}")
                elif detail.get('source') == 'fda_api':
                    fda_data = detail.get('data', {})
                    print(f"    Brand Name: {fda_data.get('brand_name')}")
                    print(f"    Marketing Status: {fda_data.get('marketing_status')}")

if __name__ == "__main__":
    asyncio.run(check_drug())
