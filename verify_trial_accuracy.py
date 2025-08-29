#!/usr/bin/env python3
"""
Verify accuracy of clinical trial data and GPT analysis
"""
import asyncio
import aiohttp
import json
import os
from openai import AsyncOpenAI

async def verify_trial_accuracy():
    """Verify actual trial data vs GPT analysis"""
    
    # Get actual trial data from ClinicalTrials.gov API
    async with aiohttp.ClientSession() as session:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.term": "Anetumab Ravtansine",
            "format": "json",
            "pageSize": 20
        }
        
        print("🔍 Fetching actual trial data from ClinicalTrials.gov...")
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    studies = data.get('studies', [])
                    print(f"📊 Found {len(studies)} studies")
                    
                    # Extract key trials mentioned in the fact-check
                    key_trials = {
                        'NCT02839681': None,
                        'NCT03926143': None, 
                        'NCT03455556': None,
                        'NCT02639091': None,
                        'NCT02610140': None
                    }
                    
                    verified_trials = []
                    
                    for study in studies:
                        protocol = study.get('protocolSection', {})
                        nct_id = protocol.get('identificationModule', {}).get('nctId')
                        
                        if nct_id in key_trials:
                            # Extract detailed information
                            id_module = protocol.get('identificationModule', {})
                            status_module = protocol.get('statusModule', {})
                            design_module = protocol.get('designModule', {})
                            
                            trial_info = {
                                'nct_id': nct_id,
                                'title': id_module.get('briefTitle', ''),
                                'official_title': id_module.get('officialTitle', ''),
                                'status': status_module.get('overallStatus', ''),
                                'why_stopped': status_module.get('whyStopped', ''),
                                'phases': design_module.get('phases', []),
                                'study_type': design_module.get('studyType', ''),
                                'enrollment': design_module.get('enrollmentInfo', {}).get('count'),
                                'start_date': status_module.get('startDateStruct', {}).get('date'),
                                'completion_date': status_module.get('completionDateStruct', {}).get('date'),
                                'primary_completion_date': status_module.get('primaryCompletionDateStruct', {}).get('date')
                            }
                            
                            verified_trials.append(trial_info)
                            key_trials[nct_id] = trial_info
                            
                            print(f"\n📋 {nct_id}:")
                            print(f"  Title: {trial_info['title'][:80]}...")
                            print(f"  Status: {trial_info['status']}")
                            print(f"  Why Stopped: {trial_info['why_stopped']}")
                            print(f"  Phases: {trial_info['phases']}")
                            print(f"  Enrollment: {trial_info['enrollment']}")
                    
                    # Now test GPT analysis with actual data
                    print(f"\n🤖 Testing GPT analysis with verified data...")
                    
                    openai_api_key = os.environ.get('OPENAI_API_KEY')
                    if openai_api_key:
                        openai_client = AsyncOpenAI(api_key=openai_api_key)
                        
                        # Create improved prompt with actual trial data
                        trials_prompt = f"""
                        Analyze these Anetumab Ravtansine clinical trials. Be EXTREMELY ACCURATE about termination reasons and avoid making efficacy claims without evidence.
                        
                        CRITICAL INSTRUCTIONS:
                        - Use ONLY the provided data for termination reasons
                        - Do NOT infer efficacy unless explicitly stated in results
                        - If "whyStopped" mentions accrual, enrollment, or strategic reasons, use those exact terms
                        - Do NOT claim efficacy for safety/tolerability studies
                        - Be conservative with claims
                        
                        Trial data: {json.dumps(verified_trials, indent=2)}
                        
                        Return ONLY valid JSON:
                        [
                            {{"nct_id": "NCT12345", "phase": "PHASE1", "notes": "Exact reason based on data provided"}}
                        ]
                        
                        Requirements:
                        - Use exact termination reasons from whyStopped field
                        - No efficacy claims unless proven
                        - Keep notes under 80 characters
                        - Be factually accurate
                        """
                        
                        try:
                            response = await openai_client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are a clinical trial analyst. Be extremely accurate and conservative. Use only provided data. Do not make unsupported efficacy claims."},
                                    {"role": "user", "content": trials_prompt}
                                ],
                                temperature=0.0  # Maximum accuracy
                            )
                            
                            gpt_response = response.choices[0].message.content
                            print(f"\n📝 GPT Analysis:")
                            print(gpt_response)
                            
                            # Parse and compare
                            try:
                                gpt_analysis = json.loads(gpt_response)
                                print(f"\n📊 Comparison of GPT vs Actual Data:")
                                
                                for trial in gpt_analysis:
                                    nct_id = trial.get('nct_id')
                                    if nct_id in key_trials and key_trials[nct_id]:
                                        actual = key_trials[nct_id]
                                        print(f"\n{nct_id}:")
                                        print(f"  GPT Notes: {trial.get('notes')}")
                                        print(f"  Actual Status: {actual['status']}")
                                        print(f"  Actual Why Stopped: {actual['why_stopped']}")
                                        print(f"  Actual Enrollment: {actual['enrollment']}")
                                        
                            except json.JSONDecodeError:
                                print("❌ GPT response is not valid JSON")
                                
                        except Exception as e:
                            print(f"❌ GPT API error: {e}")
                    else:
                        print("❌ No OpenAI API key found")
                        
                else:
                    print(f"❌ API request failed: {response.status}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(verify_trial_accuracy())
