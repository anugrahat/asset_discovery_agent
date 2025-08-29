#!/usr/bin/env python3
"""
Validate the enhanced scoring system step-by-step
"""

import json

def validate_scoring():
    print("🔍 SCORING VALIDATION - Step by Step Breakdown")
    print("=" * 80)
    
    # Load the results
    with open('/tmp/hypertension_reason_weighted.json', 'r') as f:
        data = json.load(f)
    
    candidates = data.get('drug_rescue', [])[:8]
    
    print(f"📊 DETAILED SCORING BREAKDOWN:")
    print()
    
    for i, candidate in enumerate(candidates, 1):
        drug = candidate.get('drug', 'Unknown')
        total_trials = candidate.get('total_trials', 0)
        phases = candidate.get('phases', [])
        completed = candidate.get('completed', 0)
        failed = candidate.get('failed', 0)
        
        # Failure reasons
        recruitment_failures = candidate.get('recruitment_failures', 0)
        business_failures = candidate.get('business_failures', 0)
        safety_failures = candidate.get('safety_failures', 0)
        efficacy_failures = candidate.get('efficacy_failures', 0)
        other_failures = candidate.get('other_failures', 0)
        
        final_score = candidate.get('repurposing_score', 0)
        
        print(f"{i}. {drug}")
        print("-" * 50)
        
        # Manually calculate expected score
        expected_score = 0
        
        # 1. Trial volume (max 50)
        trial_points = min(total_trials * 10, 50)
        expected_score += trial_points
        print(f"   Trial Volume: {total_trials} trials × 10 = {trial_points} points")
        
        # 2. Phase bonuses
        phase_points = 0
        if "PHASE1" in phases:
            phase_points += 20
            print(f"   Phase 1 Bonus: +20 points")
        if "PHASE2" in phases:
            phase_points += 15
            print(f"   Phase 2 Bonus: +15 points")
        if "PHASE3" not in phases:
            phase_points += 10
            print(f"   No Phase 3 Bonus: +10 points")
        expected_score += phase_points
        
        # 3. Completion bonus
        completion_points = 0
        if completed > 0:
            completion_points = 15
            print(f"   Completion Bonus: {completed} completed → +15 points")
        expected_score += completion_points
        
        # 4. Failure penalty
        failure_rate = failed / max(total_trials, 1) if total_trials > 0 else 0
        failure_penalty = failure_rate * 20
        expected_score -= failure_penalty
        print(f"   Failure Penalty: {failed}/{total_trials} = {failure_rate:.2f} → -{failure_penalty:.1f} points")
        
        # 5. FAILURE REASON BONUSES (NEW!)
        reason_bonus = 0
        if recruitment_failures > 0:
            reason_weight = recruitment_failures / max(total_trials, 1)
            bonus = 50 * reason_weight
            reason_bonus += bonus
            print(f"   🎯 Recruitment Bonus: {recruitment_failures}/{total_trials} × 50 = +{bonus:.1f} points")
        
        if business_failures > 0:
            reason_weight = business_failures / max(total_trials, 1)
            bonus = 40 * reason_weight
            reason_bonus += bonus
            print(f"   💰 Business Bonus: {business_failures}/{total_trials} × 40 = +{bonus:.1f} points")
        
        if safety_failures > 0:
            reason_weight = safety_failures / max(total_trials, 1)
            penalty = 50 * reason_weight
            reason_bonus -= penalty
            print(f"   ⚠️ Safety Penalty: {safety_failures}/{total_trials} × 50 = -{penalty:.1f} points")
        
        if efficacy_failures > 0:
            reason_weight = efficacy_failures / max(total_trials, 1)
            penalty = 30 * reason_weight
            reason_bonus -= penalty
            print(f"   📉 Efficacy Penalty: {efficacy_failures}/{total_trials} × 30 = -{penalty:.1f} points")
        
        if other_failures > 0:
            reason_weight = other_failures / max(total_trials, 1)
            bonus = 10 * reason_weight
            reason_bonus += bonus
            print(f"   🔄 Other Bonus: {other_failures}/{total_trials} × 10 = +{bonus:.1f} points")
        
        expected_score += reason_bonus
        
        print(f"   ──────────────────────────────")
        print(f"   Expected Score: {expected_score:.1f}")
        print(f"   Actual Score:   {final_score:.1f}")
        
        # Validation check
        if abs(expected_score - final_score) < 0.1:
            print(f"   ✅ SCORING CORRECT")
        else:
            print(f"   ❌ SCORING MISMATCH!")
            print(f"   🐛 Difference: {abs(expected_score - final_score):.1f}")
        
        print()

if __name__ == "__main__":
    validate_scoring()
