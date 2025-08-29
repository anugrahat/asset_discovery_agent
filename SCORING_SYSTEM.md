# Drug Asset Discovery Scoring System

## Overview
The Drug Asset Discovery tool uses a comprehensive scoring system to identify high-potential drug assets that may have been shelved or discontinued but could be valuable for development or repurposing.

## Scoring Components

### 1. Base Score Calculation (0-100 points)

The potential score for each drug asset is calculated based on multiple factors:

#### Clinical Development Stage (Max 40 points)
- **Phase 4**: 40 points (Post-market, proven safety/efficacy)
- **Phase 3**: 30 points (Late-stage, efficacy demonstrated)
- **Phase 2**: 20 points (Mid-stage, safety established)
- **Phase 1**: 10 points (Early-stage, initial safety data)
- **Preclinical**: 5 points

#### Target Quality (Max 20 points)
- High-confidence, validated targets: 20 points
- Medium-confidence targets: 10 points
- Unknown/unvalidated targets: 0 points

#### Safety Profile (Max 20 points)
- Clean safety profile with minimal AEs: 20 points
- Moderate safety concerns: 10 points
- Significant safety issues: 0 points

#### Market Potential (Max 20 points)
- Large unmet medical need: 20 points
- Moderate market opportunity: 10 points
- Limited market potential: 5 points

### 2. Bonus Modifiers

Additional points are added for:
- **Novel mechanism of action**: +10 points
- **Strong IP position**: +10 points
- **Recent positive data**: +15 points
- **Strategic shelving** (not efficacy/safety): +20 points

### 3. Penalty Modifiers

Points are deducted for:
- **Lack of efficacy**: -30 points
- **Safety concerns**: -25 points
- **Regulatory issues**: -20 points
- **Commercial failure**: -15 points

## Star Rating System

Based on the final potential score:
- ⭐⭐⭐ **3 Stars**: Score > 80 (Exceptional opportunity)
- ⭐⭐ **2 Stars**: Score 60-80 (Strong candidate)
- ⭐ **1 Star**: Score 40-60 (Moderate potential)
- No stars: Score < 40 (Limited potential)

## Program Status Classification

Drugs are classified by development status:
- **🔄 Active**: Currently in development
- **⏸️ Discontinued**: Development halted
- **✅ Marketed**: Approved and available
- **❓ Unknown**: Status unclear

## Shelving Reason Categories

When a drug is discontinued, we identify the reason:
- **safety**: Unacceptable adverse events
- **lack_of_efficacy**: Failed to meet endpoints
- **strategic_business**: Portfolio decisions, mergers, etc.
- **regulatory**: Failed to gain approval
- **commercial**: Market conditions unfavorable
- **unknown**: Reason not disclosed

## Data Sources

The scoring system integrates data from:
1. **ChEMBL**: Drug targets, bioactivity, development phase
2. **ClinicalTrials.gov**: Trial outcomes, safety data
3. **PubMed**: Scientific literature, efficacy data
4. **FDA databases**: Regulatory status, safety reports
5. **Press releases**: Business decisions, strategic updates
6. **Patent databases**: IP status and expiration
7. **LLM analysis**: Pattern recognition, insight generation

## Example Scoring

**Drug: Atezolizumab for Lung Cancer**
- Base: Phase 4 (40) + Safety (15) + Market (15) = 70
- Bonus: Strategic shelving (+5) = 75
- Final Score: 75/100 ⭐⭐

**Drug: Failed Phase 2 candidate**
- Base: Phase 2 (20) + Safety (10) + Market (10) = 40
- Penalty: Lack of efficacy (-30) = 10
- Final Score: 10/100 (No stars)

## Usage in CLI

The scoring system automatically:
1. Calculates scores for all discovered assets
2. Ranks drugs by potential score
3. Filters to show only high-potential candidates
4. Provides detailed breakdowns for transparency
5. Highlights best opportunities with 🏆 emoji

This scoring methodology helps identify "hidden gems" - drugs that may have been shelved for non-clinical reasons but retain significant therapeutic potential.
