Essential Workers Mental Health Analysis

Analyzes mental health outcomes among essential workers with/without children using U.S. Census Household Pulse Survey data (April 2021 - August 2022).

Quick Start

pip install -r requirements.txt
python .\essential_analysis_with_tracking.py

Data Requirements

Place CSV files in `data/hps/survey/` directory:
- Files: `pulse28_puf_*.csv` through `pulse48_puf_*.csv`
- Required variables: `WEEK`, `PWEIGHT`, `SETTING`, `THHLD_NUMKID`, `ANXIOUS`, `WORRY`, `DOWN`, `EGENID_BIRTH`, `TBIRTH_YEAR`

Essential Worker Classification

Period 1: Weeks 28-33 (April-July 2021)
- Code 1: Healthcare (e.g., hospital, doctor, dentist or mental health specialist office, outpatient facility, long-term care, home health care, pharmacy, medical laboratory)
- Code 9: Food and beverage store (e.g., grocery store, warehouse club, supercenters, convenience store, specialty food store, bakery)
- Code 11: Food manufacturing facility (e.g., meat-processing, produce packing, food or beverage manufacturing)
- Code 12: Non-food manufacturing facility (e.g. metals, equipment and machinery, electronics)
- Code 15: Other job deemed “essential” during the COVID-19 pandemic

Period 2: Weeks 34-48 (July 2021-August 2022)
- Code 1: Hospital
- Code 2: Nursing and residential healthcare facility
- Code 3: Pharmacy
- Code 4: Ambulatory healthcare (e.g. doctor, dentist or mental health specialist office, outpatient facility, medical and diagnostic laboratory, home health care) 
- Code 12: Food and beverage store (e.g., grocery store, warehouse club, supercenters, convenience store, specialty food store, bakery) 
- Code 14: Food manufacturing facility (e.g., meat-processing, produce packing, food or beverage manufacturing) 
- Code 15: Non-food manufacturing facility (e.g. metals, equipment and machinery, electronics)
- Code 18: Other job deemed “essential” during the COVID-19 pandemic

Note: The Census Bureau revised SETTING categories between periods to better capture evolving essential work definitions during the pandemic.

Mental Health Measures

Binary indicators (≥3 = positive screen):
- Anxiety: "Feeling nervous, anxious, or on edge"
- Worry: "Not being able to stop or control worrying"
- Depression: "Feeling down, depressed, or hopeless"

Scale: 1=Not at all, 2=Several days, 3=More than half the days, 4=Nearly every day

Output Files

Charts:
- `essential_chart_1_side_by_side.png` - Main comparison
- `essential_chart_2_statistical_strength.png` - Evidence strength
- `essential_chart_3_gender_comparison.png` - Gender stratified
- `time_series_weeks_28_48.png` - Trends over time

Reports:
- `weeks_28_48_analysis_report.md` - Main findings
- `gender_analysis_report.md` - Gender analysis

Key Processing Steps

1. Load survey files (weeks 28-48)
2. Classify essential workers by period-specific SETTING codes
3. Apply data hygiene (exclude -99, -88 codes)
4. Filter to adults (18+ years)
5. Create children status (THHLD_NUMKID ≥ 1)
6. Generate mental health indicators
7. Calculate survey-weighted statistics

Dependencies

```
pandas, numpy, matplotlib, seaborn, scipy
```

Results Structure

- Survey-weighted prevalence rates by group
- Statistical significance tests (p < 0.05)
- Time series analysis across 20 weeks
- Gender-stratified comparisons
- Complete data tracking at each step
