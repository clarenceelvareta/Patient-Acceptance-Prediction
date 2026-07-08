# =============================================================
# FEATURE LEAKAGE DETECTION SCRIPT
# =============================================================
# Data leakage occurs when a feature contains information about
# the outcome that would NOT have been available at the time
# the referral was submitted — i.e. the model is "cheating"
# by looking into the future.
#
# Example of leakage:
#   RS Referral Status = "Closed - Admitted"
#   → This is recorded AFTER the outcome, so using it to
#     predict the outcome is circular. The model learns
#     "Closed - Admitted → Accepted" trivially.
#
# This script runs 4 leakage detection methods:
#   1. Correlation check    — features suspiciously correlated with outcome
#   2. Single-feature model — can any ONE feature alone predict outcome well?
#   3. Temporal check       — are any features recorded after the outcome?
#   4. Cardinality check    — do any features perfectly encode the outcome?
# =============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("FEATURE LEAKAGE DETECTION")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\nStep 1: Loading data...")
df = pd.read_csv('combined_repeat_patients.csv')

# Map outcomes
outcome_map = {
    'admit':                                        0,
    'reject by agency':                             1,
    'pre assign withdrawn':                         2,
    'pre assign withdrawn-bpr':                     2,
    'withdrawn':                                    2,
    'withdrawn and case closed by referral source': 2,
    'withdrawn from service provider waiting list': 2,
}
df['Outcome_Code'] = (
    df['RS Outcome Item Desc']
    .astype(str).str.strip().str.lower()
    .map({k.lower(): v for k, v in outcome_map.items()})
)
df = df[df['Outcome_Code'].notna()].copy()
df['Outcome_Code'] = df['Outcome_Code'].astype(int)

CLASS_NAMES = {0: 'Accepted', 1: 'Rejected', 2: 'Withdrawn'}
print(f"  Rows loaded: {len(df):,}")
print(f"  Outcome distribution:")
for code, name in CLASS_NAMES.items():
    n = (df['Outcome_Code'] == code).sum()
    print(f"    {name:>10}: {n:>8,}  ({n/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 2: IDENTIFY ALL FEATURE COLUMNS
# Split into categories for targeted leakage analysis
# ─────────────────────────────────────────────────────────────

# Columns that are definitionally SAFE (available at referral time)
SAFE_COLS = [
    'Gender', 'Race', 'Citizenship', 'Age AO Application', 'Age Group',
    'DOB', 'Religion', 'Living Arrangement', 'Lift Landing Ind',
    'Accom Status', 'Accom Summary', 'Num of Caregiver',
    'AIC Log Level of Assitance', 'AIC Log Feeding', 'AIC Log Mobility',
    'AIC Log Toileting', 'AIC Log Transfer', 'AIC Log Hearing Impairment',
    'AIC Log Mental Status', 'AIC Log Activity Tolerance',
    'AIC Log Visual Impairment', 'AIC Log Wound Care',
    'AIC Log Aids Needed', 'AIC Log Respiratory Care',
    'AIC Log Patient Curr Location', 'RS Svc Type Desc',
    'Referral Source', 'SP Full Name', 'During Covid?',
    'ct_brotherfamilyrelationship', 'ct_daughter', 'ct_daughter-in-law',
    'ct_neighbour', 'ct_other relatives', 'ct_otherrelative',
    'ct_parent', 'ct_sibling', 'ct_sisterfamilyrelationship',
    'ct_son', 'ct_son-in-law', 'ct_spouse',
]

# Columns that are HIGH RISK for leakage
# (recorded during or after outcome resolution)
HIGH_RISK_COLS = [
    'RS Referral Status',        # updated as referral progresses
    'RS SubOutcome Item Desc',   # recorded at outcome time
    'RS Last Status Datetime',   # timestamp of final status = after outcome
    'RS Outcome Item Desc',      # THE outcome itself — must exclude
    'joined',                    # unclear when recorded
    'count',                     # may be post-hoc aggregate
]

# Frequency-encoded columns (derived — check if they encode outcome info)
DERIVED_COLS = [
    'Referral Source_freq', 'Referral Source_is_rare',
    'SP Full Name_freq', 'SP Full Name_is_rare',
    'duplicate_count',
]

ALL_FEATURE_COLS = [c for c in df.columns if c not in [
    'Application ID', 'Pseudo NRIC', 'Outcome_Code',
    'RS Outcome Item Desc', 'RS Submit Datetime', 'RS Submit Date',
    'DOB', 'DOB Est', 'First_Submit',
]]

print(f"\n  Total feature columns to check: {len(ALL_FEATURE_COLS)}")

# ─────────────────────────────────────────────────────────────
# STEP 3: TEMPORAL LEAKAGE CHECK
# Compare timestamps: if a column's data is recorded AFTER
# RS Submit Datetime, it cannot be used as a predictor.
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("CHECK 1: Temporal Leakage (timestamp comparison)")
print("─" * 65)

datetime_cols = []
for col in df.columns:
    if 'datetime' in col.lower() or 'date' in col.lower():
        datetime_cols.append(col)

print(f"  Datetime columns found: {datetime_cols}\n")

if 'RS Submit Datetime' in df.columns and 'RS Last Status Datetime' in df.columns:
    df['RS Submit Datetime']      = pd.to_datetime(df['RS Submit Datetime'],      errors='coerce')
    df['RS Last Status Datetime'] = pd.to_datetime(df['RS Last Status Datetime'], errors='coerce')

    both_valid = df[['RS Submit Datetime', 'RS Last Status Datetime']].dropna()
    after_pct  = (both_valid['RS Last Status Datetime'] > both_valid['RS Submit Datetime']).mean()

    print(f"  RS Last Status Datetime is AFTER RS Submit Datetime")
    print(f"  in {after_pct*100:.1f}% of rows")
    print()
    if after_pct > 0.5:
        print("  ⚠️  LEAKAGE CONFIRMED: RS Last Status Datetime is recorded")
        print("     AFTER the referral is submitted — this column must be")
        print("     EXCLUDED from the model. It encodes future information.")
    else:
        print("  ✓ No temporal leakage detected in datetime columns.")
else:
    print("  RS Submit Datetime or RS Last Status Datetime not found — skipping.")

# ─────────────────────────────────────────────────────────────
# STEP 4: SINGLE-FEATURE PREDICTIVE POWER CHECK
# Train a simple model using ONLY ONE feature at a time.
# If any single feature achieves >90% accuracy alone,
# it is almost certainly leaking outcome information.
# A legitimate predictor should not be this powerful alone.
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("CHECK 2: Single-Feature Predictive Power")
print("  (any single feature >85% accuracy = suspect leakage)")
print("─" * 65)

leakage_suspects = []
results_single = []

# Check high-risk columns first, then a sample of safe columns
cols_to_check = [c for c in HIGH_RISK_COLS + DERIVED_COLS if c in df.columns and c != 'RS Outcome Item Desc']
cols_to_check += [c for c in SAFE_COLS if c in df.columns][:10]  # sample of safe cols

for col in cols_to_check:
    try:
        col_data = df[col].copy()

        # Encode if categorical
        if col_data.dtype == object:
            le = LabelEncoder()
            col_data = le.fit_transform(col_data.fillna('Unknown').astype(str))
        else:
            col_data = pd.to_numeric(col_data, errors='coerce').fillna(0).values

        X = col_data.reshape(-1, 1)
        y = df['Outcome_Code'].values

        # Simple decision tree — 3-fold CV
        from sklearn.tree import DecisionTreeClassifier
        clf   = DecisionTreeClassifier(max_depth=5, random_state=42)
        score = cross_val_score(clf, X, y, cv=3, scoring='accuracy').mean()

        is_suspect = score > 0.85
        risk_label = '⚠️  HIGH RISK' if col in HIGH_RISK_COLS else \
                     '⚡ DERIVED'    if col in DERIVED_COLS   else \
                     '✓ SAFE'

        results_single.append({
            'Feature':     col,
            'Category':    risk_label,
            'Accuracy':    round(score, 4),
            'Suspect':     is_suspect,
        })

        if is_suspect:
            leakage_suspects.append(col)

    except Exception as e:
        results_single.append({'Feature': col, 'Category': '?', 'Accuracy': None, 'Suspect': False})

results_df = pd.DataFrame(results_single).sort_values('Accuracy', ascending=False)
print(f"\n  {'Feature':<45} {'Category':<15} {'Solo Accuracy':>14} {'Flag':>8}")
print(f"  {'-'*85}")
for _, row in results_df.iterrows():
    acc_str = f"{row['Accuracy']*100:.2f}%" if row['Accuracy'] is not None else "N/A"
    flag    = "⚠️ SUSPECT" if row['Suspect'] else ""
    print(f"  {str(row['Feature']):<45} {str(row['Category']):<15} {acc_str:>14} {flag:>8}")

# ─────────────────────────────────────────────────────────────
# STEP 5: OUTCOME VALUE OVERLAP CHECK
# For categorical columns, check if any unique value
# appears EXCLUSIVELY in one outcome class.
# e.g. if RS Referral Status = "Closed - Admitted" only ever
# appears when Outcome = Accepted, it's a direct proxy leak.
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("CHECK 3: Categorical Value → Outcome Overlap")
print("  (does any single category value predict outcome perfectly?)")
print("─" * 65)

overlap_results = []
cat_cols_to_check = [c for c in HIGH_RISK_COLS if c in df.columns and c != 'RS Outcome Item Desc']

for col in cat_cols_to_check:
    value_outcomes = df.groupby(col)['Outcome_Code'].nunique()
    pure_values    = value_outcomes[value_outcomes == 1]  # values with only 1 outcome

    total_values   = value_outcomes.shape[0]
    pure_count     = pure_values.shape[0]
    pure_pct       = pure_count / total_values * 100 if total_values > 0 else 0

    overlap_results.append({
        'Column':             col,
        'Unique_Values':      total_values,
        'Pure_Values':        pure_count,
        'Pure_Pct':           round(pure_pct, 1),
        'Leakage_Risk':       'HIGH' if pure_pct > 50 else 'MODERATE' if pure_pct > 20 else 'LOW',
    })

    if pure_pct > 20:
        print(f"\n  Column: {col}")
        print(f"    {pure_count}/{total_values} unique values ({pure_pct:.1f}%) predict ONLY ONE outcome")
        print(f"    Sample pure values:")
        sample = df[df[col].isin(pure_values.index)].groupby(col)['Outcome_Code'].first().head(5)
        for val, code in sample.items():
            n = (df[col] == val).sum()
            print(f"      '{val}' → always {CLASS_NAMES[code]} ({n} rows)")

overlap_df = pd.DataFrame(overlap_results)
if not overlap_df.empty:
    print(f"\n  Summary:")
    print(overlap_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 6: CORRELATION CHECK (numerical features)
# Point-biserial correlation between each numerical feature
# and the binary Accepted (1/0) outcome.
# Legitimate features: |corr| < 0.7
# Suspect features:    |corr| > 0.7 (too perfectly correlated)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("CHECK 4: Correlation with Outcome (numerical features)")
print("  (|correlation| > 0.7 = suspect leakage)")
print("─" * 65)

df['Is_Accepted'] = (df['Outcome_Code'] == 0).astype(int)
num_cols = [c for c in NUMERICAL_COLS if c in df.columns] \
    if 'NUMERICAL_COLS' in dir() else \
    df.select_dtypes(include=[np.number]).columns.tolist()

corr_results = []
for col in num_cols:
    if col in ['Outcome_Code', 'Is_Accepted']:
        continue
    try:
        corr = df[col].corr(df['Is_Accepted'])
        if not np.isnan(corr):
            corr_results.append({'Feature': col, 'Correlation': round(corr, 4),
                                 'Abs_Corr': round(abs(corr), 4)})
    except:
        pass

corr_df = pd.DataFrame(corr_results).sort_values('Abs_Corr', ascending=False)
high_corr = corr_df[corr_df['Abs_Corr'] > 0.7]

if not high_corr.empty:
    print(f"\n  ⚠️  {len(high_corr)} features with |correlation| > 0.7:")
    print(high_corr.to_string(index=False))
else:
    print("\n  ✓ No numerical features with suspiciously high correlation.")

print(f"\n  Top 10 most correlated features:")
print(corr_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 7: FINAL VERDICT & RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("LEAKAGE VERDICT & RECOMMENDATIONS")
print("=" * 65)

confirmed_leaks  = []
suspected_leaks  = []
safe_to_use      = []

# RS Last Status Datetime — always a leak (recorded after outcome)
if 'RS Last Status Datetime' in df.columns:
    confirmed_leaks.append('RS Last Status Datetime')

# RS SubOutcome Item Desc — recorded at outcome time
if 'RS SubOutcome Item Desc' in df.columns:
    confirmed_leaks.append('RS SubOutcome Item Desc')

# Anything flagged by single-feature check
for col in leakage_suspects:
    if col not in confirmed_leaks:
        suspected_leaks.append(col)

# Overlap results
for _, row in overlap_df.iterrows():
    if row['Leakage_Risk'] == 'HIGH' and row['Column'] not in confirmed_leaks:
        suspected_leaks.append(row['Column'])

suspected_leaks = list(set(suspected_leaks))

print("\n  CONFIRMED LEAKS — remove immediately:")
if confirmed_leaks:
    for col in confirmed_leaks:
        print(f"    ✗ {col}")
else:
    print("    None confirmed")

print("\n  SUSPECTED LEAKS — investigate before using:")
if suspected_leaks:
    for col in suspected_leaks:
        print(f"    ? {col}")
else:
    print("    None suspected")

print("\n  SAFE TO USE in model:")
safe_confirmed = [c for c in SAFE_COLS if c in df.columns
                  and c not in confirmed_leaks and c not in suspected_leaks]
for col in safe_confirmed:
    print(f"    ✓ {col}")

# ─────────────────────────────────────────────────────────────
# STEP 8: GENERATE CLEAN FEATURE LIST FOR RETRAINING
# ─────────────────────────────────────────────────────────────
clean_feature_list = [c for c in df.columns
                      if c not in confirmed_leaks
                      and c not in suspected_leaks
                      and c not in ['Application ID', 'Pseudo NRIC',
                                    'Outcome_Code', 'RS Outcome Item Desc',
                                    'RS Submit Datetime', 'RS Submit Date',
                                    'RS Last Status Datetime', 'DOB',
                                    'DOB Est', 'First_Submit', 'Is_Accepted']]

print(f"\n  Clean feature count for retraining: {len(clean_feature_list)}")

# Save all results
results_df.to_csv('leakage_single_feature_check.csv',   index=False)
overlap_df.to_csv('leakage_overlap_check.csv',           index=False)
corr_df.to_csv   ('leakage_correlation_check.csv',       index=False)

pd.DataFrame({
    'Status':  ['Confirmed Leak'] * len(confirmed_leaks) +
               ['Suspected Leak'] * len(suspected_leaks) +
               ['Safe']           * len(safe_confirmed),
    'Feature': confirmed_leaks + suspected_leaks + safe_confirmed,
}).to_csv('leakage_verdict.csv', index=False)

pd.DataFrame({'Clean_Feature': clean_feature_list}).to_csv(
    'clean_feature_list.csv', index=False
)

print("\nFiles saved:")
print("  → leakage_verdict.csv              (confirmed / suspected / safe)")
print("  → leakage_single_feature_check.csv (solo accuracy per feature)")
print("  → leakage_overlap_check.csv        (categorical overlap analysis)")
print("  → leakage_correlation_check.csv    (numerical correlation scores)")
print("  → clean_feature_list.csv           (safe features for retraining)")