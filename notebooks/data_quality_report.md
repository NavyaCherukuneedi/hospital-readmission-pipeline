# Hospital Readmission Data - Quality Analysis Findings

**Date:** November 5, 2024  
**Dataset:** Hospital Readmission Data  
**Records:** 25,000  
**Features:** 17

---

## Key Questions Analysis

### 1. Missing Data: Which columns have missing values? What percentage?

#### Answer: NO MISSING VALUES detected!

- All 25,000 rows are complete
- No NULL values in any column
- **However:** "Missing" appears as a categorical value in `medical_specialty` (not as NULL)
  - Missing specialty: **12,382 records (49.53%)**
  - This is stored as the string "Missing", not a NULL value

#### Impact: Medium

- Need to handle the "Missing" category in medical_specialty during transformation

---

### 2. Outliers: Are there unrealistic values?

#### Answer: YES - Several columns have outliers detected using IQR method

#### Outliers by Column:

| Column           | Count | Percentage | Range     | Assessment            |
| ---------------- | ----- | ---------- | --------- | --------------------- |
| time_in_hospital | 580   | 2.32%      | 1-14 days | Realistic             |
| n_lab_procedures | 33    | 0.13%      | 1-113     | Realistic             |
| n_procedures     | 1,227 | 4.91%      | 0-6       | Realistic             |
| n_medications    | 844   | 3.38%      | 1-79      | Some high values      |
| n_outpatient     | 4,141 | 16.56%     | 0-33      | Some high values      |
| n_inpatient      | 1,628 | 6.51%      | 0-15      | Realistic             |
| n_emergency      | 2,728 | 10.91%     | 0-64      | Some very high values |

#### Impact: Low

- Most outliers are realistic (medical data naturally varies)
- Values are within logical bounds
- No need to remove; may be important for predictions

---

### 3. Imbalanced Classes: Is the target variable balanced?

#### Answer: SLIGHTLY IMBALANCED but reasonably balanced

#### Readmission Distribution:

- **Not Readmitted (no):** 13,246 (52.98%)
- **Readmitted (yes):** 11,754 (47.02%)
- **Readmission Rate:** 47.02%
- **Class Balance Ratio:** 1.13:1

#### Impact: Low

- Classes are close enough (only ~5% skew toward "no")
- No need for oversampling or class weights
- Stratified split will be important for train/test

---

### 4. Data Types: Are columns the right type?

#### Answer: NEEDS CONVERSION

#### Current Issues:

| Column            | Current Type | Should Be   | Issue                      | Priority |
| ----------------- | ------------ | ----------- | -------------------------- | -------- |
| age               | String       | Numeric     | Stored as "[70-80)" format | HIGH     |
| glucose_test      | String       | Categorical | Inconsistent values        | HIGH     |
| A1Ctest           | String       | Categorical | Inconsistent values        | HIGH     |
| change            | String       | Boolean     | Should be 0/1              | MEDIUM   |
| diabetes_med      | String       | Boolean     | Should be 0/1              | MEDIUM   |
| readmitted        | String       | Boolean     | Should be 0/1              | MEDIUM   |
| medical_specialty | String       | Categorical | Needs encoding             | MEDIUM   |
| diag_1            | String       | Categorical | Needs encoding             | MEDIUM   |
| diag_2            | String       | Categorical | Needs encoding             | MEDIUM   |
| diag_3            | String       | Categorical | Needs encoding             | MEDIUM   |

#### Impact: Medium

- Need to convert string columns to appropriate types for machine learning

---

### 5. Categorical Issues: Are there inconsistencies?

#### Answer: YES - Several inconsistencies found

#### Issue 1: Medical Specialty (7 unique values)

```
Missing:                 12,382 (49.53%)
InternalMedicine:        3,565 (14.26%)
Other:                   2,664 (10.66%)
Emergency/Trauma:        1,885 (7.54%)
Family/GeneralPractice:  1,882 (7.53%)
Cardiology:              1,436 (5.74%)
Orthopedics:             1,186 (4.74%)
```

**Problem:** "Missing" as explicit value represents data collection gaps

#### Issue 2: Glucose Test (3 unique values - INCONSISTENT!)

```
Expected: "yes" or "no"
Actual:   "no" (23,625), "normal" (689), "high" (686)

Problem: No "yes" values - test results stored instead!
- 94.5% marked as "no"
- 2.75% marked as "normal"
- 2.74% marked as "high"
```

#### Issue 3: A1C Test (3 unique values - INCONSISTENT!)

```
Expected: "yes" or "no"
Actual:   "no" (20,938), "high" (2,827), "normal" (1,235)

Problem: No "yes" values - test results stored instead!
- 83.75% marked as "no"
- 11.31% marked as "high"
- 4.94% marked as "normal"
```

#### Issue 4: Diagnosis Columns (8 categories each)

```
"Other" is vague:
- diag_1: 6,498 (25.99%)
- diag_2: 9,056 (36.22%)
- diag_3: 9,107 (36.43%)

"Missing" in diag_1: 4 records only
```

#### Impact: High

- Need to handle 49.53% "Missing" specialty
- Convert glucose/A1C test results to proper format
- Consolidate "Other" categories if possible

---

### 6. Feature Relationships: Which features correlate with readmission?

#### Answer: IDENTIFIED KEY PREDICTORS

#### Correlation with Readmission (sorted by strength):

| Rank | Feature          | Correlation | Interpretation                                              |
| ---- | ---------------- | ----------- | ----------------------------------------------------------- |
| 1    | n_inpatient      | 0.2125      | **STRONGEST** - Previous inpatient visits highly predictive |
| 2    | n_outpatient     | 0.0955      | Previous outpatient visits moderately predictive            |
| 3    | n_emergency      | 0.0935      | Emergency visit history moderately predictive               |
| 4    | time_in_hospital | 0.0431      | Current stay length weakly predictive                       |
| 5    | n_medications    | 0.0369      | More medications weakly predictive                          |
| 6    | n_lab_procedures | 0.0330      | Lab procedures weakly predictive                            |
| 7    | n_procedures     | -0.0444     | Negative: More procedures = slightly lower risk             |

#### Key Insights:

**By Age Group:**

```
[40-50):   44.5% readmitted
[50-60):   44.2% readmitted
[60-70):   46.9% readmitted
[70-80):   48.8% readmitted
[80-90):   49.6% readmitted (HIGHEST)
[90-100):  42.1% readmitted (drops - survivor bias?)
```

**Trend:** Readmission increases with age up to 80-90 years

**By Hospital Stay Duration:**

```
0-3 days:    44.3% readmitted
3-7 days:    49.3% readmitted (+5%)
7-14 days:   49.7% readmitted (+5.4%)
```

**Trend:** Longer stays correlate with higher readmission risk

---

## Summary of Data Quality Issues

### High Priority Issues ✗

1. **Medical specialty 49.53% "Missing" values**
2. **Glucose/A1C tests show results instead of yes/no**
3. **Age stored as string ranges, not numeric**
4. **Yes/no columns stored as strings**

### Medium Priority Issues ⚠

1. "Other" category vague in diagnoses
2. Multiple data type conversions needed
3. Outliers in n_medications, n_outpatient, n_emergency (verify validity)

### Low Priority Issues ℹ

1. Class imbalance is minimal (47% vs 53%)
2. Outliers appear realistic, probably keep

---

## Recommendations for Transformation

### Must Do

- [ ] Convert `age` from "[70-80)" to numeric midpoint (75)
- [ ] Convert yes/no columns to boolean (0/1)
- [ ] Handle medical_specialty "Missing" values (drop, impute, or flag)
- [ ] Standardize glucose_test and A1Ctest (create ordinal encoding)

### Should Do

- [ ] Create numeric features from test results
- [ ] Consolidate "Other" categories if possible
- [ ] Create interaction features (e.g., age × medical_specialty)
- [ ] Normalize/scale numeric features

### Could Do

- [ ] Create derived features (e.g., total_visits = n_outpatient + n_inpatient + n_emergency)
- [ ] Create temporal features if data available
- [ ] Handle age [90-100) separately (small sample size: 750 records)

---
