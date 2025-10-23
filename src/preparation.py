"""
Prepration Script 
Author: Saeedeh Nasrin Jamali

- Imports
- Data loading
- Basic inspection (head/info/describe)
- Column selection
- Target mapping
- EDA plots (correlations, distributions, imbalance)
- Missing-value summaries
- Categorical counts
- Placeholder replacements for '?' in selected columns
- One-hot encoding (dummies)
- Final cleaned dataframe ready for modeling (df_cleaned)

"""

# =========================
# 1) Imports & Setup
# =========================
import pandas as pd
import numpy as np

# =========================
# 2) Load Data
# =========================
import os
import pandas as pd

# load from the 'data' folder relative to this script
data_path = os.path.join("data", "insurance_claims.csv")
rawd = pd.read_csv(data_path)

# =========================
# 3) Quick Peek
# =========================
rawd.head()
rawd.info()
rawd.describe()

# =========================
# 4) Feature Selection 
# =========================
requiredcols = rawd[[
    "months_as_customer","age","policy_csl","policy_deductable","policy_annual_premium","umbrella_limit","insured_sex",
    "insured_education_level","insured_occupation","insured_hobbies","insured_relationship","incident_type","collision_type",
    "incident_severity","authorities_contacted","incident_state","number_of_vehicles_involved","property_damage","bodily_injuries","witnesses","police_report_available","injury_claim", "property_claim","vehicle_claim","auto_make","auto_year","fraud_reported"
]]
df = requiredcols.copy()
df

# =========================
# 5) Target Mapping 
# =========================
df["fraud_reported"] = df["fraud_reported"].map({"Y":1,"N":0})

# =========================
# 6) EDA 
# =========================
# Correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt
corr = df.corr()
plt.figure(figsize=(6,6))
sns.heatmap(corr)
plt.title("Correlation Heatmap")
plt.show()

# Target distribution & numeric summaries
from IPython.display import display
df["fraud_reported"].value_counts().plot(kind="bar")
plt.title("Fraud Reported Distribution")
plt.show()

df["policy_annual_premium"].hist(bins=50)
plt.title("Policy Annual Premium Distribution")
plt.show()

sns.boxplot(x="fraud_reported", y="policy_annual_premium", data=df)
plt.show()

gouped_table = df.groupby("fraud_reported")["policy_annual_premium"].describe()
display(gouped_table)

# Imbalance visualization
ax = sns.countplot(x="fraud_reported", data=df, order=[0, 1], palette="Set2")
ax.set_xticklabels(["No Fraud", "Fraud"])
ax.set_title("Target Balance: Fraud vs Non-Fraud")
ax.set_ylabel("Count")
plt.show()

# Fraud rate by policy_csl
fraud_rate_by_state = df.groupby("policy_csl")["fraud_reported"].mean()
fraud_rate_by_state.plot(kind="bar")

# =========================
# 7) Missingness Summary 
# =========================
df.isna().sum().sort_values(ascending=False)

summary = pd.DataFrame({
    "dtype": df.dtypes,
    "missing": df.isna().sum(),
    "non_null": df.notna().sum,   # kept as-is
    "n_unique": df.nunique()
})
summary
summary = summary.sort_values('dtype')
summary

# =========================
# 8) Categorical Columns & Value Counts 
# =========================
catcols = df.select_dtypes('object').columns
catcols

pdcat = df.value_counts("property_damage", dropna=False)
pdcat

# Replacement of "?" disclosure 
pdreplacement = df["property_damage"].fillna('PropdmgDisclosure').value_counts()
pdreplacement

df.loc[:, "property_damage"] = df["property_damage"].replace({"?","property_damage_disclousure"})
mask = df["property_damage"].astype(str).str.strip().eq("?")
df.loc[mask, "property_damage"] = "property_damage_disclosure"
df.value_counts("property_damage", dropna=False)

df["insured_occupation"].value_counts(dropna=False)
df.value_counts("collision_type", dropna=False)

df.loc[:, "collision_type"] = df["collision_type"].replace({"?","collision_type-disclousure"})
mask = df["collision_type"].astype(str).str.strip().eq("?")
df.loc[mask, "collision_type"] = "collision_type_disclosure"
df.value_counts("collision_type", dropna=False)

# incident_severity
df.value_counts("incident_severity", dropna=False)

# auto_make & property_damage checks
df["auto_make"].value_counts(dropna=False)
df["property_damage"].value_counts(dropna=False)
df["property_damage"] = df["property_damage"].fillna("propertydamagedisclouser")

# re-checks
df["auto_make"].value_counts(dropna=False)
df["property_damage"].value_counts(dropna=False)

# Broad categorical value counts
df["insured_sex"].value_counts(dropna=False)
df["insured_education_level"].value_counts(dropna=False)
df["insured_occupation"].value_counts(dropna=False)
df["insured_hobbies"].value_counts(dropna=False)
df["insured_relationship"].value_counts(dropna=False)
df["incident_type"].value_counts(dropna=False)
df["authorities_contacted"].value_counts(dropna=False)
df["incident_state"].value_counts(dropna=False)
df["police_report_available"].value_counts(dropna=False)

# Specific replacement for police_report_available 
df["police_report_available"].value_counts()
df["police_report_available"] = df["police_report_available"].replace("?", "NoPoliceReportAvailable")
df["police_report_available"].value_counts()

# =========================
# 9) Dummies (kept as-is, including duplicates)
# =========================
dfwithdummies = pd.get_dummies(
    df,
    columns=[
        "insured_sex","insured_education_level",
        "police_report_available","insured_occupation",
        "insured_hobbies","insured_relationship","incident_type",
        "authorities_contacted","incident_state","police_report_available","incident_severity","auto_make","policy_csl","collision_type","number_of_vehicles_involved","property_damage"
    ],
    drop_first=False
)

dfwithdummies["fraud_reported"].value_counts()
dfwithdummies["fraud_reported"].dtype

# =========================
# 10) Final pre-model dataset 
# =========================
dfwithdummies["fraud_reported"] = dfwithdummies["fraud_reported"].astype("int8")
df_cleaned = dfwithdummies
df_cleaned.info()

# (Stop here – model building starts in your main script.)
