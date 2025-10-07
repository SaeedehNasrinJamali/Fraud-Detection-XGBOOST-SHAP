#!/usr/bin/env python
# coding: utf-8

# In[142]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('pip', 'install xgboost')


# In[143]:


rawd=pd.read_csv("C:\\Users\\LENOVO\\Desktop\\AUTO INSURANCE\\insurance_claims.csv")


# In[144]:


rawd.head()


# In[110]:


rawd.info()


# In[145]:


rawd.describe()


# In[146]:





# In[147]:


requiredcols=rawd[["months_as_customer","age","policy_csl","policy_deductable","policy_annual_premium","umbrella_limit","insured_sex",
               "insured_education_level","insured_occupation","insured_hobbies","insured_relationship","incident_type","collision_type",
                "incident_severity","authorities_contacted","incident_state","number_of_vehicles_involved","property_damage","bodily_injuries","witnesses","police_report_available","injury_claim", "property_claim","vehicle_claim","auto_make","auto_year","fraud_reported"]]
df=requiredcols.copy()
df


# In[148]:


df["fraud_reported"]= df["fraud_reported"].map({"Y":1,"N":0})


# In[28]:





# In[149]:



#EDA
import seaborn as sns
import matplotlib.pyplot as plt
corr=df.corr()
plt.figure(figsize=(6
                    ,6))                  
sns.heatmap(corr)
plt.title("Correlation Heatmap")
plt.show()


# In[150]:


from IPython.display import display
#Target distribution
df["fraud_reported"].value_counts().plot(kind="bar")
plt.title("Fraud Reported Distribution")
plt.show()
#Numeric feature distributions
df["policy_annual_premium"].hist(bins=50)
plt.title("Policy Annual Premium Distribution")
plt.show()
#Boxplot by target
sns.boxplot(x="fraud_reported",y="policy_annual_premium",data=df)
plt.show()
gouped_table=df.groupby("fraud_reported")["policy_annual_premium"].describe()
display(gouped_table)


#imbalanceness
ax = sns.countplot(x="fraud_reported", data=df, order=[0, 1], palette="Set2")
ax.set_xticklabels(["No Fraud", "Fraud"])
ax.set_title("Target Balance: Fraud vs Non-Fraud")
ax.set_ylabel("Count")
plt.show()
#fraud rate
fraud_rate_by_state = df.groupby("policy_csl")["fraud_reported"].mean()
fraud_rate_by_state.plot(kind="bar")


# In[151]:


df.isna().sum().sort_values(ascending=False)


# In[152]:


summary=pd.DataFrame({"dtype":df.dtypes,"missing":df.isna().sum(),"non_null":df.notna().sum,"n_unique":df.nunique()})
summary


# In[153]:


summary=summary.sort_values('dtype')
summary


# In[154]:


catcols=df.select_dtypes('object').columns  # all categorical columns
catcols


# In[155]:


pdcat=df.value_counts("property_damage",dropna=False)
pdcat
#replacement of "?" disclosure
pdreplacement=df["property_damage"].fillna('PropdmgDisclosure').value_counts()
pdreplacement

df.loc[:, "property_damage"]=df["property_damage"].replace({"?","property_damage_disclousure"})
mask = df["property_damage"].astype(str).str.strip().eq("?")
df.loc[mask, "property_damage"] = "property_damage_disclosure"
df.value_counts("property_damage",dropna=False)


# In[156]:


df["insured_occupation"].value_counts(dropna=False)


# In[157]:


df.value_counts("collision_type",dropna=False)


# In[159]:



df.loc[:, "collision_type"]=df["collision_type"].replace({"?","collision_type-disclousure"})
mask = df["collision_type"].astype(str).str.strip().eq("?")
df.loc[mask, "collision_type"] = "collision_type_disclosure"
df.value_counts("collision_type",dropna=False)


# In[160]:


#incident_severity
df.value_counts("incident_severity",dropna=False)


# In[103]:


#auto_make
df["auto_make"].value_counts(dropna=False)
#property_damage
df["property_damage"].value_counts(dropna=False)
df["property_damage"]=df["property_damage"].fillna("propertydamagedisclouser")


# In[161]:


#auto_make
df["auto_make"].value_counts(dropna=False)
df["property_damage"].value_counts(dropna=False)


# In[162]:



df["insured_sex"].value_counts(dropna=False)

df["insured_education_level"].value_counts(dropna=False)

df["insured_occupation"].value_counts(dropna=False)

df["insured_hobbies"].value_counts(dropna=False)

df["insured_relationship"].value_counts(dropna=False)
df["incident_type"].value_counts(dropna=False)
#authorities_contacted
df["authorities_contacted"].value_counts(dropna=False)
#incident_state
df["incident_state"].value_counts(dropna=False)
df["police_report_available"].value_counts(dropna=False)


# In[163]:


df["police_report_available"].value_counts()
df["police_report_available"] = df["police_report_available"].replace("?", "NoPoliceReportAvailable")


# In[164]:


df["police_report_available"].value_counts()


# In[165]:


dfwithdummies=pd.get_dummies(df,columns=["insured_sex","insured_education_level",
                              "police_report_available","insured_occupation",
                              "insured_hobbies","insured_relationship","incident_type",
                             "authorities_contacted","incident_state","police_report_available","incident_severity","auto_make","policy_csl","collision_type","number_of_vehicles_involved","property_damage"],drop_first=False)


# In[166]:


dfwithdummies["fraud_reported"].value_counts()


# In[167]:


dfwithdummies["fraud_reported"].dtype


# In[174]:


dfwithdummies["fraud_reported"]=dfwithdummies["fraud_reported"].astype("int8")
df_cleaned=dfwithdummies
df_cleaned.info()


# In[169]:


#model building


# In[180]:


from sklearn.model_selection import train_test_split


# In[181]:


x=df_cleaned.drop(columns=["fraud_reported"])
y=df_cleaned["fraud_reported"]
# 2) Sanity checks — no objects allowed
assert x.select_dtypes(include="object").empty, X.select_dtypes("object").columns.tolist()


# In[190]:


# making test and train matrices
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
# check on the split (just in case)
assert X_train.select_dtypes(include="object").empty
print(type(y_train), y_train.shape, getattr(y_train, "ndim", None))
print(y_train.head())
print("X_train:", type(X_train), getattr(X_train, "shape", None))
print("y_train:", type(y_train), y_train.shape)


# In[195]:


from xgboost import XGBClassifier
import numpy as np

Xtr = X_train.to_numpy(dtype=np.float32, copy=True)
ytr = y_train.to_numpy()
xgb_model = XGBClassifier(
    learning_rate=0.05, n_estimators=2000, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss", tree_method="hist"
)

xgb_model.fit(Xtr, ytr) 


# In[199]:


import numpy as np

Xtr = X_train.to_numpy(dtype=np.float32, copy=True)
Xte = X_test.to_numpy(dtype=np.float32, copy=True)

from xgboost import XGBClassifier
import numpy as np

Xtr = X_train.to_numpy(dtype=np.float32, copy=True)
ytr = y_train.to_numpy()
xgb_model = XGBClassifier(
    learning_rate=0.05, n_estimators=2000, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss", tree_method="hist"
)

xgb_model.fit(Xtr, ytr) 



# In[202]:


# Predict
y_pred = xgb_model.predict(Xte)
y_pred_prob = xgb_model.predict_proba(Xte)[:, 1]


# In[204]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))
RocCurveDisplay.from_predictions(y_test, y_pred_prob)












