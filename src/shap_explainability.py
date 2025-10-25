# ==============================
# SHAP Explainability Section
# ==============================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 1. Create output folder 
os.makedirs("outputs", exist_ok=True)

# 2. Build SHAP explainer and compute SHAP values on test data
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(Xte)

# 3. Beeswarm plot (global feature impact + direction)
plt.figure(facecolor='white')
shap.summary_plot(
    shap_values,
    Xte,
    feature_names=x.columns,
    show=False
)
plt.gcf().set_facecolor('white')
plt.tight_layout()
plt.savefig(
    "outputs/shap_summary_beeswarm.png",
    dpi=300,
    bbox_inches="tight",
    facecolor='white'
)
plt.close()
print("Saved: outputs/shap_summary_beeswarm.png")

# 4. Bar plot (ranked mean|SHAP| importance per feature)
plt.figure(facecolor='white')
shap.summary_plot(
    shap_values,
    Xte,
    feature_names=x.columns,
    plot_type="bar",
    show=False
)
plt.gcf().set_facecolor('white')
plt.tight_layout()
plt.savefig(
    "outputs/shap_summary_bar.png",
    dpi=300,
    bbox_inches="tight",
    facecolor='white'
)
plt.close()
print("Saved: outputs/shap_summary_bar.png")

# 5. Export numeric importance for documentation / README
shap_df = pd.DataFrame(shap_values, columns=x.columns)
mean_abs_importance = (
    np.abs(shap_df)
    .mean()
    .sort_values(ascending=False)
)

top_feats = mean_abs_importance.head(20)
top_feats.to_csv(
    "outputs/shap_top_features.csv",
    header=["mean_abs_SHAP"]
)
print("Saved: outputs/shap_top_features.csv")
print(top_feats.head(10))

