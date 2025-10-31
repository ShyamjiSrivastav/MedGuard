# ============================================
# 🩺 MedGuard – SHAP Explainability Script (Booster Compatible + Multi-class Safe)
# Works with: XGBoost Booster (.json) model
# Author: Shyam Ji Srivastava
# ============================================

import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
import os

# ✅ Use headless Matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------
# 1️⃣ Load Booster model and label encoder
# --------------------------------------------
print("🔹 Loading saved Booster model and label encoder...")

booster_path = "medguard_booster.json"
encoder_path = "label_encoder.pkl"

if not os.path.exists(booster_path):
    raise FileNotFoundError("❌ Booster model not found! Train the model to generate 'medguard_booster.json'.")

if not os.path.exists(encoder_path):
    raise FileNotFoundError("❌ Label encoder not found! Re-save label encoder file.")

# Load Booster and encoder
booster = xgb.Booster()
booster.load_model(booster_path)
le = joblib.load(encoder_path)
print("✅ Booster and Label Encoder loaded successfully!")

# --------------------------------------------
# 2️⃣ Load dataset
# --------------------------------------------
print("📂 Loading dataset...")
df = pd.read_csv("./data/Training.csv")
X = df.drop("prognosis", axis=1)
y = df["prognosis"]
print(f"✅ Dataset loaded successfully! Shape: {X.shape}")

# --------------------------------------------
# 3️⃣ Convert Booster → SHAP-compatible model
# --------------------------------------------
print("🧠 Initializing SHAP TreeExplainer for Booster...")
model = xgb.XGBClassifier()
model._Booster = booster  # mimic sklearn wrapper
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
print("✅ SHAP values computed successfully!")

# --------------------------------------------
# 4️⃣ Handle Multi-class SHAP properly
# --------------------------------------------
if isinstance(shap_values, list):
    print(f"📘 Detected multi-class model with {len(shap_values)} disease classes.")
    # Take mean absolute SHAP across classes for visualization
    shap_values_mean = np.mean([np.abs(vals) for vals in shap_values], axis=0)
else:
    shap_values_mean = shap_values

# --------------------------------------------
# 5️⃣ GLOBAL EXPLANATION (Clean Blue Plot)
# --------------------------------------------
print("📊 Generating Clean Global SHAP Feature Importance Plot...")
mean_abs_shap = np.abs(shap_values_mean).mean(axis=0)
importance_df = pd.DataFrame({
    "Symptom": X.columns,
    "Mean_SHAP_Value": mean_abs_shap
}).sort_values(by="Mean_SHAP_Value", ascending=True)

plt.figure(figsize=(8, 8))
plt.barh(importance_df["Symptom"], importance_df["Mean_SHAP_Value"], color="#4C72B0")
plt.title("Global Symptom Importance", fontsize=13, fontweight="bold", pad=10)
plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output Magnitude)", fontsize=11)
plt.grid(alpha=0.3, linestyle="--", linewidth=0.6, axis="x")
plt.tight_layout()
plt.savefig("shap_global_importance.png", dpi=400, bbox_inches="tight", transparent=False)
plt.close()
print("✅ Saved clean global SHAP bar chart (shap_global_importance.png).")

# --------------------------------------------
# 6️⃣ LOCAL EXPLANATION (Clean Orange Plot)
# --------------------------------------------
example_index = 0  # Change this to inspect another patient
print(f"🔍 Generating Clean Local Explanation for sample #{example_index} ({y.iloc[example_index]})")

# Handle multiclass SHAP values correctly
if isinstance(shap_values, list):
    local_values = np.mean([np.abs(vals[example_index]) for vals in shap_values], axis=0)
else:
    local_values = np.abs(shap_values[example_index])

# Create clean horizontal bar chart
local_importance_df = pd.DataFrame({
    "Symptom": X.columns,
    "Local_SHAP_Value": local_values
}).sort_values(by="Local_SHAP_Value", ascending=True)

plt.figure(figsize=(8, 8))
plt.barh(local_importance_df["Symptom"], local_importance_df["Local_SHAP_Value"], color="#DD8452")
plt.title("Local Explanation (Based on Example Patient)", fontsize=13, fontweight="bold", pad=10)
plt.xlabel("Mean |SHAP Value| (Impact on This Prediction)", fontsize=11)
plt.grid(alpha=0.3, linestyle="--", linewidth=0.6, axis="x")
plt.tight_layout()
plt.savefig("shap_local_explanation.png", dpi=400, bbox_inches="tight", transparent=False)
plt.close()
print("✅ Saved clean local SHAP bar chart (shap_local_explanation.png).")

# --------------------------------------------
# 7️⃣ SHAP Feature Ranking CSV
# --------------------------------------------
print("📑 Generating SHAP Feature Ranking CSV...")
shap_mean_importance = np.abs(shap_values_mean).mean(axis=0)
importance_df = pd.DataFrame(
    list(zip(X.columns, shap_mean_importance)),
    columns=["Symptom", "Mean_SHAP_Importance"]
).sort_values(by="Mean_SHAP_Importance", ascending=False)

importance_df.to_csv("shap_feature_ranking.csv", index=False)
print("✅ Saved: shap_feature_ranking.csv")

# --------------------------------------------
# ✅ Final Summary
# --------------------------------------------
print("\n🎯 SHAP Explainability generation completed successfully!")
print("🖼️ Files generated:")
print("   - shap_global_importance.png")
print("   - shap_local_explanation.png")
print("   - shap_feature_ranking.csv")

