# ============================================
# 🩺 MedGuard Web App – Early Symptom Checker (Stable)
# Author: Shyam Ji Srivastava
# Version: 3.2 (Feature-Aligned + SHAP Safe)
# ============================================

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
import shap

# ✅ Fix for Matplotlib (headless mode)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# --------------------------------------------
# 1️⃣ Load Booster model and Label Encoder
# --------------------------------------------
booster_path = "medguard_booster.json"
encoder_path = "label_encoder.pkl"

if not os.path.exists(booster_path):
    raise FileNotFoundError("❌ Booster model not found! Please train the model first.")

if not os.path.exists(encoder_path):
    raise FileNotFoundError("❌ Label encoder not found! Please re-save it from training.")

print("✅ Loading model and encoder...")
booster = xgb.Booster()
booster.load_model(booster_path)
le = joblib.load(encoder_path)
print("✅ Booster and Label Encoder loaded successfully!")

# --------------------------------------------
# 2️⃣ Load Dataset + Clean Columns (symptom list)
# --------------------------------------------
df = pd.read_csv("./data/Training.csv")

# Remove unnamed or unwanted columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
# Clean column names (remove spaces, special chars)
df.columns = df.columns.str.strip()

# Extract clean symptom list
symptoms = [col for col in df.columns if col != "prognosis"]
print(f"✅ Loaded {len(symptoms)} symptoms for prediction input.")

# --------------------------------------------
# 🏠 3️⃣ Home Page
# --------------------------------------------
@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms)

# --------------------------------------------
# 🔮 4️⃣ Predict Route
# --------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # Step 1️⃣: Capture symptoms selected by user
    selected = request.form.getlist('symptoms')
    input_data = [0] * len(symptoms)
    for s in selected:
        if s in symptoms:
            input_data[symptoms.index(s)] = 1

    # Step 2️⃣: Prepare user input DataFrame
    user_df = pd.DataFrame([input_data], columns=symptoms)

    # ✅ Ensure exact feature match with training booster
    booster_features = booster.feature_names
    if booster_features is None:
        # Safety fallback: use current symptom list
        booster_features = symptoms
    user_df = user_df.reindex(columns=booster_features, fill_value=0)

    # Step 3️⃣: Predict using Booster model
    dmatrix = xgb.DMatrix(user_df)
    pred_probs = booster.predict(dmatrix)
    pred_label = int(np.argmax(pred_probs, axis=1)[0])
    confidence = round(float(np.max(pred_probs) * 100), 2)

    # Decode prediction to disease name
    disease = le.inverse_transform([pred_label])[0]
    print(f"🎯 Predicted Disease: {disease} (Confidence: {confidence}%)")

    # Step 4️⃣: Generate SHAP local explanation safely
    try:
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(user_df)

        # Ensure /static folder exists
        os.makedirs("static", exist_ok=True)

        plt.figure(figsize=(7, 6))
        shap.summary_plot(shap_values, user_df, plot_type="bar", show=False)
        plt.title("Local Explanation (Based on Your Input)", fontsize=12)
        plt.tight_layout()
        shap_local_path = "static/local_user_explanation.png"
        plt.savefig(shap_local_path, dpi=300)
        plt.close()
        print("✅ SHAP local explanation generated successfully.")
    except Exception as e:
        print(f"⚠️ SHAP Error: {e}")
        shap_local_path = "shap_local_explanation.png"

    # Step 5️⃣: Render Results
    return render_template(
        'index.html',
        symptoms=symptoms,
        result=disease,
        confidence=confidence,
        selected=selected,
        shap_global='shap_global_importance.png',
        shap_local='local_user_explanation.png'
    )

# --------------------------------------------
# 🚀 5️⃣ Run Flask App
# --------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)

