# ============================================
# 🩺 MedGuard – XGBoost Symptom Checker
# Author: Shyam Ji Srivastava
# ============================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt

# 1️⃣ Load Dataset
data_path = './data/Training.csv'  # make sure Training.csv is in the 'data' folder
df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2️⃣ Encode Target Labels (diagnosis column)
le = LabelEncoder()
df['prognosis'] = le.fit_transform(df['prognosis'])

# 3️⃣ Split features and labels
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# 4️⃣ Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 5️⃣ Train XGBoost model
model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)

print("\n🚀 Training model...")
model.fit(X_train, y_train)
print("✅ Model training complete!")

# 6️⃣ Evaluate performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed classification report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 7️⃣ SHAP Explainability
print("\n🔍 Calculating SHAP values for explainability...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)





print("✅ SHAP values calculated. Displaying symptom importance plot...")
shap.summary_plot(shap_values, X_test)
plt.show()

print("\n✅ MedGuard training and explainability completed successfully!")
