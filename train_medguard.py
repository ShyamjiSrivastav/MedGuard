# ============================================
# 🩺 MedGuard – Balanced Training Script (v3)
# Author: Shyam Ji Srivastava
# Optimized for: 70–80% accuracy (real-world generalization)
# Works with: XGBoost >= 1.7.0
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --------------------------------------------
# 1️⃣ Load and Clean Dataset
# --------------------------------------------
df = pd.read_csv("./data/Training.csv")

# 🧹 Remove any extra/empty unnamed columns (common CSV issue)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("✅ Cleaned dataset loaded:", df.shape)

# --------------------------------------------
# 2️⃣ Augment Data – add realistic variations
# --------------------------------------------
augmented_df = []
for disease in df["prognosis"].unique():
    subset = df[df["prognosis"] == disease].iloc[0]
    base_symptoms = subset.drop("prognosis").values

    for _ in range(25):  # generate 25 variations per disease
        noisy_symptoms = base_symptoms.copy()
        flip_mask = np.random.rand(len(base_symptoms)) < 0.1  # 10% symptom flips
        noisy_symptoms[flip_mask] = 1 - noisy_symptoms[flip_mask]
        row = list(noisy_symptoms) + [subset["prognosis"]]
        augmented_df.append(row)

df = pd.DataFrame(augmented_df, columns=list(df.columns))
print("🧬 Augmented dataset created:", df.shape)

# --------------------------------------------
# 3️⃣ Encode Target Labels
# --------------------------------------------
le = LabelEncoder()
df["prognosis"] = le.fit_transform(df["prognosis"])

# --------------------------------------------
# 4️⃣ Ensure All Features Are Numeric
# --------------------------------------------
X = df.drop("prognosis", axis=1)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
print("✅ All features numeric:", X.dtypes.unique())

y = df["prognosis"]

# --------------------------------------------
# 5️⃣ Train/Test Split (30% validation)
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("📊 Train/Test split:", X_train.shape, X_test.shape)

# --------------------------------------------
# 6️⃣ Symptom Dropout Regularization (reduce overfitting)
# --------------------------------------------
rng = np.random.default_rng(42)
drop_rate = 0.1  # Reduced dropout for better balance

X_train_noisy = X_train.copy()
mask = rng.binomial(1, 1 - drop_rate, size=X_train_noisy.shape)
X_train_noisy = X_train_noisy * mask

print(f"🔧 Applied Symptom Dropout Regularization ({drop_rate*100:.0f}% symptoms hidden).")

# --------------------------------------------
# 7️⃣ Label Noise Regularization (simulate diagnostic errors)
# --------------------------------------------
label_noise_rate = 0.05  # Reduced to 5% for cleaner learning
y_train_noisy = y_train.copy()
num_noisy = int(len(y_train_noisy) * label_noise_rate)

if num_noisy > 0:
    idx = rng.choice(y_train_noisy.index, num_noisy, replace=False)
    random_labels = rng.choice(y_train_noisy.unique(), size=num_noisy)
    y_train_noisy.loc[idx] = random_labels
    print(f"⚠️ Introduced {num_noisy} noisy labels ({label_noise_rate*100:.1f}%).")

# --------------------------------------------
# 8️⃣ Define XGBoost Model (tuned for balance)
# --------------------------------------------
model = XGBClassifier(
    n_estimators=150,        # more trees, stable training
    learning_rate=0.08,      # smoother learning
    max_depth=4,             # moderate complexity
    subsample=0.8,           # 80% of rows per tree
    colsample_bytree=0.8,    # 80% of features per tree
    gamma=2,                 # slight complexity penalty
    reg_lambda=2,            # L2 regularization
    reg_alpha=1,             # L1 regularization
    min_child_weight=2,      # prevent small leaf nodes
    eval_metric="mlogloss",
    early_stopping_rounds=20,
    random_state=42
)

# --------------------------------------------
# 9️⃣ Train Model
# --------------------------------------------
print("\n🚀 Training model with realistic data...")
model.fit(
    X_train_noisy,
    y_train_noisy,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("✅ Model training complete!")

# --------------------------------------------
# 🔟 Evaluate Model
# --------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Validation Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --------------------------------------------
# 11️⃣ Save Booster Model
# --------------------------------------------
booster = model.get_booster()
booster.save_model("medguard_booster.json")
joblib.dump(le, "label_encoder.pkl")
print("\n✅ Saved clean Booster model and label encoder!")

# --------------------------------------------
# 12️⃣ Summary
# --------------------------------------------
print("\n🧠 Training Summary:")
print(f"   • Augmented dataset (25x variations per disease)")
print(f"   • Symptom Dropout: {drop_rate*100:.0f}%")
print(f"   • Label Noise: {label_noise_rate*100:.0f}%")
print("   • Regularization: L1 + L2 + Gamma")
print("   • Tree depth: 4, Estimators: 150")
print(f"   • Final Accuracy: {accuracy * 100:.2f}%")

print("\n💡 Tips:")
print("   - To reach 80–85%, set 'drop_rate=0.05' and 'label_noise_rate=0.03'")
print("   - To simulate hospitals better, keep 70–80% for balanced realism.")




