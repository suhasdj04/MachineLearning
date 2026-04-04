# =============================================
# train_and_save_models.py
# Run this ONCE to train and save all models
# =============================================

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ── 1. Load dataset ──────────────────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# Place heart.csv in the same folder as this file
df = pd.read_csv("heart.csv")

# ── 2. Outlier removal (IQR) – matches your notebook ─────────────────────────
Q1  = df.quantile(0.25)
Q3  = df.quantile(0.75)
IQR = Q3 - Q1
df  = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# ── 3. Split ──────────────────────────────────────────────────────────────────
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. Train all 4 models ─────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "SVM":                 SVC(probability=True, kernel="rbf"),   # probability=True for predict_proba
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_sc))
    accuracies[name] = round(acc * 100, 2)
    print(f"{name}: {acc:.4f}")

# ── 6. Save models + scaler + accuracies ─────────────────────────────────────
with open("models/scaler.pkl",      "wb") as f: pickle.dump(scaler,      f)
with open("models/logistic.pkl",    "wb") as f: pickle.dump(models["Logistic Regression"], f)
with open("models/random_forest.pkl","wb") as f: pickle.dump(models["Random Forest"],      f)
with open("models/knn.pkl",         "wb") as f: pickle.dump(models["KNN"],                 f)
with open("models/svm.pkl",         "wb") as f: pickle.dump(models["SVM"],                 f)
with open("models/accuracies.json", "w")  as f: json.dump(accuracies, f)

print("\n✅ All models saved to /models folder!")
print("Accuracies:", accuracies)
