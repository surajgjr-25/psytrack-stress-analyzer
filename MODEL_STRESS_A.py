# ---------------------------------------------------------
# MODEL STRESS A – STUDENT STRESS TYPE CLASSIFICATION
# 4 SUPERVISED + 2 UNSUPERVISED (KMeans + PCA)
# ---------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

path = r"E:\suraj aiml\Stress_Dataset.csv"
data = pd.read_csv(path)

print("Dataset loaded:", data.shape)
print(data.head())
print("\nColumns:", data.columns.tolist())

# ---------------------------------------------------------
# 2. BASIC CHECKS
# ---------------------------------------------------------

print("\nMissing values per column:")
print(data.isnull().sum())

# ---------------------------------------------------------
# 3. TARGET COLUMN
# ---------------------------------------------------------
# EXACT name from your screenshot
target_col = "Which type of stress do you primarily experience?"

if target_col not in data.columns:
    print("\nAvailable columns in CSV:")
    print(data.columns.tolist())
    raise ValueError(f"Column '{target_col}' not found in dataset.")

print("\nTarget column:", target_col)
print("Target value counts:")
print(data[target_col].value_counts())

# ---------------------------------------------------------
# 4. FEATURES & TARGET
# ---------------------------------------------------------

X = data.drop(columns=[target_col])
y = data[target_col]

# Keep only numeric features (Gender, Age, Likert scale answers)
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols]

print("\nUsing numeric feature columns:")
print(numeric_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

# ---------------------------------------------------------
# 5. DEFINE 4 SUPERVISED MODELS (PIPELINES)
# ---------------------------------------------------------

pipelines = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            class_weight="balanced"
        ))
    ]),

    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            class_weight="balanced"
        ))
    ]),

    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced"
        ))
    ]),

    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42))
    ]),
}

# ---------------------------------------------------------
# 6. CROSS-VALIDATION COMPARISON
# ---------------------------------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

print("\n--- SUPERVISED MODEL COMPARISON (ACCURACY) ---")
for name, pipe in pipelines.items():
    scores = cross_val_score(
        pipe, X, y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    cv_results[name] = scores
    print(f"\nModel: {name}")
    print("CV accuracies:", np.round(scores, 3))
    print("Mean accuracy:", round(scores.mean(), 3))

# ---------------------------------------------------------
# 7. SELECT BEST BASE MODEL
# ---------------------------------------------------------

best_base_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
print("\nBest base model:", best_base_model_name)

# ---------------------------------------------------------
# 8. HYPERPARAMETER TUNING ON BEST MODEL
# ---------------------------------------------------------

if best_base_model_name == "RandomForest":
    pipe = pipelines["RandomForest"]
    param_distributions = {
        "clf__n_estimators": [200, 300, 400],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None],
    }

elif best_base_model_name == "SVM_RBF":
    pipe = pipelines["SVM_RBF"]
    param_distributions = {
        "clf__C": [0.1, 1, 10, 50],
        "clf__gamma": ["scale", 0.01, 0.001],
    }

elif best_base_model_name == "GradientBoosting":
    pipe = pipelines["GradientBoosting"]
    param_distributions = {
        "clf__n_estimators": [100, 200, 300],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__max_depth": [2, 3, 4],
    }

else:  # LogisticRegression
    pipe = pipelines["LogisticRegression"]
    param_distributions = {
        "clf__C": [0.01, 0.1, 1, 10, 100]
    }

print("\nStarting hyperparameter tuning on:", best_base_model_name)

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=15,
    scoring="accuracy",
    n_jobs=-1,
    cv=cv,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)

print("\nBest hyperparameters:")
print(search.best_params_)
print("Best CV accuracy:", round(search.best_score_, 3))

best_model = search.best_estimator_

# ---------------------------------------------------------
# 9. FINAL TEST EVALUATION
# ---------------------------------------------------------

y_pred = best_model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
print("\nFinal Test Accuracy:", round(test_acc, 3))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix plot
labels_sorted = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels_sorted,
    yticklabels=labels_sorted,
    cmap="YlGnBu"
)
plt.title("Confusion Matrix – Best Student Stress Model")
plt.xlabel("Predicted Stress Type")
plt.ylabel("Actual Stress Type")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. UNSUPERVISED LEARNING (KMEANS + PCA)
# ---------------------------------------------------------

print("\n--- UNSUPERVISED ANALYSIS (KMeans + PCA) ---")

X_unsup = X.copy()

scaler_unsup = StandardScaler()
X_unsup_scaled = scaler_unsup.fit_transform(X_unsup)

# Number of clusters = number of unique stress types
n_clusters = len(np.unique(y))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_unsup_scaled)
data["cluster"] = clusters

print("\nCluster counts:")
print(data["cluster"].value_counts())

# PCA 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_unsup_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=clusters,
    alpha=0.7
)
plt.title("PCA (2D) of Stress Dataset with K-Means Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 11. SAVE BEST MODEL
# ---------------------------------------------------------

joblib.dump(best_model, "best_stress_model.pkl")
print("\nSaved model as best_stress_model.pkl")

# ---------------------------------------------------------
# 12. DEMO PREDICTION
# ---------------------------------------------------------

sample = X.iloc[[0]].copy()
demo_pred = best_model.predict(sample)[0]
print("\nDemo prediction for first row:", demo_pred)
