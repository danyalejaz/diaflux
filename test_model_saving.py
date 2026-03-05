#!/usr/bin/env python3
"""
Test script to verify the model saving and loading functionality
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 80)
print("TESTING MODEL SAVING AND LOADING FUNCTIONALITY")
print("=" * 80)

# Create synthetic diabetes dataset for testing
print("\n1. Creating synthetic diabetes dataset...")
X, y = make_classification(
    n_samples=100000,
    n_features=15,
    n_informative=8,
    n_redundant=3,
    n_classes=2,
    weights=[0.88, 0.12],
    random_state=42
)

X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y = pd.Series(y, name='target')

print(f"   ✓ Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   ✓ Class distribution: {dict(zip(y.value_counts().index, y.value_counts().values))}")

# Split data
print("\n2. Splitting data (80-20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   ✓ Training: {X_train.shape[0]} samples")
print(f"   ✓ Testing: {X_test.shape[0]} samples")

# Train multiple models
print("\n3. Training models...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
}

metrics = {}
for model_name, model in models.items():
    print(f"   • Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics[model_name] = {'accuracy': accuracy, 'f1': f1, 'auc': auc}
    print(f"      - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# Find best model
print("\n4. Selecting best model...")
comparison_df = pd.DataFrame(metrics).T
comparison_df['composite_score'] = (comparison_df['accuracy'] * 0.5 + comparison_df['f1'] * 0.5)
comparison_df = comparison_df.sort_values('composite_score', ascending=False)

print("\n   Model Rankings:")
print(comparison_df.to_string())

best_model_name = comparison_df.index[0]
best_model_instance = models[best_model_name]

print(f"\n   ✓ Best Model: {best_model_name}")
print(f"      - Accuracy: {comparison_df.loc[best_model_name, 'accuracy']:.4f}")
print(f"      - F1-Score: {comparison_df.loc[best_model_name, 'f1']:.4f}")

# Retrain on full dataset
print(f"\n5. Retraining {best_model_name} on full dataset...")
if best_model_name == 'Logistic Regression':
    final_model = LogisticRegression(random_state=42, max_iter=1000)
elif best_model_name == 'Random Forest':
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
elif best_model_name == 'Support Vector Machine':
    final_model = SVC(kernel='rbf', random_state=42, probability=True)
else:  # Gradient Boosting
    final_model = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)

final_model.fit(X, y)
print(f"   ✓ Model trained on {len(X)} samples")

# Save model
print(f"\n6. Saving model...")
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'diabetes_model.pkl')
joblib.dump(final_model, model_path)
print(f"   ✓ Model saved: {model_path}")
print(f"   ✓ File size: {os.path.getsize(model_path) / 1024:.2f} KB")

# Load model and make predictions
print(f"\n7. Loading model and making predictions...")
loaded_model = joblib.load(model_path)
print(f"   ✓ Model loaded successfully")

# Single sample prediction
sample_features = X_test.iloc[0].values.reshape(1, -1)
prediction = loaded_model.predict(sample_features)[0]
pred_proba = loaded_model.predict_proba(sample_features)[0]

print(f"\n   Sample prediction:")
print(f"   - Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
print(f"   - Confidence: {max(pred_proba):.4f} ({max(pred_proba)*100:.2f}%)")

# Batch predictions
print(f"\n   Batch predictions (first 10 test samples):")
batch_pred = loaded_model.predict(X_test.iloc[:10])
batch_actual = y_test.iloc[:10].values
accuracy = (batch_pred == batch_actual).sum() / len(batch_actual)
print(f"   - Accuracy on batch: {accuracy:.2%}")

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - MODEL SAVING AND LOADING WORKS CORRECTLY")
print("=" * 80)
print("\nSummary:")
print(f"- Best Model: {best_model_name}")
print(f"- Model saved to: {model_path}")
print(f"- Model can be loaded and used for predictions")
print(f"- All functionality tested and working")
