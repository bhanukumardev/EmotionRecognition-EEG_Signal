#!/usr/bin/env python3
"""Test model performance - accuracy and F1 score on real EEG data"""

import sys
sys.path.insert(0, 'src')
import process_real_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print('='*70)
print('EEG EMOTION RECOGNITION - MODEL TESTING ON REAL DATA')
print('='*70)

# Load and process data
print("\n[1/4] Loading EEG data from emotions.csv...")
df = process_real_data.load_and_explore_data('data/real_eeg/emotions.csv')

print("\n[2/4] Preprocessing data...")
X, y, label_encoder = process_real_data.preprocess_data(df)

print("\n[3/4] Extracting EEG features...")
X_selected, feature_cols = process_real_data.extract_eeg_features(X)

# Split data
print("\n[4/4] Splitting data (80/20 train/test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Feature dimensions: {X_selected.shape[1]}")
print(f"Classes: {label_encoder.classes_}")

# Train models
print("\n" + "="*70)
print("TRAINING MODELS...")
print("="*70)
results, X_test_scaled = process_real_data.train_models(X_train, X_test, y_train, y_test, label_encoder)

# Display results
print("\n" + "="*70)
print("RANDOM FOREST - RESULTS")
print("="*70)

y_pred_rf = results['rf']['predictions']
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
rec_rf = recall_score(y_test, y_pred_rf, average='weighted')

print(f"Accuracy:  {acc_rf:.4f}  ({acc_rf*100:.2f}%)")
print(f"F1-Score:  {f1_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall:    {rec_rf:.4f}")

print("\n" + "="*70)
print("SVM - RESULTS")
print("="*70)

y_pred_svm = results['svm']['predictions']
acc_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
prec_svm = precision_score(y_test, y_pred_svm, average='weighted')
rec_svm = recall_score(y_test, y_pred_svm, average='weighted')

print(f"Accuracy:  {acc_svm:.4f}  ({acc_svm*100:.2f}%)")
print(f"F1-Score:  {f1_svm:.4f}")
print(f"Precision: {prec_svm:.4f}")
print(f"Recall:    {rec_svm:.4f}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

if acc_rf > acc_svm:
    winner = "Random Forest"
    winner_acc = acc_rf
else:
    winner = "SVM"
    winner_acc = acc_svm

print(f"\nRandom Forest Accuracy: {acc_rf*100:.2f}%")
print(f"SVM Accuracy:           {acc_svm*100:.2f}%")
print(f"\n⭐ Winner: {winner} with accuracy of {winner_acc*100:.2f}%")
print("="*70)
