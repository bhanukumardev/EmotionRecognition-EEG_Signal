"""
EEG Emotion Recognition: Model Evaluation
Generate evaluation metrics and visualizations
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import os


def evaluate_model():
    """Evaluate trained models on test data."""
    print("="*60)
    print("EEG EMOTION RECOGNITION - MODEL EVALUATION")
    print("="*60)
    
    # Load test data
    X_test = np.load('/app/eeg_emotion_recognition_1806/data/processed/X_test_features.npy')
    y_test = np.load('/app/eeg_emotion_recognition_1806/data/processed/y_test.npy')
    
    # Load model info
    with open('/app/eeg_emotion_recognition_1806/model/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    emotions = model_info['emotions']
    
    print(f"\nTest samples: {len(y_test)}")
    print(f"Emotions: {emotions}")
    print(f"Feature dimensions: {X_test.shape[1]}")
    
    # Load models
    model_rf = joblib.load('/app/eeg_emotion_recognition_1806/model/emotion_model_rf.pkl')
    scaler_rf = joblib.load('/app/eeg_emotion_recognition_1806/model/scaler_rf.pkl')
    
    model_svm = joblib.load('/app/eeg_emotion_recognition_1806/model/emotion_model_svm.pkl')
    scaler_svm = joblib.load('/app/eeg_emotion_recognition_1806/model/scaler_svm.pkl')
    
    # Scale test data
    X_test_scaled_rf = scaler_rf.transform(X_test)
    X_test_scaled_svm = scaler_svm.transform(X_test)
    
    # Get predictions
    y_pred_rf = model_rf.predict(X_test_scaled_rf)
    y_pred_svm = model_svm.predict(X_test_scaled_svm)
    
    # Get probabilities (only RF has predict_proba)
    y_prob_rf = model_rf.predict_proba(X_test_scaled_rf)
    
    # Calculate accuracies
    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    
    print(f"\n{'='*60}")
    print("ACCURACY RESULTS")
    print(f"{'='*60}")
    print(f"Random Forest Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")
    print(f"SVM Accuracy: {acc_svm:.4f} ({acc_svm*100:.2f}%)")
    
    # Classification reports
    print(f"\n{'='*60}")
    print("RANDOM FOREST - CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred_rf, target_names=emotions))
    
    print(f"\n{'='*60}")
    print("SVM - CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred_svm, target_names=emotions))
    
    # Confusion matrices
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions, ax=axes[0])
    axes[0].set_title('Random Forest - Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens',
                xticklabels=emotions, yticklabels=emotions, ax=axes[1])
    axes[1].set_title('SVM - Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('/app/eeg_emotion_recognition_1806/results/confusion_matrices.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrices saved to results/confusion_matrices.png")
    
    # ROC Curves (only for Random Forest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'green', 'red']
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_rf[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{emotions[i]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Random Forest - ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('/app/eeg_emotion_recognition_1806/results/roc_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ ROC curves saved to results/roc_curves.png")
    
    # Feature importance (Random Forest)
    importances = model_rf.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(20), importances[indices], align='center')
    ax.set_yticks(range(20))
    ax.set_yticklabels([f'Feature {i}' for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('/app/eeg_emotion_recognition_1806/results/feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"✓ Feature importance saved to results/feature_importance.png")
    
    # Calculate per-class metrics
    precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average=None)
    precision_svm, recall_svm, f1_svm, _ = precision_recall_fscore_support(y_test, y_pred_svm, average=None)
    
    # Summary table
    print(f"\n{'='*60}")
    print("PER-CLASS METRICS")
    print(f"{'='*60}")
    print(f"{'Emotion':<12} {'RF Prec':<10} {'RF Rec':<10} {'RF F1':<10} {'SVM Prec':<10} {'SVM Rec':<10} {'SVM F1':<10}")
    print("-" * 82)
    for i, emotion in enumerate(emotions):
        print(f"{emotion:<12} {precision_rf[i]:<10.4f} {recall_rf[i]:<10.4f} {f1_rf[i]:<10.4f} "
              f"{precision_svm[i]:<10.4f} {recall_svm[i]:<10.4f} {f1_svm[i]:<10.4f}")
    
    # Macro averages
    macro_f1_rf = np.mean(f1_rf)
    macro_f1_svm = np.mean(f1_svm)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Random Forest:")
    print(f"  - Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")
    print(f"  - Macro F1: {macro_f1_rf:.4f}")
    print(f"\nSVM:")
    print(f"  - Accuracy: {acc_svm:.4f} ({acc_svm*100:.2f}%)")
    print(f"  - Macro F1: {macro_f1_svm:.4f}")
    
    # Save evaluation metrics
    evaluation_results = {
        'random_forest': {
            'accuracy': float(acc_rf),
            'precision': precision_rf.tolist(),
            'recall': recall_rf.tolist(),
            'f1_score': f1_rf.tolist(),
            'macro_f1': float(macro_f1_rf),
            'confusion_matrix': cm_rf.tolist()
        },
        'svm': {
            'accuracy': float(acc_svm),
            'precision': precision_svm.tolist(),
            'recall': recall_svm.tolist(),
            'f1_score': f1_svm.tolist(),
            'macro_f1': float(macro_f1_svm),
            'confusion_matrix': cm_svm.tolist()
        },
        'emotions': emotions,
        'test_samples': len(y_test)
    }
    
    with open('/app/eeg_emotion_recognition_1806/results/evaluation_metrics.pkl', 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    print(f"\n✓ Evaluation metrics saved to results/evaluation_metrics.pkl")
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    return evaluation_results


if __name__ == '__main__':
    evaluate_model()
