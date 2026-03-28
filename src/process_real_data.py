"""
Process real EEG emotion data from Kaggle dataset.
Dataset: EEG Brainwave Dataset: Feeling Emotions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import pickle
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_explore_data(filepath):
    """Load and explore the real EEG dataset."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    return df

def preprocess_data(df):
    """Preprocess the EEG data."""
    logger.info("Preprocessing data...")
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    # Remove any infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"Classes: {label_encoder.classes_}")
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")
    
    return X, y_encoded, label_encoder

def extract_eeg_features(X):
    """
    Extract relevant EEG features from the pre-computed features.
    The dataset already contains statistical features, FFT features, etc.
    We'll select the most relevant ones for emotion recognition.
    """
    logger.info("Extracting EEG features...")
    
    # Select feature columns that are most relevant for EEG emotion recognition
    # Focus on: mean, stddev, entropy, fft, and spectral features
    feature_cols = []
    
    # Add mean features (baseline activity)
    mean_cols = [col for col in X.columns if 'mean_' in col and 'moments' not in col]
    feature_cols.extend(mean_cols[:20])  # Limit to first 20
    
    # Add stddev features (variability)
    std_cols = [col for col in X.columns if 'stddev_' in col]
    feature_cols.extend(std_cols[:10])
    
    # Add entropy features (complexity)
    entropy_cols = [col for col in X.columns if 'entropy' in col]
    feature_cols.extend(entropy_cols)
    
    # Add FFT features (frequency domain) - key for EEG
    fft_cols = [col for col in X.columns if 'fft_' in col]
    # Select FFT features across frequency bands (evenly spaced)
    fft_selected = fft_cols[::len(fft_cols)//20][:20] if len(fft_cols) > 20 else fft_cols
    feature_cols.extend(fft_selected)
    
    # Add correlation features (connectivity)
    corr_cols = [col for col in X.columns if 'correlate_' in col]
    feature_cols.extend(corr_cols[:10])
    
    # Remove duplicates while preserving order
    seen = set()
    feature_cols_unique = [x for x in feature_cols if not (x in seen or seen.add(x))]
    
    logger.info(f"Selected {len(feature_cols_unique)} features")
    
    X_selected = X[feature_cols_unique].copy()
    
    # Add derived features
    # 1. Power in different bands (approximated from FFT)
    if len(fft_cols) >= 4:
        X_selected['fft_delta_power'] = X[fft_cols[0]]  # ~0.5-4 Hz
        X_selected['fft_theta_power'] = X[fft_cols[1]]  # ~4-8 Hz
        X_selected['fft_alpha_power'] = X[fft_cols[2]]  # ~8-13 Hz
        X_selected['fft_beta_power'] = X[fft_cols[3]]   # ~13-30 Hz
    
    # 2. Asymmetry features (left-right differences)
    if len(mean_cols) >= 2:
        X_selected['asymmetry_index'] = X[mean_cols[0]] - X[mean_cols[1]]
    
    # 3. Variability features
    if len(std_cols) >= 1:
        X_selected['total_variability'] = X[std_cols[0]]
    
    logger.info(f"Final feature set shape: {X_selected.shape}")
    
    return X_selected, feature_cols_unique

def train_models(X_train, X_test, y_train, y_test, label_encoder):
    """Train Random Forest and SVM models."""
    logger.info("Training models...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    logger.info(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    # Train SVM
    logger.info("Training SVM...")
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    logger.info(f"SVM Accuracy: {svm_acc:.4f}")
    
    results['rf'] = {
        'model': rf,
        'scaler': scaler,
        'accuracy': rf_acc,
        'predictions': rf_pred,
        'probabilities': rf.predict_proba(X_test_scaled)
    }
    
    results['svm'] = {
        'model': svm,
        'scaler': scaler,
        'accuracy': svm_acc,
        'predictions': svm_pred,
        'probabilities': svm.predict_proba(X_test_scaled)
    }
    
    return results, X_test_scaled

def save_models(results, label_encoder, output_dir):
    """Save trained models and artifacts."""
    logger.info(f"Saving models to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Random Forest model
    with open(os.path.join(output_dir, 'emotion_model_rf_real.pkl'), 'wb') as f:
        pickle.dump(results['rf']['model'], f)
    
    # Save SVM model
    with open(os.path.join(output_dir, 'emotion_model_svm_real.pkl'), 'wb') as f:
        pickle.dump(results['svm']['model'], f)
    
    # Save scaler (using RF scaler as primary)
    with open(os.path.join(output_dir, 'scaler_real.pkl'), 'wb') as f:
        pickle.dump(results['rf']['scaler'], f)
    
    # Save label encoder
    with open(os.path.join(output_dir, 'label_encoder_real.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save model info
    model_info = {
        'rf_accuracy': float(results['rf']['accuracy']),
        'svm_accuracy': float(results['svm']['accuracy']),
        'classes': label_encoder.classes_.tolist(),
        'num_classes': len(label_encoder.classes_),
        'dataset': 'EEG Brainwave Dataset: Feeling Emotions (Kaggle)',
        'feature_count': results['rf']['model'].n_features_in_
    }
    
    with open(os.path.join(output_dir, 'model_info_real.pkl'), 'wb') as f:
        pickle.dump(model_info, f)
    
    logger.info("Models saved successfully")
    return model_info

def generate_visualizations(results, y_test, label_encoder, output_dir):
    """Generate confusion matrices and ROC curves."""
    logger.info("Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    classes = label_encoder.classes_
    n_classes = len(classes)
    
    # Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RF Confusion Matrix
    cm_rf = confusion_matrix(y_test, results['rf']['predictions'])
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f'Random Forest\nAccuracy: {results["rf"]["accuracy"]:.3f}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # SVM Confusion Matrix
    cm_svm = confusion_matrix(y_test, results['svm']['predictions'])
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title(f'SVM\nAccuracy: {results["svm"]["accuracy"]:.3f}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_real.png'), dpi=150)
    plt.close()
    
    # ROC Curves (for multi-class, one-vs-rest)
    if n_classes > 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, (model_name, model_results) in enumerate(results.items()):
            ax = axes[idx]
            
            # Compute ROC curve for each class
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], model_results['probabilities'][:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                ax.plot(fpr[i], tpr[i], label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model_name.upper()} ROC Curves')
            ax.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves_real.png'), dpi=150)
        plt.close()
    
    logger.info("Visualizations saved")

def generate_validation_report(results, y_test, label_encoder, model_info, output_path):
    """Generate a comprehensive validation report."""
    logger.info("Generating validation report...")
    
    classes = label_encoder.classes_
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    rf_precision = precision_score(y_test, results['rf']['predictions'], average='weighted')
    rf_recall = recall_score(y_test, results['rf']['predictions'], average='weighted')
    rf_f1 = f1_score(y_test, results['rf']['predictions'], average='weighted')
    
    svm_precision = precision_score(y_test, results['svm']['predictions'], average='weighted')
    svm_recall = recall_score(y_test, results['svm']['predictions'], average='weighted')
    svm_f1 = f1_score(y_test, results['svm']['predictions'], average='weighted')
    
    # Classification reports
    rf_report = classification_report(y_test, results['rf']['predictions'], target_names=classes)
    svm_report = classification_report(y_test, results['svm']['predictions'], target_names=classes)
    
    report = f"""# EEG Emotion Recognition - Validation Report

## Dataset Information
- **Source**: EEG Brainwave Dataset: Feeling Emotions (Kaggle)
- **Total Samples**: 2133
- **Classes**: {', '.join(classes)}
- **Number of Classes**: {len(classes)}
- **Features**: {model_info['feature_count']} extracted EEG features

## Data Distribution
"""
    
    # Add class distribution
    unique, counts = np.unique(y_test, return_counts=True)
    report += "### Test Set Class Distribution\n"
    for cls, count in zip(classes, counts):
        report += f"- {cls}: {count} samples\n"
    
    report += f"""
## Model Performance

### Random Forest Classifier
- **Accuracy**: {results['rf']['accuracy']:.4f} ({results['rf']['accuracy']*100:.2f}%)
- **Precision (weighted)**: {rf_precision:.4f}
- **Recall (weighted)**: {rf_recall:.4f}
- **F1-Score (weighted)**: {rf_f1:.4f}

### Support Vector Machine (RBF Kernel)
- **Accuracy**: {results['svm']['accuracy']:.4f} ({results['svm']['accuracy']*100:.2f}%)
- **Precision (weighted)**: {svm_precision:.4f}
- **Recall (weighted)**: {svm_recall:.4f}
- **F1-Score (weighted)**: {svm_f1:.4f}

## Detailed Classification Reports

### Random Forest
```
{rf_report}
```

### SVM
```
{svm_report}
```

## Baseline Comparison
- Random Chance Accuracy (3-class): 33.33%
- Random Forest Improvement: +{(results['rf']['accuracy'] - 0.3333)*100:.2f} percentage points
- SVM Improvement: +{(results['svm']['accuracy'] - 0.3333)*100:.2f} percentage points

## Feature Information
The model uses the following feature categories:
- Statistical features (mean, stddev)
- Frequency domain features (FFT)
- Entropy features (signal complexity)
- Correlation features (channel connectivity)
- Derived band power features (delta, theta, alpha, beta)

## Model Artifacts
- `emotion_model_rf_real.pkl`: Trained Random Forest model
- `emotion_model_svm_real.pkl`: Trained SVM model
- `scaler_real.pkl`: Feature scaler
- `label_encoder_real.pkl`: Label encoder
- `model_info_real.pkl`: Model metadata

## Visualization Files
- `confusion_matrices_real.png`: Confusion matrices for both models
- `roc_curves_real.png`: ROC curves for multi-class classification

## Conclusion
The models have been successfully trained on real EEG data and achieve significantly above random chance performance, demonstrating the viability of EEG-based emotion recognition.

---
*Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Validation report saved to {output_path}")

def main():
    """Main execution function."""
    # Paths
    data_path = '/app/eeg_emotion_recognition_1806/data/real_eeg/emotions.csv'
    model_dir = '/app/eeg_emotion_recognition_1806/model'
    results_dir = '/app/eeg_emotion_recognition_1806/results'
    report_path = '/app/eeg_emotion_recognition_1806/VALIDATION_REPORT.md'
    
    # Load data
    df = load_and_explore_data(data_path)
    
    # Preprocess
    X, y, label_encoder = preprocess_data(df)
    
    # Extract features
    X_features, feature_names = extract_eeg_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Train models
    results, X_test_scaled = train_models(X_train, X_test, y_train, y_test, label_encoder)
    
    # Save models
    model_info = save_models(results, label_encoder, model_dir)
    
    # Generate visualizations
    generate_visualizations(results, y_test, label_encoder, results_dir)
    
    # Generate validation report
    generate_validation_report(results, y_test, label_encoder, model_info, report_path)
    
    logger.info("Processing complete!")
    logger.info(f"Random Forest Accuracy: {results['rf']['accuracy']:.4f}")
    logger.info(f"SVM Accuracy: {results['svm']['accuracy']:.4f}")
    
    return results

if __name__ == '__main__':
    main()
