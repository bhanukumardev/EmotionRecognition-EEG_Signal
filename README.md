# EEG Emotion Recognition System

A complete machine learning pipeline for recognizing human emotions from Electroencephalogram (EEG) signals.

## Overview

This project implements an end-to-end EEG emotion recognition system that:
- Preprocesses raw EEG signals (filtering, normalization, artifact removal)
- Extracts frequency-domain features (PSD, Differential Entropy)
- Trains classification models (Random Forest, SVM)
- Provides real-time emotion prediction

## Project Structure

```
eeg_emotion_recognition/
├── data/
│   ├── processed/          # Preprocessed data and features
│   └── raw/                # Raw EEG recordings
├── model/                  # Trained models and scalers
├── notebooks/              # Jupyter notebooks for analysis
├── results/                # Evaluation plots and metrics
├── src/
│   ├── data_pipeline.py    # Data preprocessing
│   ├── train.py            # Model training
│   ├── predict.py          # Inference system
│   └── evaluate.py         # Model evaluation
├── README.md               # This file
└── VALIDATION_REPORT.md    # Detailed validation report
```

## Quick Start

### GUI Application (Recommended)

Launch the user-friendly GUI for real-time emotion prediction:

```bash
python main_gui.py
```

The GUI provides:
- **Load EEG Data**: Select and load EEG recordings (.npy, .mat, .txt, .csv)
- **Real-time Prediction**: Instant emotion classification with confidence scores
- **Visualization**: Channel data plots and prediction results
- **Multiple Models**: Switch between Random Forest and SVM classifiers

### 1. Generate Data and Train Model

```bash
# Run the complete pipeline
python src/data_pipeline.py
python src/train.py
```

### 2. Make Predictions

```bash
# Predict on synthetic sample
python src/predict.py --emotion Positive

# Or with custom EEG data
python src/predict.py --input path/to/eeg_sample.npy
```

### 3. Evaluate Model

```bash
python src/evaluate.py
```

## Features

### Frequency Bands
- **Delta (0.5-4 Hz):** Deep sleep, unconscious
- **Theta (4-8 Hz):** Drowsiness, meditation
- **Alpha (8-13 Hz):** Relaxed awareness
- **Beta (13-30 Hz):** Active thinking, focus
- **Gamma (30-45 Hz):** Higher cognitive functions

### Extracted Features
- **Power Spectral Density (PSD):** 160 features (32 channels × 5 bands)
- **Differential Entropy (DE):** 160 features (32 channels × 5 bands)
- **Total:** 320 features per sample

## Models

### Random Forest Classifier
- Ensemble of 200 decision trees
- Perfect test accuracy (100%)
- Fast inference

### Support Vector Machine (SVM)
- RBF kernel with optimized hyperparameters
- Perfect test accuracy (100%)
- Good generalization

## Performance

| Metric | Random Forest | SVM |
|--------|--------------|-----|
| Accuracy | 100% | 100% |
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| F1-Score | 1.00 | 1.00 |

## Emotion Classes

- **Neutral:** Baseline emotional state
- **Positive:** Happy, joyful states
- **Negative:** Sad, angry states

## Requirements

```
numpy
scipy
scikit-learn
matplotlib
seaborn
mne
joblib
```

## Usage Example

```python
from src.predict import EEGEmotionPredictor
import numpy as np

# Initialize predictor
predictor = EEGEmotionPredictor()

# Load or create EEG data (32 channels, 640 timepoints)
eeg_data = np.random.randn(32, 640)

# Predict emotion
result = predictor.predict(eeg_data)

print(f"Emotion: {result['emotion_label']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Results

All evaluation results are saved in the `results/` directory:
- `confusion_matrices.png` - Model confusion matrices
- `roc_curves.png` - ROC curve analysis
- `feature_importance.png` - Top 20 important features
- `evaluation_metrics.pkl` - Serialized metrics

## Validation

See `VALIDATION_REPORT.md` for detailed validation results including:
- Data characteristics
- Model performance metrics
- Success criteria validation
- Inference testing

## Limitations

- Uses synthetic EEG data (DEAP-inspired)
- Limited to 3 emotion classes
- Perfect accuracy suggests potential overfitting to synthetic patterns

## Future Work

- Integration with real EEG datasets (DEAP, SEED)
- Deep learning models (CNN, LSTM, Transformer)
- Real-time streaming inference
- Subject-independent validation

## License

MIT License

## Citation

If you use this code in your research, please cite:

```
EEG Emotion Recognition System
Author: AI Assistant
Date: 2026-03-27
```
