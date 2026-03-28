# EEG Emotion Recognition System

A complete **machine learning** pipeline for recognizing human emotions from Electroencephalogram (EEG) signals, built as a 6th semester mini project at KIIT University.

---

## Overview

This project implements an end-to-end EEG emotion recognition system that:

- Preprocesses raw EEG signals (filtering, normalization, artifact removal)
- Extracts frequency-domain features (Power Spectral Density, Differential Entropy)
- Trains classical ML models (Random Forest, SVM)
- Provides a simple GUI for real-time emotion prediction and visualization

It is designed as a learning-oriented framework that can later be extended to real datasets and deep learning models.

---

## Project Structure

```
EmotionRecognition-EEG_Signal/
├── data/
│   ├── raw/              # Raw EEG recordings or synthetic data
│   └── processed/        # Preprocessed data and extracted features
├── model/                # Trained models and scalers (.pkl)
├── notebooks/            # Jupyter notebooks for experiments & EDA
├── results/              # Evaluation plots and serialized metrics
├── src/
│   ├── data_pipeline.py  # Preprocessing & feature extraction
│   ├── train.py          # Model training
│   ├── predict.py        # Inference utilities
│   └── evaluate.py       # Evaluation script
├── main_gui.py           # GUI entry point
├── emotions.csv          # Label mapping / sample metadata
├── requirements.txt      # Python dependencies
├── VALIDATION_REPORT.md  # Detailed validation & testing report
└── README.md             # Project documentation (this file)
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bhanukumardev/EmotionRecognition-EEG_Signal.git
   cd EmotionRecognition-EEG_Signal
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Dependencies (also listed in `requirements.txt`) include: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `mne`, `joblib`.

---

## Quick Start

### 1. Launch the GUI (Recommended)

Run the GUI for interactive, real-time emotion prediction:

```bash
python main_gui.py
```

The GUI allows you to:

- Load EEG data files (`.npy`, `.mat`, `.txt`, `.csv`)
- Run real-time emotion classification with confidence scores
- Visualize channel-wise EEG signals and predictions
- Switch between Random Forest and SVM models

### 2. Run the ML Pipeline from CLI

Generate features and train models:

```bash
# Preprocess data and extract features
python src/data_pipeline.py

# Train models (Random Forest, SVM) on extracted features
python src/train.py
```

Make predictions:

```bash
# Predict synthetic / sample emotion
python src/predict.py --emotion Positive

# Predict using custom EEG input
python src/predict.py --input path/to/eeg_sample.npy
```

Evaluate models:

```bash
python src/evaluate.py
```

---

## Features

### Frequency Bands

The following standard EEG frequency bands are used during feature extraction:

- **Delta (0.5–4 Hz)** – Deep sleep, unconscious states
- **Theta (4–8 Hz)** – Drowsiness, light sleep, meditation
- **Alpha (8–13 Hz)** – Relaxed wakefulness
- **Beta (13–30 Hz)** – Active thinking, concentration
- **Gamma (30–45 Hz)** – Higher-order cognitive processes

### Extracted Features

Per sample, the system computes:

- **Power Spectral Density (PSD):** 160 features (32 channels × 5 bands)
- **Differential Entropy (DE):** 160 features (32 channels × 5 bands)
- **Total:** 320 features per EEG window

---

## Models

### Random Forest

- Ensemble of 200 decision trees
- Very high test accuracy on the synthetic / DEAP-inspired setup
- Fast inference suitable for interactive GUI usage

### Support Vector Machine (SVM)

- RBF kernel with tuned hyperparameters
- Comparable performance to Random Forest on current data
- Good generalization on the synthetic task

> Note: Current results are on synthetic, DEAP-inspired data and may not reflect performance on real-world EEG signals; perfect scores are a strong indicator of overfitting to synthetic patterns.

---

## Performance

Evaluation results (for the current experimental setup):

| Metric     | Random Forest | SVM  |
|-----------|---------------|------|
| Accuracy  | 100%          | 100% |
| Precision | 1.00          | 1.00 |
| Recall    | 1.00          | 1.00 |
| F1-Score  | 1.00          | 1.00 |

All plots and metrics are stored in the `results/` directory:

- `confusion_matrices.png` – Confusion matrices
- `roc_curves.png` – ROC curve analysis
- `feature_importance.png` – Top 20 important features
- `evaluation_metrics.pkl` – Serialized metrics for further analysis

---

## Emotion Classes

The current implementation focuses on three coarse emotion categories:

- **Neutral** – Baseline or calm state
- **Positive** – Happy, joyful, or pleasant states
- **Negative** – Sad, angry, or unpleasant states

You can modify `emotions.csv` and the label pipeline to extend this to more fine-grained classes.

---

## Dataset and Limitations

- Uses synthetic EEG data inspired by popular emotion datasets such as DEAP (no real subject data included in this repository).
- Limited to 3 emotion classes in the current configuration.
- Reported perfect accuracy indicates overfitting to synthetic patterns and should not be interpreted as real-world performance.

Future work aims to address these limitations by introducing real EEG datasets and more robust validation setups.

---

## Future Work

Planned / recommended extensions:

- Integration with real EEG datasets (e.g., DEAP, SEED)
- Deep learning models (CNNs, LSTMs, Transformers)
- Real-time streaming inference from EEG headsets
- Subject-independent and cross-session validation
- More sophisticated artifact removal and channel selection

---

## Usage Example (Python API)

```python
from src.predict import EEGEmotionPredictor
import numpy as np

# Initialize predictor
predictor = EEGEmotionPredictor()

# Create or load EEG data (32 channels, 640 timepoints)
eeg_data = np.random.randn(32, 640)

# Predict emotion
result = predictor.predict(eeg_data)

print(f"Emotion: {result['emotion_label']}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## Validation

For a detailed description of experiments and testing, see `VALIDATION_REPORT.md`, which covers:

- Data characteristics and preprocessing
- Training and evaluation protocols
- Success criteria and achieved metrics
- Inference testing and edge cases

---

## Contributors

- **Bhanu Kumar Dev** (@bhanukumardev)
- **Adarsh Kumar** (@Adarshkumar0509)
- **Aman Sinha** (@amansinha11-dev)
- **Ashish Yadav** (@astreladg)
- **Kanishka** (@catharsis02)
- **Srijan** (@srijanisaDev)
Contributions, feedback, and suggestions are welcome via issues and pull requests.

---

## License

This project is released under the MIT License. See the `LICENSE` file (or this repository's GitHub "License" section) for details.

---

## Citation

If you use this repository in your academic work or projects, you can cite it as:

```
Dev, B. K. (2328162), Kumar, A. (2328063), Sinha, A. (2306096), Yadav, A. (2328157), Kanishka (2306118), & Srijan (2328235). (2026).
EEG Emotion Recognition System (Version 1.0) [Computer software]
GitHub repository: https://github.com/bhanukumardev/EmotionRecognition-EEG_Signal
```

---

## Dataset

The EEG data used in this project is sourced from the [EEG Brainwave Dataset - Feeling Emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions) available on Kaggle.

This dataset contains EEG recordings from multiple subjects experiencing different emotional states, with preprocessing and feature extraction pipelines implemented within this project.
