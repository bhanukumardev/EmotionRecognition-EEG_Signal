# 🧠 EEG Emotion Recognition using EEG Signals

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-RandomForest%20%7C%20SVM-green?style=flat-square)](#)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.89%25-brightgreen?style=flat-square)](#current-metrics)
[![License](https://img.shields.io/badge/License-Academic-yellow?style=flat-square)](#license)

*End-to-end emotion classification from EEG signals using classical ML models with GUI support*

[🚀 Live Demo](#-run-workflows) • [📖 Documentation](#-project-structure) • [📊 Metrics](#-current-metrics) • [🔧 Setup](#-setup)

</div>

---

## 🎯 Overview

End-to-end project for emotion classification from EEG data using classical ML models (Random Forest and SVM), with both CLI workflows and a desktop GUI. This project bridges the gap between neuroscience and machine learning, enabling real-time emotion recognition from brain signals.

### ✨ Key Features

- 🎮 **Interactive GUI** - User-friendly desktop application for predictions
- 🔬 **Two Pipelines** - Support for both synthetic and real EEG data
- 📈 **High Accuracy** - 97.89% accuracy on real EEG datasets
- ⚡ **Fast Inference** - Optimized models for real-time predictions
- 🎯 **Multiple Models** - Random Forest & SVM classifiers
- 📊 **Comprehensive Evaluation** - Detailed metrics and visualizations

---

## 👨‍💻 Contributors

<table>
<tr>
<td align="center">
<strong>Bhanu Kumar Dev</strong><br>(2328162)
</td>
<td align="center">
<strong>Adarsh Kumar</strong><br>(2328063)
</td>
<td align="center">
<strong>Aman Sinha</strong><br>(2306096)
</td>
</tr>
<tr>
<td align="center">
<strong>Srijan</strong><br>(2328235)
</td>
<td align="center">
<strong>Kanishka</strong><br>(2306118)
</td>
<td align="center">
<strong>Ashish Yadav</strong><br>(2328157)
</td>
</tr>
</table>

---

## 📦 What This Repository Includes

| Component | Description |
|-----------|-------------|
| 📊 **Data Generation** | Synthetic EEG data generation and preprocessing |
| 🔄 **Real Data Pipeline** | Kaggle emotion dataset processing & feature extraction |
| 🔍 **Feature Engineering** | PSD + Differential Entropy extraction |
| 🤖 **Model Training** | Random Forest & SVM implementation |
| 🎯 **Inference Tools** | Prediction utilities with confidence scoring |
| 🎨 **GUI Application** | Desktop app for easy predictions |
| 📈 **Evaluation Suite** | Confusion matrices, ROC curves, feature importance |

---

## 🧱 Project Structure

```
.
├── 🎯 main_gui.py                    # GUI application
├── 📋 emotions.csv                   # Sample dataset
├── ⚙️  requirements.txt               # Dependencies
├── 📄 VALIDATION_REPORT.md           # Performance metrics
│
├── 📁 data/
│   ├── raw/                          # Original EEG signals
│   ├── processed/                    # Preprocessed data
│   └── real_eeg/                     # Real dataset
│
├── 🤖 model/
│   ├── emotion_model_rf.pkl          # Random Forest model
│   ├── emotion_model_svm.pkl         # SVM model
│   └── scaler.pkl                    # Feature scaler
│
├── 📓 notebooks/
│   └── evaluation.ipynb              # Analysis & visualizations
│
├── 📊 results/
│   ├── confusion_matrices.png        # Model comparisons
│   └── roc_curves.png                # ROC analysis
│
└── 🔧 src/
    ├── data_pipeline.py              # Data preprocessing
    ├── train.py                      # Model training
    ├── predict.py                    # Inference module
    ├── process_real_data.py          # Real EEG processing
    └── evaluate.py                   # Evaluation utilities
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment support
- Git

### Step 1️⃣ Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Workflows

### 🌐 Live Deployment

<div align="center">

🎉 **[Try the Live Demo](https://emotionrecognition-eegsignal.streamlit.app/)** on Streamlit

</div>

### 🎮 Option A: GUI Application (Recommended for Quick Testing)

```bash
python main_gui.py
```

**What it does:**
- Loads pre-trained models
- Provides interactive interface
- Real-time prediction visualization
- Instant confidence scores

---

### 🔬 Option B: Synthetic EEG Pipeline

```bash
# Step 1: Generate synthetic data
python src/data_pipeline.py

# Step 2: Train models
python src/train.py

# Step 3: Make predictions
python src/predict.py --emotion Positive
```

**Outputs:**
- `model/` - Trained ML models
- `data/raw/` - Raw synthetic EEG signals
- `data/processed/` - Preprocessed features

📌 **Note:** Synthetic data may show artificially high metrics. Use real data for production validation.

---

### 📊 Option C: Real EEG Dataset Pipeline

**Download Dataset:**
- [Kaggle EEG Brainwave Dataset](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions)

**Setup & Run:**
```bash
# Place emotions.csv in project root
python src/process_real_data.py
```

**Generated Artifacts:**
```
✅ model/emotion_model_rf_real.pkl      # Random Forest (Real data)
✅ model/emotion_model_svm_real.pkl     # SVM (Real data)
✅ model/scaler_real.pkl                # Feature scaling
✅ model/label_encoder_real.pkl         # Label encoding
✅ model/model_info_real.pkl            # Model metadata
✅ results/confusion_matrices_real.png  # Performance comparison
✅ results/roc_curves_real.png          # ROC analysis
```

---

## 🔌 Python API Usage

```python
import numpy as np
from src.predict import EEGEmotionPredictor

# Initialize predictor
predictor = EEGEmotionPredictor()

# Prepare EEG data (32 channels, 5 seconds at 128 Hz)
eeg_data = np.random.randn(32, 640)

# Get prediction
result = predictor.predict(eeg_data)

print(f"Emotion: {result['emotion_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 📊 Current Metrics (Real EEG Validation)

**Source:** `VALIDATION_REPORT.md` (Real EEG dataset, test split)

| 📈 Metric | Random Forest | SVM |
|-----------|:-------------:|:-----:|
| **Accuracy** | **97.89%** ✅ | 95.55% |
| **Precision** (weighted) | **0.9790** | 0.9565 |
| **Recall** (weighted) | **0.9789** | 0.9555 |
| **F1-Score** (weighted) | **0.9789** | 0.9554 |

### 💡 Insights

> ✨ Random Forest outperforms SVM on this dataset with superior accuracy and balanced metrics.
>
> 📌 These realistic results are from **real EEG data** and should be preferred over synthetic-data benchmarks.

**Detailed Results:**
- 📊 View generated plots: `results/` directory
- 📄 Full report: `VALIDATION_REPORT.md`

---

## ⚠️ Important Notes

### 📦 Large Files (Git LFS)

The repository uses **Git LFS** for large `.npy` data artifacts:

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone with LFS support
git clone https://github.com/bhanukumardev/EmotionRecognition-EEG_Signal.git
```

### 🔧 Path Configuration

⚠️ `src/evaluate.py` contains environment-specific absolute paths (`/app/...`) that may need adjustment for local development:

```python
# Update paths if running locally
DATA_PATH = "./data/"  # Instead of /app/data/
```

---

## 🤝 Contributing

This is a **6th Semester Mini Project** at KIIT University. Contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is for **academic and educational use**. 

📋 Add a formal license file (MIT, Apache 2.0, etc.) if you plan to publish under a specific license.

---

## 📞 Support & Questions

For issues, questions, or suggestions:
- 🐛 Open an [Issue](https://github.com/bhanukumardev/EmotionRecognition-EEG_Signal/issues)
- 💬 Check existing [Discussions](https://github.com/bhanukumardev/EmotionRecognition-EEG_Signal/discussions)
- 📧 Contact the maintainers

---

<div align="center">

### ⭐ If you found this project helpful, please star it!

Made with ❤️ by the Team | Last Updated: 2026

</div>
