# EEG Emotion Recognition using EEG Signals

End-to-end project for emotion classification from EEG data using classical ML models (Random Forest and SVM), with both CLI workflows and a desktop GUI.

## Contributors

- Bhanu Kumar Dev (2328162)
- Adarsh Kumar (2328063)
- Aman Sinha (2306096)
- Srijan (2328235)
- Kanishka (2306118)
- Ashish Yadav (2328157)

## What This Repository Includes

- Synthetic EEG data generation and preprocessing pipeline
- Real EEG dataset processing pipeline (Kaggle emotion dataset format)
- Feature extraction (PSD + Differential Entropy for synthetic workflow)
- Model training (Random Forest, SVM)
- Inference utilities and GUI app
- Evaluation artifacts (confusion matrices, ROC curves, feature importances)

## Project Structure

```
.
├── main_gui.py
├── emotions.csv
├── requirements.txt
├── VALIDATION_REPORT.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── real_eeg/
├── model/
├── notebooks/
│   └── evaluation.ipynb
├── results/
└── src/
	├── data_pipeline.py
	├── train.py
	├── predict.py
	├── process_real_data.py
	└── evaluate.py
```

## Setup

### 1) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run Workflows

## Live Deployment

- Streamlit App: https://emotionrecognition-eegsignal.streamlit.app/

### A. GUI (fastest way to try predictions)

```bash
python main_gui.py
```

### B. Synthetic EEG pipeline

```bash
python src/data_pipeline.py
python src/train.py
python src/predict.py --emotion Positive
```

This workflow uses generated EEG segments and produces model artifacts in `model/` and data arrays in `data/raw/` and `data/processed/`.

### C. Real EEG dataset workflow

Dataset download link (Kaggle): https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions

If you have `emotions.csv` (Kaggle EEG emotion dataset format):

```bash
python src/process_real_data.py
```

Expected outputs include:
- `model/emotion_model_rf_real.pkl`
- `model/emotion_model_svm_real.pkl`
- `model/scaler_real.pkl`
- `model/label_encoder_real.pkl`
- `model/model_info_real.pkl`
- `results/confusion_matrices_real.png`
- `results/roc_curves_real.png`

## Prediction API Example

```python
import numpy as np
from src.predict import EEGEmotionPredictor

predictor = EEGEmotionPredictor()
eeg_data = np.random.randn(32, 640)  # 32 channels, 5s at 128 Hz
result = predictor.predict(eeg_data)

print(result["emotion_label"], result["confidence"])
```

## Current Metrics (Real EEG Validation)

From `VALIDATION_REPORT.md` (real EEG dataset, test split):

| Metric | Random Forest | SVM |
|--------|---------------|-----|
| Accuracy | 97.89% | 95.55% |
| Precision (weighted) | 0.9790 | 0.9565 |
| Recall (weighted) | 0.9789 | 0.9555 |
| F1-Score (weighted) | 0.9789 | 0.9554 |

These are realistic real-data results and should be preferred over synthetic-data runs, which can sometimes appear artificially high.

See `results/` for generated plots and `VALIDATION_REPORT.md` for the detailed report.

## Notes

- The repository includes large `.npy` data artifacts. Git LFS is used for these files.
- `src/evaluate.py` contains environment-specific absolute paths (`/app/...`) and may need path adjustments for local runs.

## License

This project is for academic/project use. Add a formal license file if you plan to publish under a specific license.
