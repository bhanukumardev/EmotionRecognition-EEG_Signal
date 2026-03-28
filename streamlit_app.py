import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="EEG Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "model"
REAL_DATASET_PATH = PROJECT_ROOT / "data" / "real_eeg" / "emotions.csv"


def extract_real_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """Replicate feature selection logic used during real-data training."""
    feature_cols = []

    mean_cols = [col for col in df_features.columns if 'mean_' in col and 'moments' not in col]
    feature_cols.extend(mean_cols[:20])

    std_cols = [col for col in df_features.columns if 'stddev_' in col]
    feature_cols.extend(std_cols[:10])

    entropy_cols = [col for col in df_features.columns if 'entropy' in col]
    feature_cols.extend(entropy_cols)

    fft_cols = [col for col in df_features.columns if 'fft_' in col]
    if len(fft_cols) > 20:
        step = max(1, len(fft_cols) // 20)
        fft_selected = fft_cols[::step][:20]
    else:
        fft_selected = fft_cols
    feature_cols.extend(fft_selected)

    corr_cols = [col for col in df_features.columns if 'correlate_' in col]
    feature_cols.extend(corr_cols[:10])

    seen = set()
    feature_cols_unique = [x for x in feature_cols if not (x in seen or seen.add(x))]

    X_selected = df_features[feature_cols_unique].copy()

    if len(fft_cols) >= 4:
        X_selected['fft_delta_power'] = df_features[fft_cols[0]]
        X_selected['fft_theta_power'] = df_features[fft_cols[1]]
        X_selected['fft_alpha_power'] = df_features[fft_cols[2]]
        X_selected['fft_beta_power'] = df_features[fft_cols[3]]

    if len(mean_cols) >= 2:
        X_selected['asymmetry_index'] = df_features[mean_cols[0]] - df_features[mean_cols[1]]

    if len(std_cols) >= 1:
        X_selected['total_variability'] = df_features[std_cols[0]]

    return X_selected


@st.cache_resource
def load_real_model_artifacts():
    """Load real-data model artifacts once per app session."""
    rf_path = MODEL_DIR / "emotion_model_rf_real.pkl"
    svm_path = MODEL_DIR / "emotion_model_svm_real.pkl"
    scaler_path = MODEL_DIR / "scaler_real.pkl"
    encoder_path = MODEL_DIR / "label_encoder_real.pkl"
    info_path = MODEL_DIR / "model_info_real.pkl"

    missing = [
        str(p.name) for p in [rf_path, svm_path, scaler_path, encoder_path, info_path] if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts in model/: " + ", ".join(missing)
        )

    rf_model = joblib.load(rf_path)
    svm_model = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    with open(info_path, "rb") as f:
        model_info = pickle.load(f)

    return rf_model, svm_model, scaler, label_encoder, model_info


@st.cache_data
def load_real_dataset():
    """Load Kaggle real EEG dataset."""
    if not REAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {REAL_DATASET_PATH}")

    df = pd.read_csv(REAL_DATASET_PATH)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in emotions.csv")

    X = df.drop("label", axis=1).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))
    y = df["label"].astype(str)

    X_features = extract_real_features(X)
    return df, X_features, y

st.title("🧠 EEG Emotion Recognition System")
st.markdown("""
A machine learning pipeline for recognizing human emotions from EEG signals.
This Streamlit app demonstrates the emotion classification capabilities.
""")

st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "About", "Demo", "Documentation"])

if page == "Home":
    st.header("Welcome to EEG Emotion Recognition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Overview")
        st.markdown("""
        This project implements an end-to-end EEG emotion recognition system that:
        - **Preprocesses** raw EEG signals (filtering, normalization, artifact removal)
        - **Extracts** frequency-domain features (Power Spectral Density, Differential Entropy)
        - **Trains** classical ML models (Random Forest, SVM)
        - **Predicts** emotional states with confidence scores
        """)
    
    with col2:
        st.subheader("📊 Key Features")
        st.markdown("""
        - **32 EEG Channels** for comprehensive brain activity monitoring
        - **5 Frequency Bands**: Delta, Theta, Alpha, Beta, Gamma
        - **320 Extracted Features** per EEG window
        - **3 Emotion Classes**: Neutral, Positive, Negative
        - **Real-time Prediction** capability
        """)

elif page == "About":
    st.header("📖 About the Project")
    
    st.subheader("Contributors")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Bhanu Kumar Dev** (2328162)\n@bhanukumardev")
    with col2:
        st.markdown("**Adarsh Kumar** (2328063)\n@Adarshkumar0509")
    with col3:
        st.markdown("**Aman Sinha** (2306096)\n@amansinha11-dev")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Ashish Yadav** (2328157)\n@astreladg")
    with col2:
        st.markdown("**Kanishka** (2306118)\n@catharsis02")
    with col3:
        st.markdown("**Srijan** (2328235)\n@srijanisaDev")
    
    st.subheader("License & Citation")
    st.markdown("""
    This project is released under the MIT License.
    
    If you use this repository in your academic work, please cite:
    ```
Dev, B. K. (2328162), Kumar, A. (2328063), Sinha, A. (2306096), Yadav, A. (2328157), Kanishka (2306118), & Srijan (2328235). (2026).    EEG Emotion Recognition System (Version 1.0) [Computer software].
    GitHub: https://github.com/bhanukumardev/EmotionRecognition-EEG_Signal
    ```
    """)

elif page == "Demo":
    st.header("🎮 Interactive Demo")

    st.info("📝 This demo now uses the real Kaggle EEG dataset and real trained models.")
    st.markdown(
        "Dataset: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions"
    )

    try:
        rf_model, svm_model, scaler, label_encoder, model_info = load_real_model_artifacts()
        raw_df, X_features, y_labels = load_real_dataset()
    except Exception as e:
        st.error(f"Unable to load real model/data: {e}")
        st.stop()

    st.subheader("Real Dataset Inference")
    model_choice = st.selectbox("Select model", ["Random Forest", "SVM"], index=0)

    col1, col2 = st.columns([2, 1])
    with col1:
        sample_index = st.slider("Select dataset row", 0, len(raw_df) - 1, 0)
    with col2:
        if st.button("🎲 Random sample", key="random_sample"):
            st.session_state["sample_index"] = int(np.random.randint(0, len(raw_df)))

    if "sample_index" in st.session_state:
        sample_index = int(st.session_state["sample_index"])

    selected_row = raw_df.iloc[sample_index]
    true_label = str(selected_row["label"])

    st.caption(f"Selected row: {sample_index} | True label: {true_label}")

    preview_cols = [c for c in raw_df.columns if c != "label"][:8]
    st.write("Feature preview (first 8 columns):")
    st.dataframe(selected_row[preview_cols].to_frame().T, use_container_width=True)

    st.subheader("Emotion Prediction")
    if st.button("🔍 Predict Emotion", key="predict_btn"):
        model = rf_model if model_choice == "Random Forest" else svm_model

        x_row = X_features.iloc[[sample_index]].values
        x_scaled = scaler.transform(x_row)

        pred_idx = int(model.predict(x_scaled)[0])
        pred_label = str(label_encoder.inverse_transform([pred_idx])[0])

        proba_dict = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_scaled)[0]
            class_labels = label_encoder.inverse_transform(model.classes_.astype(int))
            proba_dict = {
                str(lbl): float(prob) for lbl, prob in zip(class_labels, probs)
            }

        labels = list(label_encoder.classes_)
        probs_for_labels = [proba_dict.get(lbl, 0.0) for lbl in labels]

        pred_conf = proba_dict.get(pred_label, 0.0)
        is_correct = pred_label == true_label

        metric_cols = st.columns(3)
        for idx, lbl in enumerate(labels[:3]):
            metric_cols[idx].metric(lbl, f"{probs_for_labels[idx] * 100:.2f}%")

        if is_correct:
            st.success(
                f"🎯 Predicted: **{pred_label}** (Confidence: {pred_conf * 100:.2f}%) | True label: **{true_label}**"
            )
        else:
            st.warning(
                f"Predicted: **{pred_label}** (Confidence: {pred_conf * 100:.2f}%) | True label: **{true_label}**"
            )

        st.caption(
            f"Using {model_choice} | Model dataset: {model_info.get('dataset', 'N/A')}"
        )

elif page == "Documentation":
    st.header("📚 Documentation")
    
    st.subheader("EEG Frequency Bands")
    band_info = {
        "Delta (0.5-4 Hz)": "Deep sleep, unconscious states",
        "Theta (4-8 Hz)": "Drowsiness, light sleep, meditation",
        "Alpha (8-13 Hz)": "Relaxed wakefulness",
        "Beta (13-30 Hz)": "Active thinking, concentration",
        "Gamma (30-45 Hz)": "Higher-order cognitive processes"
    }
    
    for band, description in band_info.items():
        st.write(f"**{band}**: {description}")
    
    st.subheader("Model Performance")
    performance_data = {
        "Metric": ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-Score (weighted)"],
        "Random Forest": [0.9789, 0.9790, 0.9789, 0.9789],
        "SVM": [0.9555, 0.9565, 0.9555, 0.9554]
    }
    
    st.dataframe(performance_data, use_container_width=True)
    
    st.info(
        "Results shown are from the real Kaggle EEG dataset. "
        "Metrics are high but not perfect, which is expected for realistic data."
    )
    
    st.subheader("GitHub Repository")
    st.markdown("""
    For more information, source code, and documentation, visit:
    [EmotionRecognition-EEG_Signal on GitHub](https://github.com/bhanukumardev/EmotionRecognition-EEG_Signal)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧠 EEG Emotion Recognition System | 6th Semester Mini Project | KIIT University</p>
    <p>Built with ❤️ using Streamlit | <a href='https://github.com/bhanukumardev/EmotionRecognition-EEG_Signal'>GitHub Repo</a></p>
</div>
""", unsafe_allow_html=True)
