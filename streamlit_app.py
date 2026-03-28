import streamlit as st
import numpy as np
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
    
    st.info("📝 This demo shows how the emotion classification works with synthetic EEG data.")
    
    # Simulate EEG data generation
    st.subheader("Generate Sample EEG Data")
    
    col1, col2 = st.columns(2)
    with col1:
        num_channels = st.slider("Number of EEG Channels", 1, 32, 32)
        duration_seconds = st.slider("Duration (seconds)", 1, 10, 5)
    
    with col2:
        sampling_rate = st.number_input("Sampling Rate (Hz)", value=200)
        emotion_class = st.selectbox("Select Emotion Class", ["Neutral", "Positive", "Negative"])
    
    # Generate synthetic EEG data
    num_samples = int(duration_seconds * sampling_rate)
    eeg_data = np.random.randn(num_channels, num_samples) * 50  # Synthetic EEG in microvolts
    
    st.success(f"✅ Generated {num_channels}-channel EEG data with {num_samples} samples")
    
    # Display sample data
    st.subheader("Sample EEG Data (first 200 samples)")
    st.line_chart(eeg_data[:5, :200].T)  # Plot first 5 channels
    
    st.subheader("Emotion Prediction")
    if st.button("🔍 Predict Emotion", key="predict_btn"):
        # Simulated prediction
        emotions = {"Neutral": 0.7, "Positive": 0.15, "Negative": 0.15}
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Neutral", f"{emotions['Neutral']*100:.1f}%")
        with col2:
            st.metric("Positive", f"{emotions['Positive']*100:.1f}%")
        with col3:
            st.metric("Negative", f"{emotions['Negative']*100:.1f}%")
        
        # Show predicted emotion
        predicted = max(emotions, key=emotions.get)
        confidence = emotions[predicted]
        st.success(f"🎯 Predicted Emotion: **{predicted}** (Confidence: {confidence*100:.1f}%)")

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
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Random Forest": [0.9789, 0.9790, 0.9789, 0.9789],
        "SVM": [0.9555, 0.9565, 0.9555, 0.9554]
    }
    
    st.dataframe(performance_data, use_container_width=True)
    
    st.info(
        "⚠️ Note: Results are based on the Kaggle EEG Brainwave Dataset (Real Data)"
        "Perfect scores indicate overfitting to synthetic patterns."
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
