"""
EEG Emotion Recognition: Inference Script
Load trained model and predict emotion from EEG data
"""
import numpy as np
import pickle
import joblib
import os
import argparse
from scipy.signal import welch


class EEGEmotionPredictor:
    """Predict emotion from EEG data using trained model."""

    def __init__(self, model_dir='/app/eeg_emotion_recognition_1806/model', use_real_model=True):
        """
        Initialize predictor with trained model.

        Args:
            model_dir: Directory containing trained model files
            use_real_model: If True, use models trained on real EEG data
        """
        self.model_dir = model_dir
        self.use_real_model = use_real_model

        # Determine which model to load
        if use_real_model and os.path.exists(os.path.join(model_dir, 'model_info_real.pkl')):
            model_suffix = '_real'
            print("Loading model trained on REAL EEG data")
        else:
            model_suffix = ''
            print("Loading model trained on synthetic data")

        # Load model info
        info_file = f'model_info{model_suffix}.pkl'
        with open(os.path.join(model_dir, info_file), 'rb') as f:
            self.model_info = pickle.load(f)

        # Handle both old and new model info formats
        if 'model_type' in self.model_info:
            self.model_type = self.model_info['model_type']
            self.emotions = self.model_info['emotions']
            self.sampling_rate = self.model_info['sampling_rate']
            self.n_channels = self.model_info['n_channels']
        else:
            # New format from real data training
            self.model_type = 'random_forest'  # Default to RF as primary
            self.emotions = self.model_info.get('classes', ['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
            self.sampling_rate = 128
            self.n_channels = 32

        # Load model and scaler
        if model_suffix == '_real':
            # Load real-data models
            self.rf_model = joblib.load(os.path.join(model_dir, 'emotion_model_rf_real.pkl'))
            self.svm_model = joblib.load(os.path.join(model_dir, 'emotion_model_svm_real.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler_real.pkl'))
            self.model = self.rf_model  # Use RF as default
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder_real.pkl'))
        else:
            # Load original models
            if self.model_type == 'random_forest':
                self.model = joblib.load(os.path.join(model_dir, 'emotion_model_rf.pkl'))
                self.scaler = joblib.load(os.path.join(model_dir, 'scaler_rf.pkl'))
            else:
                self.model = joblib.load(os.path.join(model_dir, 'emotion_model_svm.pkl'))
                self.scaler = joblib.load(os.path.join(model_dir, 'scaler_svm.pkl'))

        # Define frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        print(f"Loaded {self.model_type} model")
        print(f"Emotions: {self.emotions}")
        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"Expected channels: {self.n_channels}")

    def extract_psd_features(self, eeg_segment):
        """Extract PSD features from EEG segment."""
        n_channels = eeg_segment.shape[0]
        features = []

        for ch in range(n_channels):
            channel_data = eeg_segment[ch, :]

            # Compute PSD using Welch's method
            freqs, psd = welch(
                channel_data,
                fs=self.sampling_rate,
                nperseg=min(256, len(channel_data)),
                noverlap=min(128, len(channel_data)//2)
            )

            # Extract power for each frequency band
            for band_name, (low_freq, high_freq) in self.bands.items():
                band_indices = np.logical_and(freqs >= low_freq, freqs <= high_freq)

                if np.any(band_indices):
                    band_power = np.mean(psd[band_indices])
                    features.append(band_power)
                else:
                    features.append(0.0)

        return np.array(features)

    def extract_differential_entropy(self, eeg_segment):
        """Extract Differential Entropy features from EEG segment."""
        n_channels = eeg_segment.shape[0]
        features = []

        for ch in range(n_channels):
            channel_data = eeg_segment[ch, :]

            # Compute PSD
            freqs, psd = welch(
                channel_data,
                fs=self.sampling_rate,
                nperseg=min(256, len(channel_data)),
                noverlap=min(128, len(channel_data)//2)
            )

            # Extract DE for each frequency band
            for band_name, (low_freq, high_freq) in self.bands.items():
                band_indices = np.logical_and(freqs >= low_freq, freqs <= high_freq)

                if np.any(band_indices):
                    band_power = np.mean(psd[band_indices])
                    if band_power > 0:
                        de = 0.5 * np.log(2 * np.pi * np.e * band_power)
                    else:
                        de = 0.0
                    features.append(de)
                else:
                    features.append(0.0)

        return np.array(features)

    def extract_features(self, eeg_segment):
        """
        Extract all features from EEG segment.

        Args:
            eeg_segment: EEG data array (n_channels, n_timepoints)

        Returns:
            Feature vector
        """
        psd_features = self.extract_psd_features(eeg_segment)
        de_features = self.extract_differential_entropy(eeg_segment)
        return np.concatenate([psd_features, de_features])

    def predict(self, eeg_segment):
        """
        Predict emotion from EEG segment.

        Args:
            eeg_segment: EEG data array (n_channels, n_timepoints)

        Returns:
            Dictionary with prediction results
        """
        # Validate input shape
        if eeg_segment.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {eeg_segment.shape[0]}"
            )

        # Extract features
        features = self.extract_features(eeg_segment)
        features = features.reshape(1, -1)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Get emotion label
        if hasattr(self, 'label_encoder'):
            emotion_label = self.label_encoder.inverse_transform([prediction])[0]
        else:
            emotion_label = self.emotions[prediction]

        return {
            'emotion_label': emotion_label,
            'emotion_index': prediction,
            'probabilities': {
                emotion: prob
                for emotion, prob in zip(self.emotions, probabilities)
            },
            'confidence': probabilities[prediction]
        }

    def predict_batch(self, eeg_segments):
        """
        Predict emotions for multiple EEG segments.

        Args:
            eeg_segments: EEG data array (n_samples, n_channels, n_timepoints)

        Returns:
            List of prediction results
        """
        results = []
        for segment in eeg_segments:
            result = self.predict(segment)
            results.append(result)
        return results


def generate_sample_eeg(emotion='Neutral', n_channels=32, sampling_rate=128, duration=5):
    """
    Generate a synthetic EEG sample for testing.

    Args:
        emotion: Emotion label ('Neutral', 'Positive', 'Negative')
        n_channels: Number of EEG channels
        sampling_rate: Sampling rate in Hz
        duration: Duration in seconds

    Returns:
        EEG segment array
    """
    n_timepoints = sampling_rate * duration
    eeg_data = np.zeros((n_channels, n_timepoints))

    # Emotion-specific band powers
    band_powers = {
        'Neutral': {'delta': 0.3, 'theta': 0.2, 'alpha': 0.25, 'beta': 0.15, 'gamma': 0.1},
        'Positive': {'delta': 0.15, 'theta': 0.15, 'alpha': 0.35, 'beta': 0.2, 'gamma': 0.15},
        'Negative': {'delta': 0.2, 'theta': 0.25, 'alpha': 0.15, 'beta': 0.25, 'gamma': 0.15}
    }

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    t = np.arange(n_timepoints) / sampling_rate

    for ch in range(n_channels):
        channel_signal = np.zeros(n_timepoints)

        for band_name, (low, high) in bands.items():
            power = band_powers[emotion][band_name]

            # Generate frequency components
            for _ in range(3):
                freq = np.random.uniform(low, high)
                amplitude = np.sqrt(power / 3) * np.random.uniform(0.5, 1.5)
                phase = np.random.uniform(0, 2 * np.pi)

                component = amplitude * np.sin(2 * np.pi * freq * t + phase)
                channel_signal += component

        # Add noise
        noise = np.random.normal(0, 0.1, n_timepoints)
        channel_signal += noise

        eeg_data[ch, :] = channel_signal

    return eeg_data


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='EEG Emotion Recognition Prediction')
    parser.add_argument('--input', type=str, help='Path to EEG data file (.npy)')
    parser.add_argument('--emotion', type=str, default='Neutral',
                        choices=['Neutral', 'Positive', 'Negative'],
                        help='Generate synthetic sample with specified emotion')
    parser.add_argument('--model-dir', type=str,
                        default='/app/eeg_emotion_recognition_1806/model',
                        help='Path to model directory')
    parser.add_argument('--use-real-model', action='store_true',
                        help='Use model trained on real EEG data')

    args = parser.parse_args()

    print("="*60)
    print("EEG EMOTION RECOGNITION - INFERENCE")
    print("="*60)

    # Initialize predictor
    predictor = EEGEmotionPredictor(model_dir=args.model_dir, use_real_model=args.use_real_model)

    # Load or generate EEG data
    if args.input:
        print(f"\nLoading EEG data from: {args.input}")
        eeg_segment = np.load(args.input)
        if eeg_segment.ndim == 3:
            eeg_segment = eeg_segment[0]  # Take first sample if batch
    else:
        print(f"\nGenerating synthetic EEG sample with emotion: {args.emotion}")
        eeg_segment = generate_sample_eeg(
            emotion=args.emotion,
            n_channels=predictor.n_channels,
            sampling_rate=predictor.sampling_rate
        )

    print(f"EEG segment shape: {eeg_segment.shape}")

    # Make prediction
    print("\n" + "-"*60)
    print("PREDICTION")
    print("-"*60)

    result = predictor.predict(eeg_segment)

    print(f"\nPredicted Emotion: {result['emotion_label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nProbabilities:")
    for emotion, prob in result['probabilities'].items():
        bar = "█" * int(prob * 30)
        print(f"  {emotion:10s}: {prob:.4f} {bar}")

    print("\n" + "="*60)
    print("Inference completed successfully!")
    print("="*60)

    return result


if __name__ == '__main__':
    main()
