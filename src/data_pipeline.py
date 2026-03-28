"""
EEG Data Pipeline: Synthetic Data Generator and Preprocessing
Generates realistic EEG signals with emotion-specific patterns and applies preprocessing
"""
import numpy as np
import pandas as pd
import os
import pickle
from scipy import signal
from scipy.signal import butter, filtfilt, welch
import mne


class EEGDataGenerator:
    """Generate synthetic EEG data with emotion-specific frequency patterns."""
    
    def __init__(self, n_samples=1000, n_channels=32, sampling_rate=128, segment_length=5):
        """
        Initialize EEG data generator.
        
        Args:
            n_samples: Number of EEG segments to generate
            n_channels: Number of EEG channels
            sampling_rate: Sampling rate in Hz
            segment_length: Length of each segment in seconds
        """
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.n_timepoints = sampling_rate * segment_length
        
        # Emotion labels: 0=Neutral, 1=Positive, 2=Negative
        self.emotions = ['Neutral', 'Positive', 'Negative']
        self.emotion_map = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
        
        # Channel names (standard 10-20 system subset)
        all_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2', 'A1', 'A2',
            'FC1', 'FC2', 'CP1', 'CP2',
            'FC5', 'FC6', 'CP5', 'CP6',
            'TP7', 'TP8', 'POz', 'Oz'
        ]
        self.channel_names = all_channels[:n_channels]
        
    def generate_band_power(self, emotion, band):
        """
        Generate power for specific frequency band based on emotion.
        
        Frequency bands:
        - Delta: 0.5-4 Hz (deep sleep, unconscious)
        - Theta: 4-8 Hz (meditation, drowsiness)
        - Alpha: 8-13 Hz (relaxed, eyes closed)
        - Beta: 13-30 Hz (active thinking, alert)
        - Gamma: 30-45 Hz (cognitive processing)
        """
        band_powers = {
            'Neutral': {'delta': 0.3, 'theta': 0.2, 'alpha': 0.25, 'beta': 0.15, 'gamma': 0.1},
            'Positive': {'delta': 0.15, 'theta': 0.15, 'alpha': 0.35, 'beta': 0.2, 'gamma': 0.15},
            'Negative': {'delta': 0.2, 'theta': 0.25, 'alpha': 0.15, 'beta': 0.25, 'gamma': 0.15}
        }
        
        base_power = band_powers[emotion][band]
        # Add random variation
        noise = np.random.normal(0, 0.05)
        return max(0.01, base_power + noise)
    
    def generate_eeg_segment(self, emotion):
        """Generate a single EEG segment for given emotion."""
        eeg_data = np.zeros((self.n_channels, self.n_timepoints))
        
        # Frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        t = np.arange(self.n_timepoints) / self.sampling_rate
        
        for ch in range(self.n_channels):
            channel_signal = np.zeros(self.n_timepoints)
            
            for band_name, (low, high) in bands.items():
                power = self.generate_band_power(emotion, band_name)
                
                # Generate multiple frequency components within band
                n_components = 3
                for _ in range(n_components):
                    freq = np.random.uniform(low, high)
                    amplitude = np.sqrt(power / n_components) * np.random.uniform(0.5, 1.5)
                    phase = np.random.uniform(0, 2 * np.pi)
                    
                    # Add slight frequency modulation for realism
                    fm = np.sin(2 * np.pi * 0.1 * t) * 0.5
                    instantaneous_freq = freq + fm
                    
                    component = amplitude * np.sin(2 * np.pi * instantaneous_freq * t + phase)
                    channel_signal += component
            
            # Add channel-specific noise
            noise_level = np.random.uniform(0.05, 0.15)
            noise = np.random.normal(0, noise_level, self.n_timepoints)
            channel_signal += noise
            
            # Add occasional artifacts (eye blinks, muscle activity)
            if np.random.random() < 0.1:  # 10% chance of artifact
                artifact_start = np.random.randint(0, self.n_timepoints - 30)
                artifact_duration = np.random.randint(10, 25)
                artifact_amplitude = np.random.uniform(2, 5)
                artifact_end = min(artifact_start + artifact_duration, self.n_timepoints)
                actual_duration = artifact_end - artifact_start
                channel_signal[artifact_start:artifact_end] += \
                    artifact_amplitude * np.sin(np.linspace(0, np.pi, actual_duration))
            
            eeg_data[ch, :] = channel_signal
        
        return eeg_data
    
    def generate_dataset(self, output_dir):
        """Generate complete dataset with train/test split."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {self.n_samples} EEG segments...")
        print(f"Channels: {self.n_channels}, Sampling rate: {self.sampling_rate} Hz")
        print(f"Segment length: {self.segment_length}s ({self.n_timepoints} timepoints)")
        
        # Generate data
        data = []
        labels = []
        metadata = []
        
        samples_per_emotion = self.n_samples // 3
        
        for emotion in self.emotions:
            print(f"Generating {samples_per_emotion} samples for emotion: {emotion}")
            for i in range(samples_per_emotion):
                eeg_segment = self.generate_eeg_segment(emotion)
                data.append(eeg_segment)
                labels.append(self.emotion_map[emotion])
                metadata.append({
                    'emotion': emotion,
                    'label': self.emotion_map[emotion],
                    'subject_id': np.random.randint(1, 20),
                    'trial': i
                })
        
        # Shuffle data
        indices = np.random.permutation(len(data))
        data = [data[i] for i in indices]
        labels = [labels[i] for i in indices]
        metadata = [metadata[i] for i in indices]
        
        # Convert to arrays
        X = np.array(data)  # Shape: (n_samples, n_channels, n_timepoints)
        y = np.array(labels)
        
        # Split into train/test (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        metadata_train = metadata[:split_idx]
        metadata_test = metadata[split_idx:]
        
        # Save raw data
        np.save(os.path.join(output_dir, 'X_train_raw.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test_raw.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        with open(os.path.join(output_dir, 'metadata_train.pkl'), 'wb') as f:
            pickle.dump(metadata_train, f)
        with open(os.path.join(output_dir, 'metadata_test.pkl'), 'wb') as f:
            pickle.dump(metadata_test, f)
        
        # Save channel info
        channel_info = {
            'channel_names': self.channel_names,
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'n_timepoints': self.n_timepoints,
            'segment_length': self.segment_length,
            'emotions': self.emotions,
            'emotion_map': self.emotion_map
        }
        with open(os.path.join(output_dir, 'channel_info.pkl'), 'wb') as f:
            pickle.dump(channel_info, f)
        
        print(f"\nDataset saved to {output_dir}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Data shape: {X_train.shape}")
        print(f"Classes distribution (train): {np.bincount(y_train)}")
        
        return X_train, X_test, y_train, y_test


class EEGPreprocessor:
    """Preprocess EEG data using MNE and custom filters."""
    
    def __init__(self, sampling_rate=128, l_freq=0.5, h_freq=45):
        """
        Initialize EEG preprocessor.
        
        Args:
            sampling_rate: Sampling rate in Hz
            l_freq: Lower frequency cutoff for bandpass filter
            h_freq: Upper frequency cutoff for bandpass filter
        """
        self.sampling_rate = sampling_rate
        self.l_freq = l_freq
        self.h_freq = h_freq
        
    def create_mne_info(self, n_channels, channel_names):
        """Create MNE info structure."""
        # Create channel types (all EEG)
        ch_types = ['eeg'] * n_channels
        
        # Create MNE info
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.sampling_rate,
            ch_types=ch_types
        )
        return info
    
    def bandpass_filter(self, data, n_channels, channel_names):
        """
        Apply bandpass filter using MNE.
        
        Args:
            data: EEG data array (n_channels, n_timepoints)
            n_channels: Number of channels
            channel_names: List of channel names
            
        Returns:
            Filtered EEG data
        """
        # Create MNE info
        info = self.create_mne_info(n_channels, channel_names)
        
        # Create RawArray
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Apply bandpass filter
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, verbose=False)
        
        # Get filtered data
        filtered_data = raw.get_data()
        
        return filtered_data
    
    def remove_artifacts(self, data, n_channels, channel_names):
        """
        Remove artifacts using simple thresholding.
        
        Args:
            data: EEG data array (n_channels, n_timepoints)
            n_channels: Number of channels
            channel_names: List of channel names
            
        Returns:
            Cleaned EEG data
        """
        # Z-score normalization per channel
        cleaned_data = np.zeros_like(data)
        for ch in range(n_channels):
            channel_data = data[ch, :]
            
            # Remove extreme outliers (artifacts)
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            
            # Clip values beyond 4 standard deviations
            lower_bound = mean - 4 * std
            upper_bound = mean + 4 * std
            channel_data = np.clip(channel_data, lower_bound, upper_bound)
            
            cleaned_data[ch, :] = channel_data
        
        return cleaned_data
    
    def normalize(self, data):
        """
        Normalize EEG data using z-score normalization.
        
        Args:
            data: EEG data array (n_channels, n_timepoints)
            
        Returns:
            Normalized EEG data
        """
        normalized_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            channel_data = data[ch, :]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:
                normalized_data[ch, :] = (channel_data - mean) / std
            else:
                normalized_data[ch, :] = channel_data - mean
        
        return normalized_data
    
    def preprocess_segment(self, segment, channel_names):
        """
        Preprocess a single EEG segment.
        
        Args:
            segment: EEG segment (n_channels, n_timepoints)
            channel_names: List of channel names
            
        Returns:
            Preprocessed EEG segment
        """
        n_channels = segment.shape[0]
        
        # Step 1: Bandpass filter (0.5-45 Hz)
        filtered = self.bandpass_filter(segment, n_channels, channel_names)
        
        # Step 2: Remove artifacts
        cleaned = self.remove_artifacts(filtered, n_channels, channel_names)
        
        # Step 3: Normalize
        normalized = self.normalize(cleaned)
        
        return normalized
    
    def preprocess_dataset(self, X, output_path, channel_names):
        """
        Preprocess entire dataset.
        
        Args:
            X: EEG data array (n_samples, n_channels, n_timepoints)
            output_path: Path to save preprocessed data
            channel_names: List of channel names
            
        Returns:
            Preprocessed EEG data
        """
        print(f"Preprocessing {len(X)} EEG segments...")
        
        X_processed = np.zeros_like(X)
        
        for i in range(len(X)):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(X)} segments...")
            
            X_processed[i] = self.preprocess_segment(X[i], channel_names)
        
        # Save preprocessed data
        np.save(output_path, X_processed)
        print(f"Preprocessed data saved to {output_path}")
        
        return X_processed


def verify_preprocessing(X_raw, X_processed, sampling_rate):
    """Verify preprocessing results."""
    print("\n" + "="*60)
    print("PREPROCESSING VERIFICATION")
    print("="*60)
    
    # Check shapes
    print(f"\nRaw data shape: {X_raw.shape}")
    print(f"Processed data shape: {X_processed.shape}")
    assert X_raw.shape == X_processed.shape, "Shape mismatch!"
    print("✓ Shapes match")
    
    # Check for NaN/Inf
    print(f"\nRaw data - NaN count: {np.isnan(X_raw).sum()}, Inf count: {np.isinf(X_raw).sum()}")
    print(f"Processed data - NaN count: {np.isnan(X_processed).sum()}, Inf count: {np.isinf(X_processed).sum()}")
    assert np.isnan(X_processed).sum() == 0, "NaN values found in processed data!"
    assert np.isinf(X_processed).sum() == 0, "Inf values found in processed data!"
    print("✓ No NaN or Inf values")
    
    # Check normalization (mean ~0, std ~1)
    sample_idx = 0
    channel_idx = 0
    mean = np.mean(X_processed[sample_idx, channel_idx, :])
    std = np.std(X_processed[sample_idx, channel_idx, :])
    print(f"\nSample segment statistics (channel {channel_idx}):")
    print(f"  Mean: {mean:.4f} (expected ~0)")
    print(f"  Std: {std:.4f} (expected ~1)")
    assert abs(mean) < 0.1, "Mean not close to 0!"
    assert 0.8 < std < 1.2, "Std not close to 1!"
    print("✓ Normalization verified")
    
    # Check frequency content
    freqs, psd = welch(X_processed[sample_idx, channel_idx, :], fs=sampling_rate, nperseg=128)
    
    # Check if frequencies are within 0.5-45 Hz
    significant_freqs = freqs[psd > np.max(psd) * 0.1]
    if len(significant_freqs) > 0:
        min_freq = np.min(significant_freqs)
        max_freq = np.max(significant_freqs)
        print(f"\nFrequency content:")
        print(f"  Min significant frequency: {min_freq:.2f} Hz")
        print(f"  Max significant frequency: {max_freq:.2f} Hz")
        assert min_freq >= 0.5, f"Frequencies below 0.5 Hz detected: {min_freq:.2f} Hz"
        assert max_freq <= 45, f"Frequencies above 45 Hz detected: {max_freq:.2f} Hz"
        print("✓ Frequency content within 0.5-45 Hz band")
    
    print("\n" + "="*60)
    print("ALL VERIFICATION CHECKS PASSED!")
    print("="*60)


def main():
    """Main function to generate and preprocess data."""
    # Configuration
    output_dir = '/app/eeg_emotion_recognition_1806/data/raw'
    processed_dir = '/app/eeg_emotion_recognition_1806/data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Generate synthetic data
    print("="*60)
    print("STEP 1: GENERATING SYNTHETIC EEG DATA")
    print("="*60)
    
    generator = EEGDataGenerator(
        n_samples=1200,
        n_channels=32,
        sampling_rate=128,
        segment_length=5
    )
    X_train, X_test, y_train, y_test = generator.generate_dataset(output_dir)
    
    # Load channel info
    with open(os.path.join(output_dir, 'channel_info.pkl'), 'rb') as f:
        channel_info = pickle.load(f)
    channel_names = channel_info['channel_names']
    sampling_rate = channel_info['sampling_rate']
    
    # Preprocess data
    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING EEG DATA")
    print("="*60)
    
    preprocessor = EEGPreprocessor(
        sampling_rate=sampling_rate,
        l_freq=0.5,
        h_freq=45
    )
    
    # Preprocess training data
    X_train_processed = preprocessor.preprocess_dataset(
        X_train,
        os.path.join(processed_dir, 'X_train_processed.npy'),
        channel_names
    )
    
    # Preprocess test data
    X_test_processed = preprocessor.preprocess_dataset(
        X_test,
        os.path.join(processed_dir, 'X_test_processed.npy'),
        channel_names
    )
    
    # Copy labels
    import shutil
    shutil.copy(os.path.join(output_dir, 'y_train.npy'), os.path.join(processed_dir, 'y_train.npy'))
    shutil.copy(os.path.join(output_dir, 'y_test.npy'), os.path.join(processed_dir, 'y_test.npy'))
    shutil.copy(os.path.join(output_dir, 'channel_info.pkl'), os.path.join(processed_dir, 'channel_info.pkl'))
    
    # Verify preprocessing
    print("\n" + "="*60)
    print("STEP 3: VERIFICATION")
    print("="*60)
    
    # Test with a small subset for verification
    verify_preprocessing(X_train[:10], X_train_processed[:10], sampling_rate)
    
    print("\n" + "="*60)
    print("DATA PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nProcessed data saved to: {processed_dir}")
    print(f"  - X_train_processed.npy: {X_train_processed.shape}")
    print(f"  - X_test_processed.npy: {X_test_processed.shape}")


if __name__ == '__main__':
    main()
