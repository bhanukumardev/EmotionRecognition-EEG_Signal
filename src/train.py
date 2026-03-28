"""
EEG Emotion Recognition: Feature Extraction and Model Training
Extracts PSD features from EEG data and trains classification models
"""
import numpy as np
import pickle
import os
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


class EEGFeatureExtractor:
    """Extract Power Spectral Density (PSD) features from EEG data."""
    
    def __init__(self, sampling_rate=128):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
        # Define frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    
    def extract_psd_features(self, eeg_segment):
        """
        Extract PSD features for all frequency bands.
        
        Args:
            eeg_segment: EEG data array (n_channels, n_timepoints)
            
        Returns:
            Feature vector containing PSD for each band per channel
        """
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
                # Find frequency indices within band
                band_indices = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                
                if np.any(band_indices):
                    # Calculate mean power in band
                    band_power = np.mean(psd[band_indices])
                    features.append(band_power)
                else:
                    features.append(0.0)
        
        return np.array(features)
    
    def extract_differential_entropy(self, eeg_segment):
        """
        Extract Differential Entropy (DE) features.
        
        Args:
            eeg_segment: EEG data array (n_channels, n_timepoints)
            
        Returns:
            Feature vector containing DE for each band per channel
        """
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
                    # Differential Entropy: 0.5 * log(2*pi*e*variance)
                    band_power = np.mean(psd[band_indices])
                    if band_power > 0:
                        de = 0.5 * np.log(2 * np.pi * np.e * band_power)
                    else:
                        de = 0.0
                    features.append(de)
                else:
                    features.append(0.0)
        
        return np.array(features)
    
    def extract_all_features(self, X):
        """
        Extract features for entire dataset.
        
        Args:
            X: EEG data array (n_samples, n_channels, n_timepoints)
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        print(f"Extracting features from {len(X)} samples...")
        
        features_list = []
        for i in range(len(X)):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(X)} samples...")
            
            # Extract PSD features
            psd_features = self.extract_psd_features(X[i])
            
            # Extract DE features
            de_features = self.extract_differential_entropy(X[i])
            
            # Combine features
            combined_features = np.concatenate([psd_features, de_features])
            features_list.append(combined_features)
        
        return np.array(features_list)


class EEGEmotionClassifier:
    """Train and evaluate emotion classification models."""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize classifier.
        
        Args:
            model_type: 'random_forest' or 'svm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\nTraining {self.model_type} model...")
        print(f"Training data shape: {X_train.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
        
        return train_acc
    
    def evaluate(self, X_test, y_test, emotion_labels=None):
        """
        Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            emotion_labels: List of emotion label names
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating model on test set...")
        print(f"Test data shape: {X_test.shape}")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:")
        
        if emotion_labels is None:
            emotion_labels = ['Neutral', 'Positive', 'Negative']
        
        print(classification_report(y_test, y_pred, target_names=emotion_labels))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
    
    def save_model(self, model_path, scaler_path):
        """Save trained model and scaler."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        """Load trained model and scaler."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from: {model_path}")


def main():
    """Main training pipeline."""
    # Paths
    data_dir = '/app/eeg_emotion_recognition_1806/data/processed'
    model_dir = '/app/eeg_emotion_recognition_1806/model'
    os.makedirs(model_dir, exist_ok=True)
    
    print("="*60)
    print("EEG EMOTION RECOGNITION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading preprocessed data...")
    X_train = np.load(os.path.join(data_dir, 'X_train_processed.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_processed.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load channel info
    with open(os.path.join(data_dir, 'channel_info.pkl'), 'rb') as f:
        channel_info = pickle.load(f)
    
    sampling_rate = channel_info['sampling_rate']
    emotions = channel_info['emotions']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Data shape: {X_train.shape}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Emotions: {emotions}")
    
    # Extract features
    print("\n" + "="*60)
    print("STEP 1: FEATURE EXTRACTION")
    print("="*60)
    
    extractor = EEGFeatureExtractor(sampling_rate=sampling_rate)
    
    # Extract features for training data
    X_train_features = extractor.extract_all_features(X_train)
    print(f"\nTraining features shape: {X_train_features.shape}")
    
    # Extract features for test data
    X_test_features = extractor.extract_all_features(X_test)
    print(f"Test features shape: {X_test_features.shape}")
    
    # Feature info
    n_channels = X_train.shape[1]
    n_bands = 5
    print(f"\nFeature breakdown:")
    print(f"  - PSD features: {n_channels * n_bands} ({n_channels} channels x {n_bands} bands)")
    print(f"  - DE features: {n_channels * n_bands} ({n_channels} channels x {n_bands} bands)")
    print(f"  - Total features: {X_train_features.shape[1]}")
    
    # Save features
    np.save(os.path.join(data_dir, 'X_train_features.npy'), X_train_features)
    np.save(os.path.join(data_dir, 'X_test_features.npy'), X_test_features)
    print(f"\nFeatures saved to {data_dir}")
    
    # Train models
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    # Train Random Forest
    print("\n--- Random Forest Classifier ---")
    rf_classifier = EEGEmotionClassifier(model_type='random_forest')
    rf_train_acc = rf_classifier.train(X_train_features, y_train)
    rf_results = rf_classifier.evaluate(X_test_features, y_test, emotions)
    
    # Save Random Forest model
    rf_classifier.save_model(
        os.path.join(model_dir, 'emotion_model_rf.pkl'),
        os.path.join(model_dir, 'scaler_rf.pkl')
    )
    
    # Train SVM
    print("\n--- SVM Classifier ---")
    svm_classifier = EEGEmotionClassifier(model_type='svm')
    svm_train_acc = svm_classifier.train(X_train_features, y_train)
    svm_results = svm_classifier.evaluate(X_test_features, y_test, emotions)
    
    # Save SVM model
    svm_classifier.save_model(
        os.path.join(model_dir, 'emotion_model_svm.pkl'),
        os.path.join(model_dir, 'scaler_svm.pkl')
    )
    
    # Compare models
    print("\n" + "="*60)
    print("STEP 3: MODEL COMPARISON")
    print("="*60)
    print(f"\nRandom Forest:")
    print(f"  Training Accuracy: {rf_train_acc:.4f}")
    print(f"  Test Accuracy: {rf_results['accuracy']:.4f}")
    
    print(f"\nSVM:")
    print(f"  Training Accuracy: {svm_train_acc:.4f}")
    print(f"  Test Accuracy: {svm_results['accuracy']:.4f}")
    
    # Select best model
    if rf_results['accuracy'] >= svm_results['accuracy']:
        best_model = 'random_forest'
        best_accuracy = rf_results['accuracy']
        print(f"\nBest model: Random Forest (accuracy: {best_accuracy:.4f})")
    else:
        best_model = 'svm'
        best_accuracy = svm_results['accuracy']
        print(f"\nBest model: SVM (accuracy: {best_accuracy:.4f})")
    
    # Save best model info
    best_model_info = {
        'model_type': best_model,
        'accuracy': best_accuracy,
        'emotions': emotions,
        'sampling_rate': sampling_rate,
        'n_channels': X_train.shape[1],
        'n_features': X_train_features.shape[1]
    }
    
    with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
        pickle.dump(best_model_info, f)
    
    # Check if accuracy meets target
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    if best_accuracy >= 0.60:
        print(f"✓ SUCCESS: Model achieved {best_accuracy*100:.2f}% accuracy (target: >60%)")
    else:
        print(f"⚠ WARNING: Model achieved {best_accuracy*100:.2f}% accuracy (target: >60%)")
        print("  Consider tuning hyperparameters or collecting more data.")
    
    print(f"\nModels saved to: {model_dir}")
    print("="*60)
    
    return best_model, best_accuracy


if __name__ == '__main__':
    main()
