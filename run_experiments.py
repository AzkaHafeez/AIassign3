"""
Experimental Setup for Assignment 3
Trains improved CNN-MFCC model with SpecAugment and Attention

Usage:
    python run_experiments.py --data_path /path/to/data --epochs 60
"""

import os
import argparse
import json
import numpy as np
import librosa
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import re

from model_proposed import ImprovedCNNMFCC

# ============================================================
# DATA LOADER
# ============================================================

class AudioDataLoader:
    """Load and preprocess audio dataset"""
    
    def __init__(self, data_path, sample_rate=16000):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.categories = ['phonationA', 'phonationE', 'phonationI', 'phonationO', 
                          'phonationU', 'rhythmKA', 'rhythmPA', 'rhythmTA']
        
    def load_audio_file(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_dataset(self, max_duration=5.0):
        """Load all audio files"""
        audio_data = []
        labels = []
        file_paths = []
        max_length = int(max_duration * self.sample_rate)
        
        print(f"Loading dataset from: {self.data_path}")
        
        for idx, category in enumerate(self.categories):
            category_path = os.path.join(self.data_path, category)
            if not os.path.exists(category_path):
                continue
                
            files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            print(f"  {category}: {len(files)} files")
            
            for file in files:
                file_path = os.path.join(category_path, file)
                audio = self.load_audio_file(file_path)
                
                if audio is not None:
                    if len(audio) < max_length:
                        audio = np.pad(audio, (0, max_length - len(audio)))
                    else:
                        audio = audio[:max_length]
                    
                    audio_data.append(audio)
                    labels.append(idx)
                    file_paths.append(file_path)
        
        print(f"\n✓ Total samples loaded: {len(audio_data)}")
        return np.array(audio_data), np.array(labels), file_paths
    
    def get_class_names(self):
        return self.categories


# ============================================================
# MFCC PREPROCESSOR WITH SPECAUGMENT
# ============================================================

class ImprovedMFCCPreprocessor:
    """
    MFCC Preprocessor with SpecAugment data augmentation
    
    Reference: Park et al. "SpecAugment: A Simple Data Augmentation 
               Method for Automatic Speech Recognition" (2019)
    """
    
    def __init__(self, sample_rate=16000, n_mfcc=40, n_fft=2048, hop_length=512,
                 use_specaugment=True, freq_mask_param=8, time_mask_param=25):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_specaugment = use_specaugment
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        
        print(f"MFCC Preprocessor initialized:")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  N_MFCC: {n_mfcc}")
        print(f"  SpecAugment: {'ENABLED' if use_specaugment else 'Disabled'}")
    
    def spec_augment(self, mfcc):
        """Apply SpecAugment to MFCC features"""
        mfcc_aug = mfcc.copy()
        
        # Frequency masking
        freq_mask = np.random.randint(0, self.freq_mask_param)
        f0 = np.random.randint(0, max(1, mfcc.shape[0] - freq_mask))
        mfcc_aug[f0:f0+freq_mask, :] = 0
        
        # Time masking
        time_mask = np.random.randint(0, self.time_mask_param)
        t0 = np.random.randint(0, max(1, mfcc.shape[1] - time_mask))
        mfcc_aug[:, t0:t0+time_mask] = 0
        
        return mfcc_aug
    
    def extract_mfcc(self, audio_array, apply_augment=False):
        """Extract MFCC with optional SpecAugment"""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_array, sr=self.sample_rate,
                n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Normalize
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            
            # Apply SpecAugment (only during training)
            if apply_augment and self.use_specaugment:
                mfcc = self.spec_augment(mfcc)
            
            return mfcc
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return None
    
    def process_batch(self, audio_batch, max_time_steps=None, apply_augment=False):
        """Process batch of audio arrays"""
        desc = "Processing (with SpecAugment)" if apply_augment else "Processing"
        
        mfcc_features = []
        for audio in tqdm(audio_batch, desc=desc):
            mfcc = self.extract_mfcc(audio, apply_augment=apply_augment)
            if mfcc is not None:
                mfcc_features.append(mfcc)
        
        if max_time_steps is None:
            max_time_steps = max([f.shape[1] for f in mfcc_features])
        
        processed = []
        for mfcc in mfcc_features:
            if mfcc.shape[1] < max_time_steps:
                mfcc = np.pad(mfcc, ((0, 0), (0, max_time_steps - mfcc.shape[1])))
            else:
                mfcc = mfcc[:, :max_time_steps]
            processed.append(mfcc)
        
        processed = np.array(processed)
        processed = np.transpose(processed, (0, 2, 1))
        processed = np.expand_dims(processed, axis=-1)
        
        return processed


# ============================================================
# DATA SPLITTING
# ============================================================

def create_data_split(audio_data, labels, test_size=0.30, val_size=0.20, random_state=42):
    """Create stratified train/val/test split"""
    print("\n" + "="*60)
    print("Creating data split...")
    print("="*60)
    
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss1.split(audio_data, labels))
    
    X_temp = audio_data[train_val_idx]
    y_temp = labels[train_val_idx]
    X_test = audio_data[test_idx]
    y_test = labels[test_idx]
    
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=random_state)
    train_idx, val_idx = next(sss2.split(X_temp, y_temp))
    
    X_train = X_temp[train_idx]
    y_train = y_temp[train_idx]
    X_val = X_temp[val_idx]
    y_val = y_temp[val_idx]
    
    print(f"  Train: {len(X_train)} samples (~56%)")
    print(f"  Val: {len(X_val)} samples (~14%)")
    print(f"  Test: {len(X_test)} samples (~30%)")
    print("="*60)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# GPU SETUP
# ============================================================

def setup_gpu():
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU Available: {len(gpus)} GPU(s)")
            print(f"  Device: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("⚠ No GPU found. Using CPU.")
    
    print(f"TensorFlow version: {tf.__version__}")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Run improved CNN-MFCC experiments')
    parser.add_argument('--data_path', type=str, 
                       default='SAND_Challenge_task1_dataset/task1/training',
                       help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--max_duration', type=float, default=5.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--n_mfcc', type=int, default=40,
                       help='Number of MFCC coefficients')
    parser.add_argument('--use_specaugment', action='store_true', default=True,
                       help='Use SpecAugment data augmentation')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("="*80)
    print("ASSIGNMENT 3: IMPROVED CNN-MFCC EXPERIMENTS")
    print("="*80)
    
    setup_gpu()
    
    # Load data
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    loader = AudioDataLoader(args.data_path, sample_rate=16000)
    audio_data, labels, file_paths = loader.load_dataset(max_duration=args.max_duration)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_split(
        audio_data, labels, test_size=0.30, val_size=0.20
    )
    
    # Extract MFCC features
    print("\n" + "="*60)
    print("Extracting MFCC features...")
    print("="*60)
    preprocessor = ImprovedMFCCPreprocessor(
        sample_rate=16000,
        n_mfcc=args.n_mfcc,
        use_specaugment=args.use_specaugment
    )
    
    X_train_mfcc = preprocessor.process_batch(X_train, apply_augment=True)
    X_val_mfcc = preprocessor.process_batch(X_val, max_time_steps=X_train_mfcc.shape[1], apply_augment=False)
    X_test_mfcc = preprocessor.process_batch(X_test, max_time_steps=X_train_mfcc.shape[1], apply_augment=False)
    
    print(f"\nFeature shapes:")
    print(f"  Train: {X_train_mfcc.shape}")
    print(f"  Val: {X_val_mfcc.shape}")
    print(f"  Test: {X_test_mfcc.shape}")
    
    # Build and train model
    print("\n" + "="*60)
    print("Building and training model...")
    print("="*60)
    
    input_shape = X_train_mfcc.shape[1:]
    model = ImprovedCNNMFCC(
        num_classes=len(loader.get_class_names()),
        learning_rate=args.learning_rate
    )
    model.build_model(input_shape)
    
    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(args.results_dir, f'best_model_improved_{timestamp}.h5')
    log_path = os.path.join(args.results_dir, f'training_log_improved_{timestamp}.csv')
    
    # Train
    history = model.train(
        X_train_mfcc, y_train,
        X_val_mfcc, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=checkpoint_path,
        log_path=log_path
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    test_loss, test_acc, test_f1, y_pred = model.evaluate(
        X_test_mfcc, y_test, 
        batch_size=args.batch_size,
        class_names=loader.get_class_names()
    )
    
    # Save results
    results = {
        'model': 'Improved_CNN_MFCC_with_Attention',
        'assignment': 'Assignment 3',
        'hyperparameters': model.get_hyperparameters(),
        'training_args': vars(args),
        'test_results': {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1)
        },
        'training_results': {
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_val_loss': float(min(history.history['val_loss'])),
            'total_epochs': len(history.history['loss'])
        },
        'timestamp': timestamp
    }
    
    results_file = os.path.join(args.results_dir, f'results_improved_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to {results_file}")
    print(f"✓ Model saved to {checkpoint_path}")
    print(f"✓ Training log saved to {log_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Final Test F1-Score: {test_f1:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()