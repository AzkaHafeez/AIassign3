"""
Evaluation and Visualization for Assignment 3
Generates comparison plots, confusion matrices, and analysis

Usage:
    python evaluate_results.py --model_path results/best_model.h5 --baseline_acc 0.528 --baseline_f1 0.471 --history_log results/training_history.csv
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import json
import pandas as pd
from datetime import datetime
import librosa
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from model_proposed import ImprovedCNNMFCC

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================
# DATA LOADING (Same as run_experiments.py)
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
            return None
    
    def load_dataset(self, max_duration=5.0):
        audio_data = []
        labels = []
        file_paths = []
        max_length = int(max_duration * self.sample_rate)
        
        for idx, category in enumerate(self.categories):
            category_path = os.path.join(self.data_path, category)
            if not os.path.exists(category_path):
                continue
                
            files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            
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
        
        return np.array(audio_data), np.array(labels), file_paths
    
    def get_class_names(self):
        return self.categories


class MFCCPreprocessor:
    """Basic MFCC preprocessor (no augmentation for evaluation)"""
    
    def __init__(self, sample_rate=16000, n_mfcc=40, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_mfcc(self, audio_array):
        mfcc = librosa.feature.mfcc(
            y=audio_array, sr=self.sample_rate,
            n_mfcc=self.n_mfcc, n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        return mfcc
    
    def process_batch(self, audio_batch, max_time_steps=None):
        mfcc_features = []
        for audio in tqdm(audio_batch, desc="Processing"):
            mfcc = self.extract_mfcc(audio)
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
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 11})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_training_history(history_file, save_path, baseline_acc=None, baseline_f1=None,
                          y_test=None, y_pred=None, class_names=None,
                          improved_acc=None, improved_f1=None):
    """
    Plot comprehensive training analysis with 4 subplots:
    1. Model Accuracy
    2. Model Loss
    3. Per-Class F1 Scores
    4. Performance Comparison: Baseline vs Improved
    """
    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        return
    
    df = pd.read_csv(history_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ============================================================
    # 1. Model Accuracy (Top Left)
    # ============================================================
    axes[0, 0].plot(df['epoch'], df['accuracy'], label='Train', 
                    linewidth=2.5, marker='o', markersize=4, color='#3498db')
    axes[0, 0].plot(df['epoch'], df['val_accuracy'], label='Val', 
                    linewidth=2.5, marker='s', markersize=4, color='#e74c3c')
    if baseline_acc:
        axes[0, 0].axhline(y=baseline_acc, color='gray', linestyle='--', 
                          linewidth=2, alpha=0.7, label='Baseline')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # ============================================================
    # 2. Model Loss (Top Right)
    # ============================================================
    axes[0, 1].plot(df['epoch'], df['loss'], label='Train', 
                    linewidth=2.5, marker='o', markersize=4, color='#3498db')
    axes[0, 1].plot(df['epoch'], df['val_loss'], label='Val', 
                    linewidth=2.5, marker='s', markersize=4, color='#e74c3c')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # ============================================================
    # 3. Per-Class F1 Scores (Bottom Left)
    # ============================================================
    if y_test is not None and y_pred is not None and class_names is not None:
        report = classification_report(y_test, y_pred, target_names=class_names,
                                      output_dict=True, zero_division=0)
        f1_scores = [report[cls]['f1-score'] for cls in class_names]
        
        x = np.arange(len(class_names))
        bars = axes[1, 0].bar(x, f1_scores, color='#2ecc71', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
        
        axes[1, 0].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1.1])
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Threshold')
        axes[1, 0].legend(fontsize=10)
    else:
        axes[1, 0].text(0.5, 0.5, 'Per-Class F1 data not available', 
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
    
    # ============================================================
    # 4. Performance Comparison: Baseline vs Improved (Bottom Right)
    # ============================================================
    if baseline_acc and baseline_f1 and improved_acc and improved_f1:
        metrics = ['Accuracy', 'F1-Score']
        baseline_vals = [baseline_acc, baseline_f1]
        improved_vals = [improved_acc, improved_f1]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = axes[1, 1].bar(x - width/2, baseline_vals, width, 
                              label='Baseline', color='lightcoral', 
                              alpha=0.8, edgecolor='black')
        bars2 = axes[1, 1].bar(x + width/2, improved_vals, width, 
                              label='Improved', color='lightgreen', 
                              alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom', 
                               fontsize=10, fontweight='bold')
        
        # Add improvement annotations
        for i, (baseline, improved) in enumerate(zip(baseline_vals, improved_vals)):
            improvement = (improved - baseline) * 100
            axes[1, 1].annotate(f'+{improvement:.2f}%', 
                               xy=(i, max(baseline, improved) + 0.05),
                               ha='center', fontsize=11, fontweight='bold', 
                               color='darkgreen')
        
        axes[1, 1].set_xlabel('Metric', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Performance Comparison: Baseline vs Improved', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics, fontsize=11)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim([0, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'Comparison data not available', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Performance Comparison: Baseline vs Improved', 
                            fontsize=14, fontweight='bold')
    
    plt.suptitle('Training Analysis - Improved CNN-MFCC', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Plot per-class performance metrics"""
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [report[cls][metric] for cls in class_names] for metric in metrics}
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i*width, data[metric], width, 
                      label=metric.capitalize(), color=colors[i], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics - Improved Model', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_comparison(improved_acc, improved_f1, baseline_acc, baseline_f1, save_path):
    """Plot baseline vs improved comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Accuracy', 'F1-Score']
    baseline_vals = [baseline_acc, baseline_f1]
    improved_vals = [improved_acc, improved_f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, improved_vals, width, label='Improved', 
                   color='lightgreen', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Baseline vs Improved', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
    
    # Add improvement annotations
    for i, (baseline, improved) in enumerate(zip(baseline_vals, improved_vals)):
        improvement = (improved - baseline) * 100
        ax.annotate(f'+{improvement:.2f}%', 
                   xy=(i, max(baseline, improved) + 0.05),
                   ha='center', fontsize=12, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate improved model and generate plots')
    parser.add_argument('--data_path', type=str,
                       default='SAND_Challenge_task1_dataset/task1/training',
                       help='Path to data')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.h5)')
    parser.add_argument('--history_log', type=str,
                       help='Path to training history CSV')
    parser.add_argument('--baseline_acc', type=float, default=0.528,
                       help='Baseline accuracy for comparison')
    parser.add_argument('--baseline_f1', type=float, default=0.471,
                       help='Baseline F1-score for comparison')
    parser.add_argument('--plots_dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed to match training split')
    
    args = parser.parse_args()
    
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*80)
    print("EVALUATING IMPROVED MODEL")
    print("="*80)
    
    # Validate inputs
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if args.history_log and not os.path.isfile(args.history_log):
        print(f"Warning: history_log not found: {args.history_log}. Continuing without training curves.")

    # Load data
    print("\nLoading dataset...")
    loader = AudioDataLoader(args.data_path, sample_rate=16000)
    audio_data, labels, _ = loader.load_dataset(max_duration=5.0)
    class_names = loader.get_class_names()
    
    # Split data (same split as training)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
    _, test_idx = next(sss1.split(audio_data, labels))
    
    X_test = audio_data[test_idx]
    y_test = labels[test_idx]
    
    print(f"Test set: {len(X_test)} samples")
    
    # Extract MFCC
    print("\nExtracting MFCC features...")
    preprocessor = MFCCPreprocessor(sample_rate=16000, n_mfcc=40)
    X_test_mfcc = preprocessor.process_batch(X_test)
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = ImprovedCNNMFCC(num_classes=len(class_names))
    model.load_model(args.model_path)
    
    # Evaluate
    print("\nEvaluating...")
    test_loss, test_acc, test_f1, y_pred = model.evaluate(
        X_test_mfcc, y_test, class_names=class_names
    )
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    # Confusion matrix
    cm_path = os.path.join(args.plots_dir, 'confusion_matrix_improved.pdf')
    plot_confusion_matrix(y_test, y_pred, class_names, cm_path,
                         title=f'Improved Model - Accuracy: {test_acc:.2%} | F1: {test_f1:.4f}')
    
    # Comprehensive training analysis (4 subplots)
    if args.history_log and os.path.exists(args.history_log):
        history_path = os.path.join(args.plots_dir, 'training_analysis.pdf')
        plot_training_history(
            history_file=args.history_log,
            save_path=history_path,
            baseline_acc=args.baseline_acc,
            baseline_f1=args.baseline_f1,
            y_test=y_test,
            y_pred=y_pred,
            class_names=class_names,
            improved_acc=test_acc,
            improved_f1=test_f1
        )
    
    # Per-class metrics (separate detailed plot)
    metrics_path = os.path.join(args.plots_dir, 'per_class_metrics.pdf')
    plot_per_class_metrics(y_test, y_pred, class_names, metrics_path)
    
    # Comparison plot (separate plot)
    comparison_path = os.path.join(args.plots_dir, 'baseline_vs_improved.pdf')
    plot_comparison(test_acc, test_f1, args.baseline_acc, args.baseline_f1, comparison_path)
    
    # Save evaluation results
    eval_results = {
        'model': 'Improved_CNN_MFCC_with_Attention',
        'baseline_results': {
            'accuracy': args.baseline_acc,
            'f1_macro': args.baseline_f1
        },
        'improved_results': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1),
            'loss': float(test_loss)
        },
        'improvement': {
            'accuracy_gain': float((test_acc - args.baseline_acc) * 100),
            'f1_gain': float((test_f1 - args.baseline_f1) * 100),
            'relative_improvement': float((test_acc - args.baseline_acc) / args.baseline_acc * 100)
        },
        'classification_report': classification_report(y_test, y_pred, 
                                                       target_names=class_names,
                                                       output_dict=True,
                                                       zero_division=0)
    }
    
    eval_file = os.path.join(args.results_dir, 'evaluation_results.json')
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"\n✓ Evaluation results saved to {eval_file}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"\nImprovement Summary:")
    print(f"  Baseline Accuracy: {args.baseline_acc:.4f}")
    print(f"  Improved Accuracy: {test_acc:.4f}")
    print(f"  Gain: +{(test_acc - args.baseline_acc)*100:.2f}%")
    print(f"  Baseline F1: {args.baseline_f1:.4f}")
    print(f"  Improved F1: {test_f1:.4f}")
    print(f"  Gain: +{(test_f1 - args.baseline_f1)*100:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()