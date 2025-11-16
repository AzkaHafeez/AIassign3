"""
Advanced Model Analysis and Performance Profiling
Provides detailed insights into model behavior, error analysis, and optimization suggestions

Author: Your Name
Date: November 2024
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc
)
import argparse
from collections import Counter


class ModelAnalyzer:
    """
    Comprehensive model analysis toolkit
    Provides insights beyond basic accuracy metrics
    """
    
    def __init__(self, results_dir='results', plots_dir='plots'):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
        
    def load_training_history(self, history_file):
        """Load training history from CSV"""
        if not os.path.exists(history_file):
            raise FileNotFoundError(f"History file not found: {history_file}")
        return pd.read_csv(history_file)
    
    def analyze_training_dynamics(self, history_df, save_path=None):
        """
        Analyze training dynamics and identify potential issues
        """
        print("\n" + "="*80)
        print("TRAINING DYNAMICS ANALYSIS")
        print("="*80)
        
        # Detect overfitting
        final_train_acc = history_df['accuracy'].iloc[-1]
        final_val_acc = history_df['val_accuracy'].iloc[-1]
        overfit_gap = final_train_acc - final_val_acc
        
        print(f"\nOverfitting Analysis:")
        print(f"  Final Train Accuracy: {final_train_acc:.4f}")
        print(f"  Final Val Accuracy: {final_val_acc:.4f}")
        print(f"  Gap (Train - Val): {overfit_gap:.4f}")
        
        if overfit_gap > 0.1:
            print("WARNING: Significant overfitting detected!")
            print("  Suggestions:")
            print("    - Increase dropout rates")
            print("    - Add more data augmentation")
            print("    - Reduce model complexity")
        elif overfit_gap > 0.05:
            print("Moderate overfitting (acceptable)")
        else:
            print("Good generalization")
        
        # Learning rate analysis
        if 'lr' in history_df.columns:
            print(f"\nLearning Rate Schedule:")
            print(f"  Initial LR: {history_df['lr'].iloc[0]:.6f}")
            print(f"  Final LR: {history_df['lr'].iloc[-1]:.6f}")
            lr_reductions = (history_df['lr'].diff() < 0).sum()
            print(f"  LR Reductions: {lr_reductions}")
        
        # Convergence analysis
        best_epoch = history_df['val_accuracy'].idxmax() + 1
        total_epochs = len(history_df)
        
        print(f"\nConvergence Analysis:")
        print(f"  Best Epoch: {best_epoch}/{total_epochs}")
        print(f"  Early Stopping Gap: {total_epochs - best_epoch} epochs")
        
        if total_epochs - best_epoch > 20:
            print("Model stopped improving significantly")
            print("  Suggestion: Consider earlier stopping")
        
        # Training stability
        val_acc_std = history_df['val_accuracy'].std()
        print(f"\nTraining Stability:")
        print(f"  Val Accuracy Std Dev: {val_acc_std:.4f}")
        
        if val_acc_std > 0.05:
            print("High variance in validation accuracy")
            print("  Suggestions:")
            print("    - Increase batch size")
            print("    - Use learning rate warmup")
            print("    - Check data quality")
        else:
            print("Stable training")
        
        return {
            'overfit_gap': overfit_gap,
            'best_epoch': best_epoch,
            'val_acc_std': val_acc_std
        }
    
    def analyze_per_class_errors(self, y_true, y_pred, class_names):
        """
        Detailed per-class error analysis
        """
        print("\n" + "="*80)
        print("PER-CLASS ERROR ANALYSIS")
        print("="*80)
        
        cm = confusion_matrix(y_true, y_pred)
        
        for i, class_name in enumerate(class_names):
            true_positives = cm[i, i]
            false_negatives = cm[i, :].sum() - true_positives
            false_positives = cm[:, i].sum() - true_positives
            
            total_actual = cm[i, :].sum()
            accuracy = true_positives / total_actual if total_actual > 0 else 0
            
            print(f"\n{class_name}:")
            print(f"  Samples: {total_actual}")
            print(f"  Correct: {true_positives} ({accuracy*100:.1f}%)")
            print(f"  False Negatives: {false_negatives}")
            print(f"  False Positives: {false_positives}")
            
            # Most common misclassifications
            if false_negatives > 0:
                misclass = []
                for j in range(len(class_names)):
                    if i != j and cm[i, j] > 0:
                        misclass.append((class_names[j], cm[i, j]))
                
                misclass.sort(key=lambda x: x[1], reverse=True)
                
                if misclass:
                    print(f"  Most Confused With:")
                    for confused_class, count in misclass[:3]:
                        print(f"    - {confused_class}: {count} times")
        
    def generate_detailed_report(self, y_true, y_pred, class_names, save_path):
        """
        Generate comprehensive classification report with insights
        """
        report_dict = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Create detailed report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("DETAILED CLASSIFICATION REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Overall metrics
        report_lines.append("Overall Performance:")
        report_lines.append(f"  Accuracy: {report_dict['accuracy']:.4f}")
        report_lines.append(f"  Macro Avg Precision: {report_dict['macro avg']['precision']:.4f}")
        report_lines.append(f"  Macro Avg Recall: {report_dict['macro avg']['recall']:.4f}")
        report_lines.append(f"  Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
        report_lines.append("")
        
        # Per-class analysis
        report_lines.append("Per-Class Performance:")
        report_lines.append("-"*80)
        
        for class_name in class_names:
            metrics = report_dict[class_name]
            report_lines.append(f"\n{class_name}:")
            report_lines.append(f"  Support: {int(metrics['support'])} samples")
            report_lines.append(f"  Precision: {metrics['precision']:.4f}")
            report_lines.append(f"  Recall: {metrics['recall']:.4f}")
            report_lines.append(f"  F1-Score: {metrics['f1-score']:.4f}")
            
            # Performance assessment
            if metrics['f1-score'] >= 0.70:
                assessment = "Excellent"
            elif metrics['f1-score'] >= 0.60:
                assessment = "✓ Good"
            elif metrics['f1-score'] >= 0.50:
                assessment = "Acceptable"
            else:
                assessment = "Needs Improvement"
            
            report_lines.append(f"  Assessment: {assessment}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nDetailed report saved to: {save_path}")
        
        # Print to console
        print('\n'.join(report_lines))
    
    def plot_error_distribution(self, y_true, y_pred, class_names, save_path):
        """
        Visualize error distribution across classes
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate error rates
        error_rates = []
        for i in range(len(class_names)):
            total = cm[i, :].sum()
            errors = total - cm[i, i]
            error_rate = errors / total if total > 0 else 0
            error_rates.append(error_rate * 100)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['green' if er < 30 else 'orange' if er < 40 else 'red' 
                  for er in error_rates]
        
        bars = ax.bar(range(len(class_names)), error_rates, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, rate in zip(bars, error_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Error Rates', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Target (<30%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to: {save_path}")
        plt.close()
    
    def suggest_improvements(self, analysis_results):
        """
        Provide actionable improvement suggestions based on analysis
        """
        print("\n" + "="*80)
        print("IMPROVEMENT SUGGESTIONS")
        print("="*80)
        
        suggestions = []
        
        # Based on overfitting
        if analysis_results['overfit_gap'] > 0.1:
            suggestions.append({
                'issue': 'High overfitting',
                'suggestions': [
                    'Increase dropout from 0.5 to 0.6-0.7',
                    'Add more SpecAugment (increase freq/time mask params)',
                    'Collect more training data',
                    'Reduce model complexity (fewer filters)'
                ]
            })
        
        # Based on training stability
        if analysis_results['val_acc_std'] > 0.05:
            suggestions.append({
                'issue': 'Training instability',
                'suggestions': [
                    'Increase batch size to 64',
                    'Add learning rate warmup (5-10 epochs)',
                    'Use gradient clipping (max_norm=1.0)',
                    'Check for data quality issues'
                ]
            })
        
        # Based on early stopping
        epochs_after_best = 60 - analysis_results['best_epoch']
        if epochs_after_best > 15:
            suggestions.append({
                'issue': 'Model stopped improving early',
                'suggestions': [
                    'Reduce early stopping patience to 10',
                    'Try different learning rate (0.0003 or 0.0007)',
                    'Experiment with different optimizers (SGD with momentum)',
                    'Add cyclical learning rate schedule'
                ]
            })
        
        # Print suggestions
        if suggestions:
            for i, item in enumerate(suggestions, 1):
                print(f"\n{i}. {item['issue']}:")
                for suggestion in item['suggestions']:
                    print(f"   • {suggestion}")
        else:
            print("\nTraining appears optimal! Minor tuning may still help:")
            print("   • Fine-tune learning rate slightly")
            print("   • Experiment with different MFCC parameters")
            print("   • Try ensemble methods")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Advanced model analysis and performance profiling'
    )
    parser.add_argument('--history_log', type=str, required=True,
                       help='Path to training history CSV')
    parser.add_argument('--results_json', type=str,
                       help='Path to results JSON file')
    parser.add_argument('--plots_dir', type=str, default='plots',
                       help='Directory to save analysis plots')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(plots_dir=args.plots_dir)
    
    print("="*80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Load training history
    print("\nLoading training history...")
    history_df = analyzer.load_training_history(args.history_log)
    
    # Analyze training dynamics
    analysis_results = analyzer.analyze_training_dynamics(history_df)
    
    # Generate improvement suggestions
    analyzer.suggest_improvements(analysis_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {args.plots_dir}/")
    print("\nNext steps:")
    print("  1. Review the improvement suggestions above")
    print("  2. Check the generated plots for visual insights")
    print("  3. Implement suggested changes incrementally")
    print("  4. Re-train and compare results")
    print("="*80)


if __name__ == '__main__':
    main()