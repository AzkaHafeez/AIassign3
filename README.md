# Audio Classification - Improved CNN-MFCC Model

## Overview
This project implements an improved CNN-MFCC model with Temporal Attention and SpecAugment for audio classification. It's designed for Assignment 3, focusing on incremental improvements over a baseline model.

## Key Features
- **Temporal Attention Mechanism**: Helps the model focus on important time steps in audio
- **SpecAugment Data Augmentation**: Improves generalization through frequency and time masking
- **Optimized Architecture**: 3 convolutional blocks with batch normalization and regularization
- **Expected Improvement**: +10-12% accuracy over baseline

## Requirements

```bash
pip install tensorflow numpy librosa scikit-learn matplotlib seaborn pandas tqdm
```

### Detailed Requirements
- Python 
- TensorFlow 
- librosa
- scikit-learn
- matplotlib
- seaborn
- pandas
- tqdm

## Project Structure

```
.
├── run_experiments.py          # Main training script
├── model_proposed.py           # Improved CNN-MFCC model with Attention
├── evaluate_results.py         # Evaluation and visualization
├── README.md                   # This file
├── SAND_Challenge_task1_dataset/
│   └── task1/
│       └── training/           # Audio dataset
│           ├── phonationA/
│           ├── phonationE/
│           ├── phonationI/
│           ├── phonationO/
│           ├── phonationU/
│           ├── rhythmKA/
│           ├── rhythmPA/
│           └── rhythmTA/
├── results/                    # Training outputs (auto-created)
└── plots/                      # Visualizations (auto-created)
```

## Usage

### 1. Training the Model

**Basic usage:**
```bash
python run_experiments.py --data_path SAND_Challenge_task1_dataset/task1/training
```

**With custom parameters:**
```bash
python run_experiments.py \
    --data_path SAND_Challenge_task1_dataset/task1/training \
    --epochs 60 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --n_mfcc 40 \
    --use_specaugment \
    --results_dir results
```

**Available arguments:**
- `--data_path`: Path to training data directory (default: `SAND_Challenge_task1_dataset/task1/training`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 60)
- `--learning_rate`: Learning rate (default: 0.0005)
- `--max_duration`: Maximum audio duration in seconds (default: 5.0)
- `--n_mfcc`: Number of MFCC coefficients (default: 40)
- `--use_specaugment`: Enable SpecAugment augmentation (default: True)
- `--results_dir`: Directory to save results (default: `results`)

### 2. Evaluating the Model

**Basic evaluation:**
```bash
python evaluate_results.py \
    --model_path results/best_model_improved_20241115_143022.h5 \
    --baseline_acc 0.528
```

**Full evaluation with all options:**
```bash
python evaluate_results.py \
    --data_path SAND_Challenge_task1_dataset/task1/training \
    --model_path results/best_model_improved_20241115_143022.h5 \
    --history_log results/training_log_improved_20241115_143022.csv \
    --baseline_acc 0.528 \
    --baseline_f1 0.471 \
    --plots_dir plots \
    --results_dir results
```

**Available arguments:**
- `--data_path`: Path to data directory (default: `SAND_Challenge_task1_dataset/task1/training`)
- `--model_path`: Path to trained model (.h5) **[REQUIRED]**
- `--history_log`: Path to training history CSV file
- `--baseline_acc`: Baseline accuracy for comparison (default: 0.528)
- `--baseline_f1`: Baseline F1-score for comparison (default: 0.471)
- `--plots_dir`: Directory to save plots (default: `plots`)
- `--results_dir`: Directory to save results (default: `results`)

## Output Files

### Training Outputs (`results/` directory)
1. **Best Model**: `best_model_improved_YYYYMMDD_HHMMSS.h5`
   - Saved weights of the best model based on validation accuracy

2. **Training Log**: `training_log_improved_YYYYMMDD_HHMMSS.csv`
   - Epoch-by-epoch metrics (loss, accuracy, learning rate)

3. **Results JSON**: `results_improved_YYYYMMDD_HHMMSS.json`
   - Hyperparameters, test metrics, and training summary

### Evaluation Outputs (`plots/` directory)
1. **Confusion Matrix**: `confusion_matrix_improved.pdf`
   - Detailed per-class prediction analysis

2. **Training History**: `training_history.pdf`
   - Accuracy, loss, learning rate, and overfitting analysis

3. **Per-Class Metrics**: `per_class_metrics.pdf`
   - Precision, recall, and F1-score for each class

4. **Baseline Comparison**: `baseline_vs_improved.pdf`
   - Side-by-side comparison with baseline model

5. **Evaluation Results**: `evaluation_results.json`
   - Detailed metrics and classification report

## Model Architecture

```
Input (Time Steps × 40 MFCCs × 1)
    ↓
Conv2D (16 filters) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv2D (32 filters) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv2D (64 filters) → BatchNorm
    ↓
Reshape → Temporal Attention
    ↓
Dropout(0.5) → Dense(128) → Dropout(0.5)
    ↓
Output (8 classes, Softmax)
```

## Key Improvements Over Baseline

1. **Temporal Attention**: Focuses on important audio segments
2. **SpecAugment**: Robust data augmentation (frequency + time masking)
3. **Better Regularization**: L2 regularization + strategic dropout
4. **Optimized Architecture**: Designed specifically for audio features

## Example Workflow

```bash
# Step 1: Train the model
python run_experiments.py \
    --data_path SAND_Challenge_task1_dataset/task1/training \
    --epochs 60 \
    --batch_size 32

# Step 2: Evaluate and visualize (use the actual filename from Step 1)
python evaluate_results.py \
    --model_path results/best_model_improved_20241115_143022.h5 \
    --history_log results/training_log_improved_20241115_143022.csv \
    --baseline_acc 0.528 \
    --baseline_f1 0.471
```

## Expected Results

- **Baseline Accuracy**: ~52.8%
- **Improved Accuracy**: ~64-66% (+10-12% improvement)
- **Training Time**: ~15-20 minutes (GPU) / ~2-3 hours (CPU)
- **Model Size**: ~50K parameters

## GPU Support

The code automatically detects and uses GPU if available:
- Recommended: NVIDIA GPU with CUDA support
- Automatic memory growth enabled
- Falls back to CPU if no GPU found

## Troubleshooting

**Issue: Out of Memory**
```bash
# Reduce batch size
python run_experiments.py --batch_size 16
```

**Issue: No module named 'librosa'**
```bash
pip install librosa
```

**Issue: Model file not found**
```bash
# Check the results directory for the exact filename
ls -lh results/
```

**Issue: Audio files not loading**
- Verify the dataset path is correct
- Ensure audio files are in .wav format
- Check folder structure matches the expected format

## Notes

- The model uses stratified train/val/test splits (56%/14%/30%)
- All audio is resampled to 16kHz and padded/truncated to 5 seconds
- MFCC features are normalized per sample
- Early stopping and learning rate reduction are applied automatically

## License

This is an academic assignment project.
