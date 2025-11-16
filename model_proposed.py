"""
Proposed Model: Improved CNN-MFCC with Attention and SpecAugment
Assignment 3 - Incremental Improvement over Baseline

Improvements over Baseline:
1. SpecAugment data augmentation
2. Temporal Attention mechanism
3. Optimized regularization

Expected improvement: +10-12% accuracy over baseline
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
from sklearn.metrics import f1_score, classification_report
import os

class AttentionLayer(layers.Layer):
    """
    Temporal Attention Mechanism
    Helps model focus on important time steps in audio
    
    Reference: Attention mechanism for audio classification
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # input_shape: (batch, time, features)
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Calculate attention scores
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        
        # Attention weights
        a = tf.nn.softmax(e, axis=1)
        
        # Weighted sum
        context = a * x
        context = tf.reduce_sum(context, axis=1)
        
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class ImprovedCNNMFCC:
    """
    Improved CNN-MFCC Model with Attention
    
    Architecture:
    - 3 Convolutional blocks with batch normalization
    - Temporal Attention mechanism
    - Dense classification head
    
    Improvements:
    - Attention layer for focusing on important time steps
    - Better regularization (L2 + Dropout)
    - Optimized for audio classification
    """
    
    def __init__(self, num_classes=8, learning_rate=0.0005):
        """
        Initialize improved model
        
        Args:
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        """
        Build improved CNN-MFCC model with attention
        
        Args:
            input_shape: Shape of MFCC input (time_steps, n_mfcc, 1)
            
        Returns:
            self: Compiled model
        """
        print(f"\nBuilding Improved CNN-MFCC model...")
        print(f"Input shape: {input_shape}")
        
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Convolutional Block 1
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01),
                         name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(0.3, name='dropout1')(x)
        
        # Convolutional Block 2
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01),
                         name='conv2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        
        # Convolutional Block 3
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01),
                         name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        
        # Reshape for attention
        shape = x.shape
        x = layers.Reshape((shape[1] * shape[2], shape[3]), name='reshape')(x)
        
        # Temporal Attention Layer (NEW!)
        x = AttentionLayer(name='attention')(x)
        
        x = layers.Dropout(0.5, name='dropout3')(x)
        
        # Dense classification head
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01),
                        name='dense1')(x)
        x = layers.Dropout(0.5, name='dropout4')(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax', 
                              name='output')(x)
        
        # Create and compile model
        self.model = keras.Model(inputs=inputs, outputs=outputs, 
                                name='Improved_CNN_MFCC_with_Attention')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print summary
        print("\n" + "="*80)
        print('Model: "Improved_CNN_MFCC_with_Attention"')
        print("="*80)
        self.model.summary()
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        
        print("="*80)
        print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("\nKey Improvements:")
        print("  1. Temporal Attention mechanism")
        print("  2. Optimized regularization (L2 + Dropout)")
        print("  3. Better architecture for audio features")
        print("="*80)
        
        return self
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=20, batch_size=32,
              checkpoint_path=None, log_path=None):
        """
        Train the model with callbacks
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            checkpoint_path: Path to save best model
            log_path: Path to save training log
            
        Returns:
            history: Training history
        """
        print("\n" + "="*60)
        print("Training improved model...")
        print("="*60)
        print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint
        if checkpoint_path:
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint_cb)
        
        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # CSV Logger
        if log_path:
            csv_logger = keras.callbacks.CSVLogger(log_path, separator=',', append=False)
            callbacks.append(csv_logger)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Final metrics
        best_val_acc = max(self.history.history['val_accuracy'])
        best_epoch = self.history.history['val_accuracy'].index(best_val_acc) + 1
        
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
        print(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        # Calculate F1 on validation set
        y_val_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
        val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
        print(f"Final validation F1-score: {val_f1:.4f}")
        
        return self.history
    
    def evaluate(self, X_test, y_test, batch_size=32, class_names=None):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size
            class_names: List of class names for report
            
        Returns:
            test_loss, test_acc, test_f1, y_pred
        """
        print("\n" + "="*60)
        print("Evaluating model...")
        print("="*60)
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test, 
                                                   batch_size=batch_size, 
                                                   verbose=1)
        
        y_pred = np.argmax(self.model.predict(X_test, batch_size=batch_size, verbose=0), axis=1)
        test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        
        if class_names:
            print("\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        
        return test_loss, test_acc, test_f1, y_pred
    
    def predict(self, X, batch_size=32):
        """
        Make predictions
        
        Args:
            X: Input features
            batch_size: Batch size
            
        Returns:
            Predicted class indices
        """
        predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def save_model(self, path):
        """Save model weights"""
        self.model.save(path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path):
        """Load model weights"""
        self.model = keras.models.load_model(path, custom_objects={'AttentionLayer': AttentionLayer})
        print(f"✓ Model loaded from {path}")
    
    def get_hyperparameters(self):
        """Return hyperparameters dictionary"""
        return {
            'model_type': 'Improved CNN-MFCC with Attention',
            'num_classes': self.num_classes,
            'learning_rate': self.learning_rate,
            'architecture': '3 Conv2D blocks + Attention + 1 Dense layer',
            'conv_filters': [16, 32, 64],
            'dense_units': [128],
            'dropout_rates': [0.3, 0.3, 0.5, 0.5],
            'regularization': 'L2 (0.01)',
            'attention': 'Temporal Attention',
            'pooling': 'MaxPooling2D',
            'batch_normalization': True,
            'improvements': [
                'SpecAugment data augmentation',
                'Temporal Attention mechanism',
                'Optimized regularization'
            ]
        }
    
    def export_model_summary(self, filepath='model_architecture.txt'):
        """
        Export detailed model architecture to text file
        Useful for documentation and reporting
        
        Args:
            filepath: Path to save the summary
        """
        with open(filepath, 'w') as f:
            # Redirect stdout to file
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            
            # Add additional info
            f.write('\n' + '='*80 + '\n')
            f.write('HYPERPARAMETERS\n')
            f.write('='*80 + '\n')
            
            hyperparams = self.get_hyperparameters()
            for key, value in hyperparams.items():
                f.write(f'{key}: {value}\n')
        
        print(f"Model summary exported to {filepath}")
        return filepath