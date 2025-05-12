import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Parameters
input_shape = (128, 256, 1)  # Mel spectrogram shape
num_classes = 6  # no_weevil, stage1, ..., stage5
augmentations_per_sample = 20  # Number of augmented samples per original
output_dir = "processed_data/augmented"
os.makedirs(output_dir, exist_ok=True)
epochs = 50
batch_size = 32

# Data Augmentation Functions
def augment_spectrogram(spec):
    """Apply augmentation to Mel spectrogram."""
    try:
        # Time stretch (simulate by resizing time axis)
        time_stretch = np.random.uniform(0.9, 1.1)
        new_length = int(spec.shape[1] * time_stretch)
        if new_length > spec.shape[1]:
            spec = np.pad(spec, ((0, 0), (0, new_length - spec.shape[1])), mode='constant')
        else:
            spec = spec[:, :new_length]
        
        # Pitch shift (simulate by shifting frequency bins)
        shift = np.random.randint(-5, 6)
        spec = np.roll(spec, shift, axis=0)
        if shift > 0:
            spec[:shift, :] = 0
        else:
            spec[shift:, :] = 0
        
        # Add noise
        noise = np.random.normal(0, 0.01, spec.shape)
        spec = spec + noise
        
        # Ensure shape
        if spec.shape[1] > 256:
            spec = spec[:, :256]
        else:
            spec = np.pad(spec, ((0, 0), (0, 256 - spec.shape[1])), mode='constant')
        
        return spec
    except Exception as e:
        print(f"Error augmenting spectrogram: {str(e)}")
        return None

# Load and Validate Dataset
def load_dataset():
    try:
        data_path = 'processed_data/mel_spectrograms.npy'
        labels_path = 'processed_data/labels.npy'
        
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Dataset files not found in processed_data/")
        
        data = np.load(data_path)
        labels = np.load(labels_path)
        
        if data.shape[0] != labels.shape[0]:
            raise ValueError("Mismatch between data and labels shapes")
        
        if data.shape[0] == 0:
            raise ValueError("Empty dataset")
        
        print(f"Loaded {data.shape[0]} samples with shape {data.shape[1:]}")
        return data, labels
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None

# Generate Augmented Dataset
def generate_augmented_dataset(data, labels):
    try:
        augmented_data = []
        augmented_labels = []
        
        for i in range(data.shape[0]):
            spec = data[i]
            label = labels[i]
            
            # Original sample
            augmented_data.append(spec)
            augmented_labels.append(label)
            
            # Augmented samples
            for _ in range(augmentations_per_sample):
                aug_spec = augment_spectrogram(spec)
                if aug_spec is not None:
                    augmented_data.append(aug_spec)
                    augmented_labels.append(label)
        
        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels)
        
        # Add channel dimension
        augmented_data = augmented_data[..., np.newaxis]
        
        # Normalize to [0, 1]
        augmented_data = (augmented_data - augmented_data.min()) / (augmented_data.max() - augmented_data.min())
        
        # Balance classes
        class_counts = np.sum(augmented_labels, axis=0)
        print(f"Class counts: {class_counts}")
        if np.any(class_counts == 0):
            print("Warning: Some classes have no samples (e.g., no_weevil). Collect more data for accuracy.")
        
        # Save augmented dataset
        np.save(os.path.join(output_dir, 'augmented_mel_spectrograms.npy'), augmented_data)
        np.save(os.path.join(output_dir, 'augmented_labels.npy'), augmented_labels)
        print(f"Augmented dataset saved in {output_dir}: {augmented_data.shape[0]} samples")
        
        return augmented_data, augmented_labels
    except Exception as e:
        print(f"Error generating augmented dataset: {str(e)}")
        return None, None

# Build CNN Model
def build_cnn_model():
    try:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error building model: {str(e)}")
        return None

# Main Execution
def main():
    # Load dataset
    data, labels = load_dataset()
    if data is None or labels is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Generate augmented dataset
    augmented_data, augmented_labels = generate_augmented_dataset(data, labels)
    if augmented_data is None or augmented_labels is None:
        print("Failed to generate augmented dataset. Exiting.")
        return
    
    # Split data
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            augmented_data, augmented_labels, test_size=0.2, random_state=42
        )
        print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    except Exception as e:
        print(f"Error splitting data: {str(e)}")
        return
    
    # Build and train model
    model = build_cnn_model()
    if model is None:
        print("Failed to build model. Exiting.")
        return
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    try:
        print("Training CNN model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val.argmax(axis=1), val_pred.argmax(axis=1), average='weighted')
        print(f"Validation F1-score: {val_f1:.4f}")
        
        # Save model
        model.save(os.path.join(output_dir, 'weevil_cnn.h5'))
        print("Model saved as weevil_cnn.h5")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(os.path.join(output_dir, 'weevil_cnn.tflite'), 'wb') as f:
            f.write(tflite_model)
        print("TFLite model saved as weevil_cnn.tflite")
    except Exception as e:
        print(f"Error training or saving model: {str(e)}")

if __name__ == "__main__":
    main()