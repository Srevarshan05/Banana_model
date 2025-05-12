import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from pydub import AudioSegment

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Parameters
input_shape = (128, 256, 1)  # Mel spectrogram shape
num_classes = 2  # no_weevil, weevil
output_dir = "processed_data/simple_model"
os.makedirs(output_dir, exist_ok=True)
epochs = 15
batch_size = 16
sr = 16000  # Sampling rate
temperature = 2.0  # For confidence scaling

# Explicitly set ffmpeg and ffprobe paths
AudioSegment.ffmpeg = r"C:\Users\sreva\Downloads\ffmpeg-2025-05-07-git-1b643e3f65-full_build\ffmpeg-2025-05-07-git-1b643e3f65-full_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\sreva\Downloads\ffmpeg-2025-05-07-git-1b643e3f65-full_build\ffmpeg-2025-05-07-git-1b643e3f65-full_build\bin\ffprobe.exe"

# Generate Synthetic No-Weevil Data
def generate_synthetic_no_weevil():
    try:
        # Reduced variance for subtler noise
        noise_spec = np.random.normal(0, 0.01, (128, 256))
        return noise_spec
    except Exception as e:
        print(f"Error generating synthetic no_weevil: {str(e)}")
        return None

# Load and Balance Dataset
def load_and_balance_dataset():
    try:
        data_path = 'processed_data/augmented/augmented_mel_spectrograms.npy'
        labels_path = 'processed_data/augmented/augmented_labels.npy'
        
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Augmented dataset files not found in processed_data/augmented/")
        
        data = np.load(data_path)
        labels = np.load(labels_path)
        
        if data.shape[0] != labels.shape[0]:
            raise ValueError("Mismatch between data and labels shapes")
        
        if data.shape[0] == 0:
            raise ValueError("Empty dataset")
        
        # Convert to binary labels: weevil (stage1-5) vs. no_weevil
        binary_labels = np.zeros((labels.shape[0], 2))  # [no_weevil, weevil]
        for i in range(labels.shape[0]):
            if np.argmax(labels[i]) == 0:  # no_weevil
                binary_labels[i] = [1, 0]
            else:  # stage1-5
                binary_labels[i] = [0, 1]
        
        # Balance classes
        no_weevil_count = np.sum(binary_labels[:, 0])
        weevil_count = np.sum(binary_labels[:, 1])
        if no_weevil_count < weevil_count:
            print(f"Balancing dataset: no_weevil={no_weevil_count}, weevil={weevil_count}")
            additional_samples = int(weevil_count - no_weevil_count)
            for _ in range(additional_samples):
                synth_spec = generate_synthetic_no_weevil()
                if synth_spec is not None:
                    data = np.concatenate([data, synth_spec[np.newaxis, ..., np.newaxis]])
                    binary_labels = np.concatenate([binary_labels, np.array([[1, 0]])])
        
        print(f"Balanced dataset: {data.shape[0]} samples with shape {data.shape[1:]}")
        print(f"Class counts: no_weevil={np.sum(binary_labels[:, 0])}, weevil={np.sum(binary_labels[:, 1])}")
        return data, binary_labels
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None

# Build Simple CNN Model
def build_cnn_model():
    try:
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=l2(0.01)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(num_classes)
        ])
        return model
    except Exception as e:
        print(f"Error building model: {str(e)}")
        return None

# Temperature Scaling Layer
class TemperatureScaling(tf.keras.layers.Layer):
    def __init__(self, temperature):
        super(TemperatureScaling, self).__init__()
        self.temperature = temperature
    
    def call(self, inputs):
        return tf.nn.softmax(inputs / self.temperature)

# Process Test Audio File
def process_test_audio(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test file not found: {file_path}")
        
        audio = AudioSegment.from_file(file_path, format="m4a")
        audio = audio.set_frame_rate(sr).set_channels(1)
        wav_path = os.path.join(output_dir, 'test.wav')
        audio.export(wav_path, format="wav")
        
        y, sr_loaded = librosa.load(wav_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if mel_spec_db.shape[1] > 256:
            mel_spec_db = mel_spec_db[:, :256]
        else:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 256 - mel_spec_db.shape[1])), mode='constant')
        
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        mel_spec_db = mel_spec_db[..., np.newaxis]
        return mel_spec_db
    except Exception as e:
        print(f"Error processing test audio {file_path}: {str(e)}")
        return None

# Main Execution
def main():
    # Load and balance dataset
    data, labels = load_and_balance_dataset()
    if data is None or labels is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Split data
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )
        print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    except Exception as e:
        print(f"Error splitting data: {str(e)}")
        return
    
    # Compute class weights (prioritize no_weevil)
    class_counts = np.sum(labels, axis=0)
    class_weights = {0: 1.5, 1: 1.0}  # Higher weight for no_weevil
    print(f"Class weights: {class_weights}")
    
    # Build and train model
    model = build_cnn_model()
    if model is None:
        print("Failed to build model. Exiting.")
        return
    
    # Add temperature scaling
    inputs = tf.keras.Input(shape=input_shape)
    x = model(inputs)
    outputs = TemperatureScaling(temperature)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    try:
        print("Training simple CNN model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate model
        val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_pred, axis=1))
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save model
        model.save(os.path.join(output_dir, 'simple_weevil_cnn.h5'))
        print("Model saved as simple_weevil_cnn.h5")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(os.path.join(output_dir, 'simple_weevil_cnn.tflite'), 'wb') as f:
            f.write(tflite_model)
        print("TFLite model saved as simple_weevil_cnn.tflite")
    except Exception as e:
        print(f"Error training or saving model: {str(e)}")
        return
    
    # Test all stages and no_weevil
    test_files = [
        ('dataset/stage1/Stage 1 (mp3cut.net).m4a', 'weevil'),
        ('dataset/stage2/Stage 2 (mp3cut.net).m4a', 'weevil'),
        ('dataset/stage3/Stage 3 (mp3cut.net).m4a', 'weevil'),
        ('dataset/stage4/Stage 4 (mp3cut.net).m4a', 'weevil'),
        ('dataset/stage5/Stage 5 (mp3cut.net).m4a', 'weevil'),
        ('dataset/no_weevil/no_weevil_sound.m4a', 'no_weevil')
    ]
    
    correct = 0
    total = len(test_files)
    
    try:
        for file_path, expected_class in test_files:
            spec = process_test_audio(file_path)
            if spec is not None:
                pred = model.predict(spec[np.newaxis, ...])
                predicted_class = "weevil" if np.argmax(pred) == 1 else "no_weevil"
                no_weevil_conf = pred[0][0]
                weevil_conf = pred[0][1]
                is_correct = predicted_class == expected_class
                if is_correct:
                    correct += 1
                print(f"File: {file_path}")
                print(f"Predicted class: {predicted_class}")
                print(f"Confidence: no_weevil={no_weevil_conf:.4f}, weevil={weevil_conf:.4f}")
                print(f"Correct: {is_correct}")
        accuracy = correct / total
        print(f"\nTest accuracy: {accuracy:.4f} ({correct}/{total} correct)")
    except Exception as e:
        print(f"Error testing model: {str(e)}")

if __name__ == "__main__":
    main()