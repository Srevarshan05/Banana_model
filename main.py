import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import noisereduce as nr
from scipy import signal
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import soundfile as sf
import warnings

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Explicitly set ffmpeg and ffprobe paths
AudioSegment.ffmpeg = r"C:\Users\sreva\Downloads\ffmpeg-2025-05-07-git-1b643e3f65-full_build\ffmpeg-2025-05-07-git-1b643e3f65-full_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\sreva\Downloads\ffmpeg-2025-05-07-git-1b643e3f65-full_build\ffmpeg-2025-05-07-git-1b643e3f65-full_build\bin\ffprobe.exe"

# Parameters
sr = 16000  # Sampling rate for resampling
n_mels = 128  # Number of Mel bands
n_frames = 256  # Fixed time frames for spectrograms
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "wav"), exist_ok=True)

# Step 1: Convert M4A to WAV and Apply Noise Reduction
def convert_and_denoise(m4a_path, wav_path):
    try:
        if not os.path.exists(m4a_path):
            raise FileNotFoundError(f"M4A file not found: {m4a_path}")
        
        # Validate M4A file
        print(f"Validating {m4a_path}...")
        try:
            audio = AudioSegment.from_file(m4a_path, format="m4a")
            if audio.frame_rate is None:
                raise ValueError("M4A file has no valid audio stream")
        except Exception as e:
            raise ValueError(f"Invalid M4A file: {str(e)}")
        
        # Convert M4A to WAV
        print(f"Converting {m4a_path} to WAV...")
        audio = audio.set_frame_rate(sr).set_channels(1)  # Mono, 16kHz
        audio.export(wav_path, format="wav")
        
        # Verify WAV file was created
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file not created: {wav_path}")
        
        # Load WAV
        print(f"Loading WAV file: {wav_path}")
        y, sample_rate = librosa.load(wav_path, sr=sr)
        
        # Apply high-pass filter to remove low-frequency buzz
        print("Applying high-pass filter...")
        sos = signal.butter(10, 100, 'highpass', fs=sr, output='sos')
        y_filtered = signal.sosfilt(sos, y)
        
        # Apply noise reduction
        print("Applying noise reduction...")
        y_denoised = nr.reduce_noise(y=y_filtered, sr=sr, prop_decrease=0.8)
        
        # Save denoised WAV
        print(f"Saving denoised WAV: {wav_path}")
        sf.write(wav_path, y_denoised, sr)
        return y_denoised, sr
    except Exception as e:
        print(f"Error processing {m4a_path}: {str(e)}")
        return None, None

# Step 2: Visualize Audio (Time-Domain and Spectrogram)
def plot_audio(y, sr, stage, file_name):
    try:
        print(f"Generating plots for {file_name}...")
        # Time-domain plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        time = np.arange(len(y)) / sr
        plt.plot(time, y, color='b')
        plt.title(f'Time-Domain Signal ({stage})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()
        
        # Spectrogram
        plt.subplot(2, 1, 2)
        D = librosa.stft(y)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram ({stage})')
        plt.ylim(0, 8000)  # Focus on 0-8kHz (weevil sounds likely below 8kHz)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "plots", f"{stage}_{file_name}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved: {plot_path}")
    except Exception as e:
        print(f"Error plotting {file_name}: {str(e)}")

# Step 3: Extract Mel Spectrogram for Dataset
def extract_mel_spectrogram(y, sr):
    try:
        print("Extracting Mel spectrogram...")
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Resize to fixed shape
        if mel_spec_db.shape[1] > n_frames:
            mel_spec_db = mel_spec_db[:, :n_frames]
        else:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, n_frames - mel_spec_db.shape[1])), mode='constant')
        return mel_spec_db
    except Exception as e:
        print(f"Error extracting spectrogram: {str(e)}")
        return None

# Main Processing Loop
data = []
labels = []
stages = ['no_weevil', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5']

for stage in stages:
    stage_dir = os.path.join('dataset', stage)
    if not os.path.exists(stage_dir):
        print(f"Directory {stage_dir} not found. Skipping.")
        continue
    for m4a_file in glob.glob(os.path.join(stage_dir, '*.m4a')):
        file_name = os.path.basename(m4a_file).replace('.m4a', '')
        wav_path = os.path.join(output_dir, 'wav', f"{stage}_{file_name}.wav")
        
        print(f"Processing: {m4a_file}")
        
        # Convert and denoise
        y, sr = convert_and_denoise(m4a_file, wav_path)
        if y is None or sr is None:
            continue
        
        # Plot
        plot_audio(y, sr, stage, file_name)
        
        # Extract Mel spectrogram
        mel_spec = extract_mel_spectrogram(y, sr)
        if mel_spec is None:
            continue
        data.append(mel_spec)
        labels.append(stage)

# Convert to arrays
if data:
    data = np.array(data)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    
    # Save dataset
    np.save(os.path.join(output_dir, 'mel_spectrograms.npy'), data)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    np.save(os.path.join(output_dir, 'label_encoder_classes.npy'), label_encoder.classes_)
    print(f"Dataset saved in {output_dir}. Plots saved in {output_dir}/plots.")
else:
    print("No data processed. Check input files or errors above.")