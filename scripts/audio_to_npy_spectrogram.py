import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from pathlib import Path
import librosa

def wav_to_npy(wav_path, output_dir=None, nperseg=1024, noverlap=512):
    """
    Converts a WAV file into a single .npy file with real and imaginary parts of STFT.
    Shape of saved array: (2, freq_bins, time_frames)
    """
    wav_path = Path(wav_path)
    if output_dir is None:
        output_dir = wav_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read audio
    fs, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]  # Convert to mono

    # Perform STFT
    _, _, Zxx = stft(audio, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # # Stack real & imaginary parts
    # combined = np.stack((np.real(Zxx), np.imag(Zxx)), axis=0)

    magnitude = np.abs(Zxx)
    log_mag_spectrogram = librosa.amplitude_to_db(magnitude)

    # Save
    output_file = output_dir / f"{wav_path.stem}.npy"
    np.save(output_file, log_mag_spectrogram)

    print(f"Converted {wav_path.name} -> {output_file} (shape: {log_mag_spectrogram.shape})")


def batch_convert_wav_to_npy(input_dir, output_dir=None, nperseg=1024, noverlap=512):
    """
    Converts all WAV files in input_dir to .npy files with real and imaginary parts of STFT.
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "npy_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    for wav_file in wav_files:
        wav_to_npy(wav_file, output_dir, nperseg=nperseg, noverlap=noverlap)

    print(f"All WAV files converted to {output_dir}")


# Example usage:
# batch_convert_wav_to_npy("path/to/wav_folder", output_dir="path/to/npy_folder")

batch_convert_wav_to_npy("C:/Users/ren_f/Documents/Projects/ENSC429_Group3/data/Samples/vocal", "C:/Users/ren_f/Documents/Projects/ENSC429_Group3/data/vocal")
