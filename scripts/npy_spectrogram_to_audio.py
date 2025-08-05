import numpy as np
import pandas as pd
import librosa
import os
from scipy.signal import istft
from scipy.io import wavfile
from pathlib import Path
import soundfile as sf

def reconstruct_audio_from_stft_npy(npy_path, output_path):
    """
    Reconstruct audio file from a given .npy file containing a log magnitude spectrogram
    :param npy_path: directory of .npy file
    :param output_path: directory to save output audio file
    """
    Zxx = np.load(npy_path)
    magnitude = librosa.db_to_amplitude(Zxx)
    y = librosa.griffinlim(magnitude, n_iter=32, hop_length=512, win_length=1024)
    y = y/np.max(np.abs(y))

    # save output to WAV
    # sf.write("reconstructed.wav", y, 44100)
    sf.write(str(output_path), y, 44100)

def reconstruct_audio_from_directory(npy_dir, output_dir):
    """
    Reconstructs audio files from all .npy files in a given directory, looping through each
    :param npy_dir: Directory containing .npy files
    :param output_dir: Directory to output .wav files to
    """
    npy_dir = Path(npy_dir)
    output_dir = Path(output_dir)

    for npy_file in npy_dir.glob('*.npy'):  # grab any file in the given directory that is .xlsx
        output_wav_path = output_dir / f"{npy_file.stem}_reconstructed.npy"
        reconstruct_audio_from_stft_npy(npy_path=npy_file, output_path=output_wav_path, nperseg=1024, noverlap=512)
