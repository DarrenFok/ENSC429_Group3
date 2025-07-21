import numpy as np
import pandas as pd
import os
from scipy.signal import istft
from scipy.io import wavfile
from pathlib import Path

def reconstruct_audio_from_stft_xlsx(xlsx_path, output_path, nperseg=1024, noverlap=512):
    """
    Reconstructs audio file from a given .xlsx file containing a STFT
    :param xlsx_path: path to .xlsx file
    :param output_path: path to output audio file
    :param nperseg: desired sampling frequency
    :param noverlap: desired overlapping frequency
    """
    # load STFT data
    read_dataframe = pd.read_excel(xlsx_path, sheet_name="Re_X")
    imaginary_dataframe = pd.read_excel(xlsx_path, sheet_name="Im_X")
    meta_dataframe = pd.read_excel(xlsx_path, sheet_name="Sampling Rate")
    fs = int(meta_dataframe.iloc[0,0])
    print(fs)

    # conversion to numpy array
    real = read_dataframe.to_numpy()
    imag = imaginary_dataframe.to_numpy()

    # combine back to complex STFT matrix
    Zxx = real + (1j * imag)

    # inverse STFT transformation
    _, reconstructed_audio = istft(Zxx, fs=fs , nperseg=nperseg, noverlap=noverlap)

    # normalizing to prevent clipping, before it sounded distorted on drums
    max_val  = np.max(np.abs(reconstructed_audio))
    if max_val > 0:
        reconstructed_audio = reconstructed_audio/max_val

    reconstructed_audio_int16 = np.int16(reconstructed_audio * 32767)  # 32676 is max positive value of 16-bit int

    # save output to WAV
    wavfile.write(output_path, fs, reconstructed_audio_int16)

def reconstruct_audio_from_directory(xlsx_dir, output_dir):
    """
    Reconstructs audio files from all .xlsx file in a given directory, looping through each\
    :param xlsx_dir: Directory containing .xlsx files
    :param output_dir: Directory to output .wav files to
    """
    xlsx_dir = Path(xlsx_dir)
    output_dir = Path(output_dir)

    for xlsx_file in xlsx_dir.glob('*.xlsx'):  # grab any file in the given directory that is .xlsx
        output_wav_path = output_dir / f"{xlsx_file.stem}_reconstructed.wav"
        reconstruct_audio_from_stft_xlsx(xlsx_path=xlsx_file, output_path=output_wav_path, nperseg=1024, noverlap=512)

if __name__ == '__main__':
    directory = (Path(os.getcwd())).parent
    reconstruct_audio_from_directory(
        xlsx_dir=directory/"data"/"Samples",
        output_dir=directory/"data"/"Reconstructed"
    )
