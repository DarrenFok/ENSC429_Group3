from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from scipy.signal import istft
import matplotlib.pyplot as plt
import os
import pandas as pd
import xlsxwriter
import openpyxl

def grab_audio_file_data_mono(filename):
    audio, fs = librosa.load('../data/Samples/' + filename, sr=None)

    # remove stereo
    if audio.ndim > 1:
        audio = audio[:, 0]

    return audio, fs

def get_STFT(audio, fs):
    f, t, X = stft(audio, fs=fs, nperseg=1024)
    return X, f, t

def write_data_to_excel(Re_X, Im_X, fs):
    # Convert to DataFrame
    df_Re_X = pd.DataFrame(Re_X)
    df_Im_X = pd.DataFrame(Im_X)
    df_fs = pd.DataFrame([[fs]])

    with pd.ExcelWriter('stft-sample.xlsx', engine="xlsxwriter") as writer:
        df_Re_X.to_excel(writer, sheet_name='Re_X', index=False)
        df_Im_X.to_excel(writer, sheet_name='Im_X', index=False)
        df_fs.to_excel(writer, sheet_name='Sampling Rate', index=False)

#Call this from main
def convert_to_excel(filename, make_spectrogram):
    audio, fs = grab_audio_file_data_mono(filename)
    X, f, t = get_STFT(audio, fs)
    write_data_to_excel(np.real(X), np.imag(X), fs)

    if make_spectrogram:
        plt.pcolormesh(t, f, np.abs(X), shading='auto')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

# def main():
#     directory = os.fsencode('../data/Samples')
#
#     # TAKES A REALLY LONG TIME FOR LONGER AUDIO FILES
#     for file in os.listdir(directory):
#         filename = os.fsdecode(file)
#         audio, fs = grab_audio_file_data_mono(filename)
#         Re_X, Im_X = get_STFT(audio, fs)
#         write_data_to_excel(Re_X, Im_X, fs)
#
# main()