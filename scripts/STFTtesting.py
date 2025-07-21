#!/usr/bin/env python3
# requirements:
#   librosa==0.9.2
#   numpy
#   matplotlib
#   pathlib
#   soundfile
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

#fs = sample rate
#audio = x[n]
#f = frequency
#t = time
#Zxx = complex valued Fourier Transform
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html

fs, audio = wavfile.read('../data/bird_chirp.wav')

#remove stereo
if audio.ndim > 1:
    audio = audio[:, 0]

# STFT
f, t, Zxx = stft(audio, fs = fs, nperseg = 1024)

print(Zxx)
#Plot |X(e^jw)|
plt.pcolormesh(t, f, np.abs(Zxx), shading='auto')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#32767 - is 2^16 values
_, reconstructed_audio = istft(Zxx, fs = fs, nperseg = 1024, noverlap = 512)
reconstructed_audio /= np.max(np.abs(reconstructed_audio))
reconstructed_audio_int16 =  np.int16(reconstructed_audio * 32767)

wavfile.write('output.wav', fs, reconstructed_audio_int16)



# def extract_metrics_from_audio():
#     """
#
#     :return:
#     """
#
# if __name__ == "__main__":
#     main()