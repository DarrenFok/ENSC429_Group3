import torch
import torch.nn as nn
from audio_to_npy_spectrogram import wav_to_npy
from npy_spectrogram_to_audio import reconstruct_audio_from_stft_npy
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import librosa
import soundfile as sf
from scipy.signal import stft
from scipy.io.wavfile import write as write_wav


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 filter for feature detection
                nn.BatchNorm2d(out_channels),  # stabilize and speeds up training
                nn.ReLU(inplace=True),  # apply non-linearity for complex pattern learning
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # Encoding - reduce spatial dimensions and abstract features so model can understand
        self.encoder1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # reduce resolution by 2 to allow for larger context
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = conv_block(512, 1024)  # decision hub - learns what high level features are

        # Decoding - up sample and reconstruct the isolated vocals
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )  # reduce back to 1 channel (vocal spectrogram)

    def forward(self, x):
        # encoding
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        center = self.center(self.pool4(e4))

        # decoding
        d4 = self.dec4(torch.cat([self.up4(center), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.final(d1)

        return out

def pad_to_multiple(tensor, multiple=16):
    """Pad tensor H and W to be multiple of `multiple` (default 16)."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = F.pad(tensor, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
    return padded, pad_h, pad_w

def run_pipeline(input_wav, output_wav, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wav_path = Path(input_wav)
    cwd = Path.cwd().parent
    temp_npy_path = Path(cwd/"data/temp")
    pred_npy_path = Path(cwd/"data/output")

    # 1. convert .wav to log mag .npy
    output_npy_path = wav_to_npy(input_wav, output_dir=temp_npy_path)

    # 2. load model and run inference
    input_npy = np.load(output_npy_path)
    log_min, log_max = input_npy.min(), input_npy.max()
    input_tensor = torch.tensor(input_npy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    input_tensor, pad_h, pad_w = pad_to_multiple(input_tensor)

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
        if pad_h > 0:
            output_tensor = output_tensor[:, :, :-pad_h, :]
        if pad_w > 0:
            output_tensor = output_tensor[:, :, :, :-pad_w]

    output_npy = output_tensor.squeeze().cpu().numpy()
    output_npy_rescaled = output_npy * (log_max - log_min) + log_min
    np.save(pred_npy_path, output_npy_rescaled)

    # reconstruct
    reconstruct_audio_from_stft_npy(output_npy_path, output_wav)

# === Example usage ===its a
# python predict_and_convert.py
if __name__ == "__main__":
    model_path = "C:/Users/Darren/Documents/Projects/ENSC429_Group3/vocal_isolator.pth"
    input_wav = "C:/Users/Darren/Documents/Projects/ENSC429_Group3/data/Samples/valmix/mix_Steven_Clark_-_Bounty.wav"
    output_wav = "C:/Users/Darren/Documents/Projects/ENSC429_Group3/mix_Steven_Clark_-_Bounty_reconstructed.wav"

    run_pipeline(input_wav, output_wav, model_path)

