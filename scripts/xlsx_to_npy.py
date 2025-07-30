import os
import numpy as np
import pandas as pd
from pathlib import Path

def convert_xlsx_to_npy(input_dir, output_dir):
    """
    Converts all .xlsx spectrogram files (with Re_X and Im_X sheets)
    from input_dir into .npy files in output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.xlsx"):
        try:
            # Load Excel sheets
            real = pd.read_excel(file, sheet_name="Re_X", header=None).values.astype(np.float32)
            imag = pd.read_excel(file, sheet_name="Im_X", header=None).values.astype(np.float32)
            stft = real + 1j * imag

            # Save as .npy
            np.save(output_dir / (file.stem + ".npy"), stft)
            print(f"Converted {file.name} -> {file.stem}.npy")
        except Exception as e:
            print(f"Error converting {file.name}: {e}")

# Example usage:
# convert_xlsx_to_npy("data/mixture/train", "data/mixture_npy/train")
# convert_xlsx_to_npy("data/vocal/train", "data/vocal_npy/train")
directory = (Path(os.getcwd())).parent
convert_xlsx_to_npy(directory/"data"/"mixture"/"train", directory/"data"/"mixture_npy"/"train")
convert_xlsx_to_npy(directory/"data"/"mixture"/"val", directory/"data"/"mixture_npy"/"val")
convert_xlsx_to_npy(directory/"data"/"vocal"/"train", directory/"data"/"vocal_npy"/"train")
convert_xlsx_to_npy(directory/"data"/"vocal"/"val", directory/"data"/"vocal_npy"/"val")
