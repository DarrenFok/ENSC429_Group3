# U-Net Based Vocal Isolator
## How to use:

1. Download model file (vocal_isolator.pth) from download link:
https://1sfu-my.sharepoint.com/:u:/g/personal/wms8_sfu_ca/EXmqgAdnHgdGja87iX4dtZEBaI4rj8CdmH2N0oi_cex6ow?e=Wuhbje

2. Place vocal_isolator.pth in project directory

3. From `<project directory>\scripts`, run the following command:
`python isolate_vocals.py <input .wav file path> <optional output.wav file path>`
    - if no output path is specified, output.wav will be output in directory where `isolate_vocals.py` is located
