import os

import librosa
import numpy as np
import getSTFT

def convert_all_files():

    directory = os.fsencode('../data/Samples/test/mix')

    # TAKES A REALLY LONG TIME FOR LONGER AUDIO FILES
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        getSTFT.convert_to_excel(filename, 0)

def convert_one(track_title):

    directory = os.fsencode('../data/Samples')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if track_title == filename:
            getSTFT.convert_to_excel(filename, 1)
            return

def main():
    print("testmain")
    # convert_one("whatsmyageagainInstrumental.mp3")
    convert_all_files()

main()