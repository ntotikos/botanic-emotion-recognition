"""Slice .wav plant data into 1s chunks and assign cleaned labels. Store as pickle."""
import os
from typing import List

import pandas as pd
from scipy.io import wavfile
import numpy as np
import pickle

from src.utils.constants import CLEANED_DATA_DIR, LABELS_DIR, DATASETS_DIR


def read_plant_file(filepath: str):
    """
    This method reads a .wav file based on the specified path.

    :param filepath: Path to plant .wav file data.
    """
    if os.path.splitext(filepath)[1] == ".wav":
        sampling_rate, plant_wave = wavfile.read(filepath)
        assert sampling_rate == 10000
        return sampling_rate, plant_wave
    else:
        raise NameError(f"File should be a \".wav\" file. Got \"{os.path.splitext(filepath)[1]}\".")


def slice_plant_signal(plant_signal: List[float], sampling_rate: int = 10000) -> List[List[float]]:
    """
    This method splits a given plant signal into 1s slices.
    :param plant_signal:
    :param sampling_rate:
    """
    slices = []
    for i in range(0, len(plant_signal), sampling_rate):
        signal_slice = plant_signal[i:i+sampling_rate]
        slices.append(signal_slice)

    slices = np.array(slices)
    return slices


def read_cleaned_emotions(emotions_path: str):

    # TODO: For now, take only first out of 5 frames. Reason: I think raw emotions had a wrong frame index (02.09.23).
    # It says 0,0,0,0,5,5,5,5,....,25,25,25,25,... but it should probably be 0,0,0,0,1,1,1,1,...,5,5,5,5,....
    # Why? because the length of the emotions file is 5785 and it should be 1157 for team_01 on day 1. 5785/1157 is
    # 5. Coincidence? I doubt.
    # TODO: Update emotions pipeline once hypothesis confirmed.
    df_emotions = pd.read_csv(emotions_path)

    df_emotions_light = df_emotions[::5]  # take every fifth element of the emotions dataframe. TO BE UPDATED.
    return df_emotions_light


def map_slice_to_emotion(slices, dataframe_emotions):
    """
    One segment consists of a 1s signal portion and the corresponding label (including IDs).
    """
    segments = []
    for i in range(len(slices)):
        segment = {
            "wav_slice": slices[i],
            "label": dataframe_emotions["Labels"].iloc[i],
            "segment_id": i,
            "frame_id": dataframe_emotions["Frame"].iloc[i]
        }
        segments.append(segment)
    return segments


def save_datapoints(datapoints):
    # TODO: it might be worthwhile to store the files into database in the future in case of frequent query.

    save_file_path = DATASETS_DIR / 'sdm_2023-01-10_team_01_8333_9490.pkl'
    if not os.path.isfile(save_file_path):
        with open(save_file_path, 'wb') as file:
            pickle.dump(datapoints, file)


if __name__ == "__main__":
    wav_file_path_test = CLEANED_DATA_DIR / r"team_01\2023-01-10\sdm_2023-01-10_team_01_8333_9490.wav"
    emotions_file_path_test = LABELS_DIR / r"team_01\2023-01-10\emotions_sdm_2023-01-10_team_01_8333_9490.csv"

    samplingrate, signal = read_plant_file(wav_file_path_test)
    signal_slices = slice_plant_signal(signal, samplingrate)
    emotions = read_cleaned_emotions(emotions_file_path_test)

    # Data points for one file! TODO: use for loop to iterate over all files!
    data_points = map_slice_to_emotion(signal_slices, emotions)
    save_datapoints(data_points)
