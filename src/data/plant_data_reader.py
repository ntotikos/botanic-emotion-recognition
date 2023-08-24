"""Data reader for plant data stored as .wav files."""
import os
from typing import List

import pandas as pd
from src.utils.constants import EMOTIONS_DIR, INTERIM_PLANT_DATA_DIR


def read_plant_file(filepath: str) -> None:
    """
    This method reads a .wav file based on the specified path.

    :param filepath: Path to plant .wav file data.
    """

    if os.path.splitext(filepath)[1] == ".wav":
        # TODO: READ scipy.io .wav file.
        # returns list of double or array
        pass
    else:
        raise NameError(f"File should be a \".wav\" file. Got \"{os.path.splitext(filepath)[1]}\".")


def slice_plant_signal(plant_signal: List[float]) -> List[List[float]]:
    """
    This method splits a given plant signal into 1s slices.
    :param plant_signal:
    """
    # TODO: implement plant signal slicing.
    print(plant_signal)
    return [[0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1]]


if __name__ == "__main__":
    csv_file_path = EMOTIONS_DIR / "team_01/2023-01-10/sdm_2023-01-10_team_01_8333_9490.csv"
    wav_file_path = INTERIM_PLANT_DATA_DIR / "team_01/2023-01-10/clip_0_8583_9740.wav"

    read_plant_file(wav_file_path)
    read_plant_file(csv_file_path)
