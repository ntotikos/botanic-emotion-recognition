"""
Mapping emotions extracted from a given .csv file to the corresponding time series signal from the .wav plant recording.
"""
import pandas as pd
from typing import Union, Tuple, Literal
import os
from src.utils.constants import EMOTIONS_DIR, INTERIM_PLANT_DATA_DIR, TEAM_NAMES, TEAMWORK_SESSION_DAYS


def custom_sort(label: str, mode: Literal["interim", "emotions"] = "interim") -> Union[int, Tuple[int, int]]:
    """
    Helper function for customized sorting of file labels in emotions folder or interim data folder.
    emotions: Sort labels like "clip_0_11509_11908.csv" first by clip id (0) and second by start frame (11509).
    interim: Sort labels like "sdm_2023-01-10_team_01_8333_9490.wav" first by clip id (0) and second by start
    frame (11509).
    """

    if mode == "interim":
        parts = label.split('_')
        return int(parts[4])
    elif mode == "emotions":
        parts = label.split('_')
        return int(parts[1]), int(parts[2])


def get_duration_from_label(label: str, mode: Literal["interim", "emotions"] = "interim") -> int:
    """
    Compute the duration of the teamwork session based on the start and end frame in the corresponding label.
    emotions: "clip_0_11509_11908.csv", i.e. 11908-11509.
    interim: "sdm_2023-01-10_team_01_8333_9490.wav", i.e. 8333-9490.
    """

    if mode == "interim":
        parts = label.split('.')[0].split("_")
        duration = int(parts[5]) - int(parts[4])
        return duration
    elif mode == "emotions":
        parts = label.split('.')[0].split("_")
        duration = int(parts[3]) - int(parts[2])
        return duration


def map_emotions_to_signal(interim_data_dir: str, emotions_dir: str) -> pd.DataFrame:
    emotion_signal_mappings = pd.DataFrame(columns=['path_emotions', 'duration_emotions', "path_interim",
                                                    "duration_interim", 'difference'])
    """
    Map .csv files containing team emotions to teamwork .wav files based on teamwork session duration 
    indicated in file name. 
    """

    for t in TEAM_NAMES:
        for d in TEAMWORK_SESSION_DAYS:
            interim_data_path = os.path.join(interim_data_dir, t, d)
            emotions_path = os.path.join(emotions_dir, t, d)

            if os.path.exists(interim_data_path) and os.path.exists(emotions_path):
                # 1. Files with emotions per second
                clip_files = os.listdir(emotions_path)
                clip_files = [item for item in clip_files if not item.startswith('team')]  # remove item "team_1...csv"

                # lambda function needed because otherwise I could not use self-implemented custom_sort
                # because it takes more than one argument.
                clip_files = sorted(clip_files, key=lambda x: custom_sort(x, mode="emotions"))

                # 2. Files with interim plant teamwork signal data
                interim_data_files = os.listdir(interim_data_path)
                interim_data_files = sorted(interim_data_files, key=custom_sort)

                for i in range(len(clip_files)):
                    emotion_signal_mappings.loc[len(emotion_signal_mappings)] = \
                        [os.path.join(t, d, clip_files[i]), get_duration_from_label(clip_files[i], mode="emotions"),
                         interim_data_files[i], get_duration_from_label(interim_data_files[i]),
                         get_duration_from_label(interim_data_files[i])-get_duration_from_label(clip_files[i],
                                                                                                mode="emotions")]
    return emotion_signal_mappings


if __name__ == "__main__":
    # Quick sanity check
    mismatches = map_emotions_to_signal(INTERIM_PLANT_DATA_DIR, EMOTIONS_DIR)
    df_bool = (mismatches["difference"] != 0)
    if df_bool.sum() == 12:
        print(f"The number of mismatches {df_bool.sum()} correct.")

    # mismatch_logs.to_excel(os.path.join(LOGS_DIR,"duration_comparison_teamwork_session.xlsx"))
