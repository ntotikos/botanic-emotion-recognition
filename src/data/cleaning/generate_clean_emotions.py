"""Label extractor for plant data. Labels are derived from facial emotions in provided .csv file."""
import os

import pandas as pd
import numpy as np
from typing import Union, List, Literal

from src.data.teamwork_durations import custom_sort, compare_indicated_tw_duration
from src.utils.constants import (EKMAN_EMOTIONS_NEUTRAL, EMOTIONS_DIR, TEAM_NAMES, TEAMWORK_SESSION_DAYS,
                                 INTERIM_PLANT_DATA_DIR, LABELS_DIR, LOGS_DIR)

import logging
from scipy.io import wavfile

logging.basicConfig(filename=LOGS_DIR / 'data-cleaning/generate_clean_emotions_all_iter2.log',
                    filemode='w',  # when creating new file change this to "a" for append
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def read_emotions_csv(filepath: str) -> pd.DataFrame:
    """
    This method reads a .csv file containing individual emotions based on the specified path.

    :param filepath: Path to team member emotions .csv file.
    """

    if os.path.splitext(filepath)[1] == ".csv":
        df = pd.read_csv(filepath)
        return df

    else:
        raise NameError(f"File should be a \".csv\" file. Got \"{os.path.splitext(filepath)[1]}\".")


def get_dominant_emotion(
        df_team_emotions_snapshot: pd.DataFrame,
        emotion_as: Literal["binary-fusion", "one-hot", "label"] = "label"
) -> Union[List[str], List[int], str]:
    """
    Get the dominant emotion based on the majority of individual emotions at exactly one particular time frame.

    :param df_team_emotions_snapshot: multiple (max. 4) rows exhibiting individual emotions at specific time instance.
    :param emotion_as: specification of the output format of the dominant emotion.
    """

    # TODO: logging
    for row in df_team_emotions_snapshot:
        pass

    frame_id = df_team_emotions_snapshot["Frame"].iloc[0]  # all rows should have same frame_id
    df_summed_rows = pd.DataFrame([df_team_emotions_snapshot.loc[:, EKMAN_EMOTIONS_NEUTRAL].sum()])

    if emotion_as == "label":
        dominant_emotion = df_summed_rows.idxmax(axis=1).iloc[0]
        # print(f"Frame {frame_id}: Dominant emotion is {dominant_emotion}.")
        return dominant_emotion
    # TODO: "one-hot" and "binary-fusion" are probably not needed HERE. Delete.
    elif emotion_as == "one-hot":
        # for logging purposes get emotion as label.
        dominant_emotion = df_summed_rows.idxmax(axis=1).iloc[0]
        print(f"Frame {frame_id}: Dominant emotion is {dominant_emotion}.")

        # Get the actual one-hot vector corresponding to particular emotion.
        dominant_emotion_onehot = np.zeros(df_summed_rows.shape, dtype=int)
        dominant_emotion_onehot[:, df_summed_rows.values.argmax(1)] = 1
        print(f"Frame {frame_id}: Dominant emotion is {dominant_emotion_onehot}.")
        return dominant_emotion_onehot
    elif emotion_as == "binary-fusion":
        # TODO: implement
        pass
    else:
        raise ValueError(f"Parameter emotion_as has to be \"label\", \"one-hot\" or \"binary-fusion\". "
                         f"Got \"{emotion_as}\".")


def extract_labels(clip_file):
    """
    Extract emotion labels from .csv file for one teamwork session of a team on a particular day.
    """
    frames = []
    labels = []
    df_snapshots = read_emotions_csv(clip_file).groupby("Frame")  # create list snapshots

    for frame, snapshot in df_snapshots:
        label = get_dominant_emotion(snapshot, emotion_as="label")
        frames.append(frame)
        labels.append(label)

    df_labels = pd.DataFrame({"Frame": frames, "Labels": labels})

    return df_labels


def _test():
    """
    Small test.
    """
    csv_file_path = EMOTIONS_DIR / "team_01/2023-01-10/clip_0_8583_9740.csv"
    df_rows = pd.read_csv(csv_file_path)[0:3]

    # TEST the different scenarios.
    get_dominant_emotion(df_rows, emotion_as="label")

    # get_dominant_emotion(df_rows, emotion_as="-hot")

    print(read_emotions_csv(csv_file_path)[0:10])


def _main(save_file=False):
    """
    Create .csv files for all valid teams and days and store emotion-frame pairs.
    """
    deviating_clip_length_and_label = pd.DataFrame(columns=['path_clip', 'length_clip', "length_clip_div_5",
                                                            "path_interim", "length_interim", 'durations_match',
                                                            "difference_lengths"])

    # iterate over all files and extract emotions.
    for t in TEAM_NAMES:
        for d in TEAMWORK_SESSION_DAYS:
            interim_data_path = os.path.join(INTERIM_PLANT_DATA_DIR, t, d)
            emotions_path = os.path.join(EMOTIONS_DIR, t, d)

            if os.path.exists(interim_data_path) and os.path.exists(emotions_path):
                # 1. Files with emotions per second
                clip_files = os.listdir(emotions_path)
                clip_files = [item for item in clip_files if not item.startswith('team')]  # remove item "team_1...csv"

                # lambda function needed because otherwise I could not use self-implemented custom_sort
                # because it takes more than one argument.
                clip_files = sorted(clip_files, key=lambda x: custom_sort(x, mode="emotions"))
                logging.info(f"Sorted clip_files of {t} on day {d}: {clip_files}")
                print(f"Sorted clip_files of {t} on day {d}: {clip_files}")

                # 2. Files with interim plant teamwork signal data
                interim_data_files = os.listdir(interim_data_path)
                interim_data_files = sorted(interim_data_files, key=custom_sort)
                logging.info(f"Sorted interim_data_files of {t} on day {d}: {interim_data_files}")
                print(f"Sorted interim_data_files of {t} on day {d}: {interim_data_files}")

                for i in range(len(clip_files)):
                    equal_durations = compare_indicated_tw_duration(interim_data_files[i], clip_files[i])
                    logging.info(f"Equal teamwork duration for clip_file and interim_data_file: {equal_durations}")
                    print(f"Equal teamwork duration for clip_file and interim_data_file: {equal_durations}")

                    csv_path = os.path.join(emotions_path, clip_files[i])
                    df_emotions = extract_labels(csv_path)
                    logging.info(f"Length of df_emotions: {len(df_emotions)/5.0}")
                    print(f"Length of df_emotions: {len(df_emotions)/5.0}")

                    plant_signal_path = os.path.join(interim_data_path, interim_data_files[i])
                    _, plant_signal = wavfile.read(plant_signal_path)

                    logging.info(f"Length of interim_data_files[i] 1s slices: {len(plant_signal)/10000}")
                    print(f"Length of interim_data_files[i] 1s slices: {len(plant_signal)/10000}")

                    deviating_clip_length_and_label.loc[len(deviating_clip_length_and_label)] = \
                        [os.path.join(t, d, clip_files[i]), len(df_emotions), len(df_emotions)/5,
                         interim_data_files[i], len(plant_signal)/10000, equal_durations,
                         (len(plant_signal) / 10000) - (len(df_emotions)/5)
                         ]

                    if save_file:
                        label_path = os.path.join(LABELS_DIR, t, d)

                        if not os.path.exists(label_path):
                            os.makedirs(label_path)

                        fragment = interim_data_files[i].split(".")[0]
                        file_path = os.path.join(label_path, f"emotions_{fragment}.csv")

                        logging.info(f"Original emotion clip label: {clip_files[i]}")
                        logging.info(f"Cleaned emotions path: {file_path}")
                        logging.info(f"Cleaned signal path: {interim_data_files[i]}\n-----")

                        if not os.path.exists(file_path):
                            df_emotions.to_csv(file_path, index=False)

                print("1 Day done!")
    if not os.path.isfile(os.path.join(LOGS_DIR / "data-cleaning",
                                       "length_comparison_clip_and_its_label_iter2.xlsx")):
        deviating_clip_length_and_label.to_excel(os.path.join(LOGS_DIR / "data-cleaning",
                                                              "length_comparison_clip_and_its_label_iter2.xlsx"))


if __name__ == "__main__":
    _test()
    #_main(save_file=False)
    # _main(save_file=True)
