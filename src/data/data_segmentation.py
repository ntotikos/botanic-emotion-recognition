"""Slice .wav plant data into 1s chunks and assign cleaned labels. Store as pickle."""
import glob
import os
from typing import List, Dict

import pandas as pd
from scipy.io import wavfile
import numpy as np
import pickle
import logging

from src.utils.constants import CLEANED_DATA_DIR, LABELS_DIR, DATASETS_DIR, TEAM_NAMES_CLEANED, LOGS_DIR

logging.basicConfig(filename=LOGS_DIR / 'data-cleaning/data_segmentation_all_mismatches_DEBUG_iter2.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


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
        #print("len(signal_slice)", len(signal_slice))

        if len(signal_slice) != 10000:
            print("------------------------------------------- Trimmed TS.")
            continue

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
    for segmend_id in range(len(slices)):
        segment = {
            "wav_slice": slices[segmend_id],
            "label": dataframe_emotions["Labels"].iloc[segmend_id],
            "segment_id": segmend_id,
            "frame_id": dataframe_emotions["Frame"].iloc[segmend_id]
        }
        segments.append(segment)
    return segments


def save_datapoints(datapoints):
    # TODO: it might be worthwhile to store the files into database in the future in case of frequent query.

    # save_file_path = DATASETS_DIR / 'sdm_2023-01-10_team_01_8333_9490.pkl'
    save_file_path = DATASETS_DIR / 'sdm_2023-01_all_valid_files_version_iter2.pkl'

    if not os.path.isfile(save_file_path):
        with open(save_file_path, 'wb') as file:
            pickle.dump(datapoints, file)
    else:
        print(f"File already exists: {save_file_path}.")


def get_all_filenames_in_dir(root_path) -> Dict:
    # TODO: Maybe I should use os.path.walk and traverse the tree?
    all_filenames_dir = {}

    # Loop through each main class folder
    for class_folder in os.listdir(root_path):
        class_path = os.path.join(root_path, class_folder)

        # Ensure the path is a directory
        if os.path.isdir(class_path):
            all_filenames_dir[class_folder] = []

            # Collect paths from each subfolder
            for subfolder in os.listdir(class_path):
                subfolder_path = os.path.join(class_path, subfolder)

                # Check if it's indeed a subfolder
                if os.path.isdir(subfolder_path):
                    image_files = glob.glob(os.path.join(subfolder_path, '*.*'))
                    all_filenames_dir[class_folder].extend(image_files)

    return all_filenames_dir


def check_durations(data_filename: str, emotion_filename: str) -> bool:
    """
    Checks the durations derived from the start and end times incorporated in the filenames.
    :param data_filename: file name of the signal file.
    :param emotion_filename: file name of the cleaned emotion file.
    """
    same_durations = True
    interval_emotion_file = emotion_filename.split(".")[0].split("_")[-4:]
    interval_data_file = data_filename.split(".")[0].split("_")[-4:]

    if interval_emotion_file != interval_data_file:
        same_durations = False
        print("Labels don't match.")
    #else:
        #print("Labels are the same.")

    return same_durations


def flatten_list_of_lists(list_of_lists):
    print(f"shape list_of_lists: {len(list_of_lists)}, {len(list_of_lists[0])}, {len(list_of_lists[1])}")
    flattened_list = []
    for sublist in list_of_lists:
        flattened_list.extend(sublist)
    print(f"shape flattened_list: {len(flattened_list)}")

    return  flattened_list


def _main_one_file():
    wav_file_path_test = CLEANED_DATA_DIR / r"team_01\2023-01-10\sdm_2023-01-10_team_01_8333_9490.wav"
    emotions_file_path_test = LABELS_DIR / r"team_01\2023-01-10\emotions_sdm_2023-01-10_team_01_8333_9490.csv"

    samplingrate, signal = read_plant_file(wav_file_path_test)
    signal_slices = slice_plant_signal(signal, samplingrate)
    emotions = read_cleaned_emotions(emotions_file_path_test)

    # Data points for one file! TODO: use for loop to iterate over all files!
    data_points = map_slice_to_emotion(signal_slices, emotions)
    save_datapoints(data_points)


def _main():
    #wav_file_path_test = CLEANED_DATA_DIR
    #emotions_file_path_test = LABELS_DIR

    datafiles_dir = get_all_filenames_in_dir(CLEANED_DATA_DIR)
    labelfiles_dir = get_all_filenames_in_dir(LABELS_DIR)
    total_samples = 0
    count = 0
    data_points_all = []
    for team in TEAM_NAMES_CLEANED:
        for idx in range(len(labelfiles_dir[team])):
            print(datafiles_dir[team][idx])
            samplingrate, signal = read_plant_file(datafiles_dir[team][idx])
            signal_slices = slice_plant_signal(signal, samplingrate)
            emotions = read_cleaned_emotions(labelfiles_dir[team][idx])

            #print(check_durations(datafiles_dir[team][idx], labelfiles_dir[team][idx]))

            interval_emotion_file = labelfiles_dir[team][idx].split(".")[0].split("_")[-4:]
            #print("duration from label: ", int(interval_emotion_file[-1])-int(interval_emotion_file[-2]))

            total_samples =  total_samples + len(signal_slices)

            logging.info(f"len(emotions): {len(emotions)}")
            logging.info(f"len(signal_slices): {len(signal_slices)}")
            if int(interval_emotion_file[-1])-int(interval_emotion_file[-2]) == len(emotions):
                #    print(check_durations(datafiles_dir[team][idx], labelfiles_dir[team][idx]))
                #if len(signal_slices) != len(emotions):
                logging.info("-----")
                logging.info(f"team, idx: {team}, {idx}")
                logging.info(f"data file: {datafiles_dir[team][idx]}")
                logging.info(f"label file: {labelfiles_dir[team][idx]}")
                count += 1
                logging.info(f"length slices, emotions: {len(signal_slices)}, {len(emotions)}")

                data_points = map_slice_to_emotion(signal_slices, emotions)
                data_points_all.append(data_points)
            #else:
            #    count += 1
            #    print(f"Information about the teamwork session duration do not match. ")
    # SAVE DATA
    logging.info(f"count:{count}")
    logging.info(f"Number of total_samples: {total_samples}")

    data_points_all = flatten_list_of_lists(data_points_all)
    print(len(data_points_all))
    print(total_samples)



    # Data points for one file! TODO: use for loop to iterate over all files!
    #save_datapoints(data_points_all)


if __name__ == "__main__":
    _main()

