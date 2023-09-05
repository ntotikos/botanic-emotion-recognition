"""
Aggregate the data from various folders (all teams and days) in a way that it complies with the defined
classification problem (7-class, multiple binary classification, ...)

LIKELY TO BE DELETED! 05.09.2023
"""

import os
import glob
from abc import ABC, abstractmethod

from src.utils.constants import TEAM_NAMES_CLEANED, LABELS_DIR, CLEANED_DATA_DIR


class DataAggregator(ABC):
    def __init__(self):
        self.folder = None

    @abstractmethod
    def get_labels(self):
        pass


class SevenClassesAggregator:
    def __init__(self):
        super().__init__()

    def get_labels(self):
        pass


class MultiBinaryAggregator(DataAggregator):
    def __init__(self):
        super().__init__()

    def get_labels(self):
        pass


def load_dataset_from_folders(root_path):
    # TODO: Maybe I should use os.path.walk and traverse the tree?
    dataset = {}

    # Loop through each main class folder
    for class_folder in os.listdir(root_path):
        class_path = os.path.join(root_path, class_folder)

        # Ensure the path is a directory
        if os.path.isdir(class_path):
            dataset[class_folder] = []

            # Collect paths from each subfolder
            for subfolder in os.listdir(class_path):
                subfolder_path = os.path.join(class_path, subfolder)

                # Check if it's indeed a subfolder
                if os.path.isdir(subfolder_path):
                    image_files = glob.glob(os.path.join(subfolder_path, '*.*'))
                    dataset[class_folder].extend(image_files)

    return dataset


emotion_files_dict = load_dataset_from_folders(LABELS_DIR)
data_files_dict = load_dataset_from_folders(CLEANED_DATA_DIR)
print(emotion_files_dict)
print(data_files_dict["team_01"])

count = 0

for team in TEAM_NAMES_CLEANED:
    print(team)
    # TODO: fix problem; len(emotion_files_dict) should be 11 for team_01 but it is 10 I think.
    for i in range(len(emotion_files_dict[team])):
        print(i)
        print(emotion_files_dict[team])
        print(emotion_files_dict[team][i])

        emotion_file = emotion_files_dict[team][i]
        data_file = data_files_dict[team][i]

        interval_emotion_file = emotion_file.split(".")[0].split("_")[-4:]
        interval_data_file = data_file.split(".")[0].split("_")[-4:]

        if interval_emotion_file == interval_data_file:
            print("Emotion file and data file match.")
            print(interval_data_file)
        else:
            raise ValueError(f"Information about the teamwork session duration in {emotion_file} and {data_file} do"
                             f"not match. ")
            count += 1

        # print(interval_data_file)

print("Count:", count)
