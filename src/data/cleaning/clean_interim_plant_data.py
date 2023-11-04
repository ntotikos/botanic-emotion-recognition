"""
Create directories with ´cleaned´ files that will be used for training, test and validation.
1. Take `clean and extracted` labels from `labels-extracted´ folder.
2. Take only .wav files from `interim-plant-data-extracted` corresponding to the .csv file labels from 1.
3. Copy the files from 2. into `cleaned-plant-data`.
"""

from src.utils.constants import INTERIM_PLANT_DATA_DIR, TEAM_NAMES_CLEANED, LABELS_DIR, CLEANED_DATA_DIR, LOGS_DIR
import os
import logging
import shutil

logging.basicConfig(filename=LOGS_DIR / 'data-cleaning/copy_valid_wav_files_iter2.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

num_total_files = 0
num_unbroken_files = 0
num_copied_files = 0

# Count number of unbroken files.
for dir_path, dir_names, file_names in os.walk(INTERIM_PLANT_DATA_DIR):
    num_total_files += len(file_names)
    if not dir_names and len(os.path.dirname(dir_path).split("\\")[-1]) < 8:
        num_unbroken_files += len(file_names)

# Count number of copied files (i.e. the ´clean´ files)
for dirpath, dirnames, filenames in os.walk(LABELS_DIR):
    # Create target folder structure.
    target_path = dirpath.replace("labels-extracted", "cleaned-plant-data")
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Copy valid files.
    if filenames:
        for file in filenames:
            emotion_file_path = os.path.join(dirpath, file)

            tmp_file_path = emotion_file_path.replace("labels-extracted", "interim-plant-data-teamwork-extracted")
            wav_file_name = tmp_file_path.split("emotions_")[-1].split(".")[0] + ".wav"
            wav_file_path = os.path.join(os.path.dirname(tmp_file_path), wav_file_name)

            destination_path = wav_file_path.replace("interim-plant-data-teamwork-extracted", "cleaned-plant-data")

            num_copied_files += 1

            if os.path.isfile(wav_file_path):
                shutil.copy(wav_file_path, destination_path)
                logging.info(f"\nCopied: {wav_file_path} \n------> {destination_path}")

logging.info(f"num_total_files: {num_total_files}")
logging.info(f"num_unbroken_files: {num_unbroken_files}")
logging.info(f"num_cleaned_files: {num_copied_files}")
