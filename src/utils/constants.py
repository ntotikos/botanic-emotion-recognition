"""
Constants that are needed multiple times.
"""

from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent  # repository root: botanic-emotion-recognition/

EMOTIONS_DIR = PROJECT_ROOT / "data/teamwork-emotions"
LABELS_DIR = PROJECT_ROOT / "data/labels-extracted"
INTERIM_PLANT_DATA_DIR = PROJECT_ROOT / "data/interim-plant-data-teamwork-extracted"
CLEANED_DATA_DIR = PROJECT_ROOT / "data/cleaned-plant-data"
DATASETS_DIR = PROJECT_ROOT / "data/datasets"  # TODO: update once target datasets are defined.
MFCC_IMAGES_DIR = PROJECT_ROOT / "data/mfcc-images"

LOGS_DIR = PROJECT_ROOT / "logs"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"


# Project Specifics
TEAM_NAMES = ["team_01", "team_02", "team_03", "team_04", "team_05", "team_06", "team_07", "team_08", "team_09",
              "team_10", "team_11", "team_12", "team_13", "team_15", "team_16", "team_17", "team_18", "team_19",
              "team_20", "team_22"]

TEAM_NAMES_CLEANED = TEAM_NAMES
#TEAM_NAMES_CLEANED = ["team_01", "team_02", "team_04", "team_07", "team_10", "team_12",
#                      "team_13", "team_15", "team_19", "team_20"]                           data cleaning iter1

TEAMWORK_SESSION_DAYS = ["2023-01-10", "2023-01-12", "2023-01-13"]


# Assumptions
EKMAN_EMOTIONS = ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear"]
EKMAN_EMOTIONS_NEUTRAL = EKMAN_EMOTIONS + ["Neutral"]

EKMAN_TO_INT_DICT = {EKMAN_EMOTIONS[i]: i for i in range(len(EKMAN_EMOTIONS))}
EKMAN_NEUTRAL_TO_INT_DICT = {EKMAN_EMOTIONS_NEUTRAL[i]: i for i in range(len(EKMAN_EMOTIONS_NEUTRAL))}

INT_TO_EKMAN_DICT = {i: EKMAN_EMOTIONS[i] for i in range(len(EKMAN_EMOTIONS))}
INT_TO_EKMAN_NEUTRAL_DICT = {i: EKMAN_EMOTIONS_NEUTRAL[i] for i in range(len(EKMAN_EMOTIONS_NEUTRAL))}


# Signal
SAMPLING_RATE = 10000  # Samples per second


def map_int_to_label(emotion: int):
    return INT_TO_EKMAN_NEUTRAL_DICT[int(emotion)]