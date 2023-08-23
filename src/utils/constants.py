"""
Constants that are needed multiple times.
"""

from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent  # repository root: botanic-emotion-recognition/

EMOTIONS_DIR = PROJECT_ROOT / "data/teamwork-emotions"
INTERIM_PLANT_DATA_DIR = PROJECT_ROOT / "data/interim-plant-data-teamwork-extracted"

LOGS_DIR = PROJECT_ROOT / "logs"


# Project Specifics
TEAM_NAMES = ["team_01", "team_02", "team_03", "team_04", "team_05", "team_06", "team_07", "team_08", "team_09",
              "team_10", "team_11", "team_12", "team_13", "team_15", "team_16", "team_17", "team_18", "team_19",
              "team_20", "team_22"]

TEAMWORK_SESSION_DAYS = ["2023-01-10", "2023-01-12", "2023-01-13"]
