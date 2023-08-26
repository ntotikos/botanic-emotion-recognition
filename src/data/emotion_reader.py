"""Label extractor for plant data. Labels are derived from facial emotions in provided .csv file."""
import os

import pandas as pd
import numpy as np
from typing import Union, List, Literal
from src.utils.constants import EKMAN_EMOTIONS_NEUTRAL, EMOTIONS_DIR


def read_emotions_csv(filepath: str) -> pd.DataFrame:
    """
    This method reads a .csv file containing individual emotions based on the specified path.

    :param filepath: Path to team member emotions .csv file.
    """

    if os.path.splitext(filepath)[1] == ".csv":
        df_emotions = pd.read_csv(filepath)
        return df_emotions

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
        print(f"Frame {frame_id}: Dominant emotion is {dominant_emotion}.")
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


# TODO: implement functionality to cover the case "fusion of 7 binary classifiers"
def placeholder_for_get_binary_fusion_classification_labels():
    """
    After the majority vote on the collective emotion, get a representation of this derived label depending on
    the chosen type of classification: "7-class" or "binary-fusion".
    """


if __name__ == "__main__":
    csv_file_path = EMOTIONS_DIR / "team_01/2023-01-10/clip_0_8583_9740.csv"
    df_rows = pd.read_csv(csv_file_path)[0:3]

    # TEST the different scenarios.
    get_dominant_emotion(df_rows, emotion_as="label")

    #get_dominant_emotion(df_rows, emotion_as="-hot")

    print(read_emotions_csv(csv_file_path)[0:10])

