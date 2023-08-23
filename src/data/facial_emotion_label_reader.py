"""Label extractor for plant data. Labels are derived from facial emotions in provided .csv file."""
import pandas as pd
from typing import Union, List, Literal


def get_dominant_emotion(
        team_emotions_snapshot: pd.DataFrame,
        emotion_as: Literal["binary-fusion", "one-hot", "label"] = "label"
) -> Union[List[str], List[int], str]:
    """
    Get the dominant emotion based on the majority of individual emotions.

    :param team_emotions_snapshot: multiple row that exhibits various individual emotions at specific time instance.
    :param emotion_as: specification of the output format of the dominant emotion.
    """

    for row in team_emotions_snapshot:
        pass

    print(team_emotions_snapshot)
    if emotion_as == "label":
        pass
    elif emotion_as == "one-hot":
        pass
    elif emotion_as == "binary-fusion":
        pass
    else:
        raise ValueError(f"Parameter emotion_as has to be label, one-hot or binary-fusion. Got {emotion_as}.")

    return "Hi"


def a():
    """
    After the majority vote on the collective emotion, get a representation of this derived label depending on
    the chosen type of classification: "7-class" or "binary-fusion".
    """
