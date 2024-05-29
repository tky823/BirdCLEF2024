import ast
import csv
import os
from typing import Any, Dict, List, Optional, Union

from audyn.utils.data.birdclef.birdclef2024 import primary_labels as birdclef2024_primary_labels

from .collator import (
    BirdCLEF2024AudioChunkingCollator,
    BirdCLEF2024BaselineCollator,
    BirdCLEF2024ChunkingCollator,
)
from .composer import (
    BirdCLEF2024AudioChunkingComposer,
    BirdCLEF2024AudioComposer,
    BirdCLEF2024PrimaryLabelComposer,
    BirdCLEF2024PrimaryLabelDistillationComposer,
    BirdCLEF2024SharedAudioComposer,
    BirdCLEF2024VadBasedSharedAudioComposer,
)
from .dataset import (
    BirdCLEF2024AudioDataset,
    BirdCLEF2024PrimaryLabelDataset,
    BirdCLEF2024PrimaryLabelDistillationDataset,
    WeightedBirdCLEF2024PrimaryLabelDataset,
)
from .sampler import BirdCLEF2024WeightedRandomSampler

__all__ = [
    "BirdCLEF2024PrimaryLabelDataset",
    "BirdCLEF2024AudioDataset",
    "WeightedBirdCLEF2024PrimaryLabelDataset",
    "BirdCLEF2024PrimaryLabelDistillationDataset",
    "BirdCLEF2024WeightedRandomSampler",
    "BirdCLEF2024PrimaryLabelComposer",
    "BirdCLEF2024PrimaryLabelDistillationComposer",
    "BirdCLEF2024AudioComposer",
    "BirdCLEF2024AudioChunkingComposer",
    "BirdCLEF2024SharedAudioComposer",
    "BirdCLEF2024VadBasedSharedAudioComposer",
    "BirdCLEF2024BaselineCollator",
    "BirdCLEF2024AudioChunkingCollator",
    "BirdCLEF2024ChunkingCollator",
    "decode_csv_line",
    "select_birdclef2024_samples",
]


def decode_csv_line(
    line: List[str],
    version: Optional[Union[str, int]] = None,
) -> Dict[str, Any]:
    """Decode line of train_metadata.csv.

    Args:
        line (list): One line of train_metadata.csv split by comma (,).
        version (str or int, optional): Version information.

    Returns:
        dict: Dictionary containing metadata of given line.

    .. note::

        Returned dictionary contains following values.

            - filename (str): Filename with out extension. e.g. ``asbfly/XC134896``.
            - primary_label (str): Primary label of bird species.
            - secondary_label (list): Secondary labels of bird species.
            - type (list): Chirp types.
            - latitude (float, optional): Latitude of recording.
            - longitude (float, optional): Longitude of recording.
            - scientific_name (str): Scientific name of bird.
            - common_name (str): Common name of bird.
            - rating (float): Rating.
            - path (str): Path to audio file equivalent to ``filename`` + ``.ogg``.
                e.g. ``asbfly/XC134896.ogg``.

    """
    if version is None:
        if len(line) == 12:
            version = 2023
        elif len(line) == 13:
            version = 2022
        else:
            raise ValueError("Invalid format of line is detected.")

    version = int(version)

    if version == 2022:
        (
            primary_label,
            secondary_labels,
            chirp_types,
            latitude,
            longitude,
            scientific_name,
            common_name,
            _,
            _,
            rating,
            _,
            _,
            path,
        ) = line
    elif version in [2023, 2024]:
        (
            primary_label,
            secondary_labels,
            chirp_types,
            latitude,
            longitude,
            scientific_name,
            common_name,
            _,
            _,
            rating,
            _,
            path,
        ) = line
    else:
        raise ValueError("Invalid format of line is detected.")

    secondary_labels = ast.literal_eval(secondary_labels)
    chirp_types = ast.literal_eval(chirp_types)
    secondary_labels = [secondary_label.lower() for secondary_label in secondary_labels]
    chirp_types = [chirp_type.lower() for chirp_type in chirp_types]

    filename, _ = os.path.splitext(path)

    if len(latitude) > 0:
        latitude = float(latitude)
    else:
        latitude = None

    if len(longitude) > 0:
        longitude = float(longitude)
    else:
        longitude = None

    data = {
        "filename": filename,
        "primary_label": primary_label,
        "secondary_label": secondary_labels,
        "type": chirp_types,
        "latitude": latitude,
        "longitude": longitude,
        "scientific_name": scientific_name,
        "common_name": common_name,
        "rating": float(rating),
        "path": path,
    }

    return data


def select_birdclef2024_samples(
    path: str,
    existing_list_path: Optional[str] = None,
    train_list_path: Optional[str] = None,
    validation_list_path: Optional[str] = None,
) -> List[str]:
    """Split dataset into training and validation.

    Args:
        path (str): Path to csv file.

    Returns:
        list: Selected filenames.

    """
    existing_filenames = set()

    if existing_list_path is not None and os.path.exists(existing_list_path):
        with open(existing_list_path) as f:
            for line in f:
                filename = line.strip()
                existing_filenames.add(filename)

    if train_list_path is not None:
        with open(train_list_path) as f:
            for line in f:
                filename = line.strip()
                existing_filenames.add(filename)

    if validation_list_path is not None:
        with open(validation_list_path) as f:
            for line in f:
                filename = line.strip()
                existing_filenames.add(filename)

    filenames = []

    with open(path) as f:
        reader = csv.reader(f)

        for idx, line in enumerate(reader):
            if idx < 1:
                continue

            primary_label, *_, filename = line

            if primary_label in birdclef2024_primary_labels:
                # sample whose class is used in BirdCLEF2024
                _filename, _ = os.path.splitext(filename)

                if _filename in existing_filenames:
                    # If given sample is included in existing_filenames,
                    # then skip it to avoid data leakage and duplicates.
                    pass
                else:
                    filenames.append(_filename)

    return filenames
