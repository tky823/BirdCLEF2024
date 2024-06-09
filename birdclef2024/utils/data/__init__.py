import ast
import csv
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from audyn.utils import audyn_cache_dir
from audyn.utils.data.birdclef.birdclef2024 import primary_labels as birdclef2024_primary_labels

from ._download import download_birdclef2024_pretrain_primary_labels
from .collator import (
    BirdCLEF2024AudioChunkingCollator,
    BirdCLEF2024BaselineCollator,
    BirdCLEF2024PretrainCollator,
)
from .composer import (
    BirdCLEF2024AudioChunkingComposer,
    BirdCLEF2024AudioComposer,
    BirdCLEF2024PretrainPrimaryLabelComposer,
    BirdCLEF2024PrimaryLabelComposer,
    BirdCLEF2024PrimaryLabelDistillationComposer,
    BirdCLEF2024SharedAudioComposer,
    BirdCLEF2024VadBasedSharedAudioComposer,
)
from .dataset import (
    BirdCLEF2024AudioDataset,
    BirdCLEF2024PrimaryLabelDataset,
    BirdCLEF2024PrimaryLabelDistillationDataset,
    BirdCLEF2024PrimaryLabelMultiDataset,
    BirdCLEF2024PrimaryLabelMultiWebDataset,
    WeightedBirdCLEF2024PrimaryLabelDataset,
)
from .sampler import BirdCLEF2024WeightedRandomSampler

__all__ = [
    "BirdCLEF2024PrimaryLabelDataset",
    "BirdCLEF2024AudioDataset",
    "WeightedBirdCLEF2024PrimaryLabelDataset",
    "BirdCLEF2024PrimaryLabelDistillationDataset",
    "BirdCLEF2024PrimaryLabelMultiDataset",
    "BirdCLEF2024PrimaryLabelMultiWebDataset",
    "BirdCLEF2024WeightedRandomSampler",
    "BirdCLEF2024PrimaryLabelComposer",
    "BirdCLEF2024PretrainPrimaryLabelComposer",
    "BirdCLEF2024PrimaryLabelDistillationComposer",
    "BirdCLEF2024AudioComposer",
    "BirdCLEF2024AudioChunkingComposer",
    "BirdCLEF2024SharedAudioComposer",
    "BirdCLEF2024VadBasedSharedAudioComposer",
    "BirdCLEF2024BaselineCollator",
    "BirdCLEF2024PretrainCollator",
    "BirdCLEF2024AudioChunkingCollator",
    "birdclef2024_pretrain_primary_labels",
    "num_birdclef2024_pretrain_primary_labels",
    "decode_csv_line",
    "select_seen_class_samples",
    "select_unseen_samples",
    "stratified_split",
    "stratified_split_2024",
    "stratified_split_unseen_samples_2023",
    "stratified_split_unseen_samples_2022",
    "stratified_split_unseen_samples_2021",
]

birdclef2024_pretrain_primary_labels = download_birdclef2024_pretrain_primary_labels()
num_birdclef2024_pretrain_primary_labels = len(birdclef2024_pretrain_primary_labels)


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
        elif len(line) == 14:
            version = 2021
        else:
            raise ValueError("Invalid format of line is detected.")

    version = int(version)

    if version == 2021:
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
            filename,
            _,
            rating,
            _,
            _,
        ) = line
        path = os.path.join(primary_label, filename)
    elif version == 2022:
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


def select_seen_class_samples(
    path: str,
    existing_list_path: Optional[str] = None,
    train_list_path: Optional[str] = None,
    validation_list_path: Optional[str] = None,
) -> List[str]:
    """Select samples from seen class.

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


def stratified_split_unseen_samples(
    primary_labels: List[str],
    path: str,
    train_ratio: float,
    existing_list_path: Optional[str] = None,
    train_list_path: Optional[str] = None,
    validation_list_path: Optional[str] = None,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Select unseen samples and split dataset.

    Args:
        path (str): Path to csv file.

    Returns:
        tuple: Tuple of lists.

            - list: Training filenames.
            - list: Validation filenames.

    """
    g = torch.Generator()
    g.manual_seed(seed)

    filenames = {primary_label: [] for primary_label in primary_labels}
    train_filenames = []
    validation_filenames = []

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

    with open(path) as f:
        reader = csv.reader(f)

        for idx, line in enumerate(reader):
            if idx < 1:
                continue

            if len(line) == 12:
                version = 2023
            elif len(line) == 13:
                version = 2022
            elif len(line) == 14:
                version = 2021
            else:
                raise ValueError("Invalid format of line is detected.")

            version = int(version)

            if version == 2021:
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
                    filename,
                    _,
                    rating,
                    _,
                    _,
                ) = line
                path = os.path.join(primary_label, filename)
            elif version == 2022:
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

            filename, _ = os.path.splitext(path)

            if filename in existing_filenames:
                # If given sample is included in existing_filenames,
                # then skip it to avoid data leakage and duplicates.
                pass
            else:
                filenames[primary_label].append(filename)

    # split dataset
    for primary_label, _filenames in filenames.items():
        num_files = len(_filenames)
        indices = torch.randperm(num_files, generator=g).tolist()

        for idx in indices[: int(train_ratio * num_files)]:
            train_filenames.append(_filenames[idx])

        for idx in indices[int(train_ratio * num_files) :]:
            validation_filenames.append(_filenames[idx])

    return train_filenames, validation_filenames


def stratified_split(
    primary_labels: List[str],
    path: str,
    train_ratio: float,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split dataset into training and validation.

    Args:
        path (str): Path to csv file.
        train_ratio (float): Ratio of training set.
        seed (int): Random seed.

    Returns:
        tuple: Splits of filenames.

            - list: List of training filenames.
            - list: List of validation filenames.

    """
    g = torch.Generator()
    g.manual_seed(seed)

    filenames = {primary_label: [] for primary_label in primary_labels}
    train_filenames = []
    validation_filenames = []

    with open(path) as f:
        reader = csv.reader(f)

        for idx, line in enumerate(reader):
            if idx < 1:
                continue

            primary_label, *_, filename = line
            filenames[primary_label].append(filename)

    # split dataset
    for primary_label, _filenames in filenames.items():
        num_files = len(_filenames)
        indices = torch.randperm(num_files, generator=g).tolist()

        for idx in indices[: int(train_ratio * num_files)]:
            train_filenames.append(_filenames[idx])

        for idx in indices[int(train_ratio * num_files) :]:
            validation_filenames.append(_filenames[idx])

    return train_filenames, validation_filenames


def stratified_split_2024(
    path: str,
    train_ratio: float,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split dataset into training and validation.

    Args:
        path (str): Path to csv file.
        train_ratio (float): Ratio of training set.
        seed (int): Random seed.

    Returns:
        tuple: Splits of filenames.

            - list: List of training filenames.
            - list: List of validation filenames.

    """
    return stratified_split(
        birdclef2024_primary_labels,
        path,
        train_ratio=train_ratio,
        seed=seed,
    )


def stratified_split_unseen_samples_2023(
    path: str,
    train_ratio: float,
    existing_list_path: Optional[str] = None,
    train_list_path: Optional[str] = None,
    validation_list_path: Optional[str] = None,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split dataset into training and validation.

    Args:
        path (str): Path to csv file.
        train_ratio (float): Ratio of training set.
        seed (int): Random seed.

    Returns:
        tuple: Splits of filenames.

            - list: List of training filenames.
            - list: List of validation filenames.

    """
    primary_labels = set()
    primary_labels_path = os.path.join(
        audyn_cache_dir, "data", "birdclef2023", "primary-labels.txt"
    )

    with open(primary_labels_path) as f:
        for line in f:
            line = line.strip()
            primary_labels.add(line)

    primary_labels = sorted(list(primary_labels))

    train_filenames, validation_filenames = stratified_split_unseen_samples(
        primary_labels,
        path,
        train_ratio=train_ratio,
        existing_list_path=existing_list_path,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
        seed=seed,
    )

    return train_filenames, validation_filenames


def stratified_split_unseen_samples_2022(
    path: str,
    train_ratio: float,
    existing_list_path: Optional[str] = None,
    train_list_path: Optional[str] = None,
    validation_list_path: Optional[str] = None,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split dataset into training and validation.

    Args:
        path (str): Path to csv file.
        train_ratio (float): Ratio of training set.
        seed (int): Random seed.

    Returns:
        tuple: Splits of filenames.

            - list: List of training filenames.
            - list: List of validation filenames.

    """
    primary_labels = set()
    primary_labels_path = os.path.join(
        audyn_cache_dir, "data", "birdclef2022", "primary-labels.txt"
    )

    with open(primary_labels_path) as f:
        for line in f:
            line = line.strip()
            primary_labels.add(line)

    train_filenames, validation_filenames = stratified_split_unseen_samples(
        primary_labels,
        path,
        train_ratio=train_ratio,
        existing_list_path=existing_list_path,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
        seed=seed,
    )

    return train_filenames, validation_filenames


def stratified_split_unseen_samples_2021(
    path: str,
    train_ratio: float,
    existing_list_path: Optional[str] = None,
    train_list_path: Optional[str] = None,
    validation_list_path: Optional[str] = None,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split dataset into training and validation.

    Args:
        path (str): Path to csv file.
        train_ratio (float): Ratio of training set.
        seed (int): Random seed.

    Returns:
        tuple: Splits of filenames.

            - list: List of training filenames.
            - list: List of validation filenames.

    """
    primary_labels = set()
    primary_labels_path = os.path.join(
        audyn_cache_dir, "data", "birdclef2021", "primary-labels.txt"
    )

    with open(primary_labels_path) as f:
        for line in f:
            line = line.strip()
            primary_labels.add(line)

    train_filenames, validation_filenames = stratified_split_unseen_samples(
        primary_labels,
        path,
        train_ratio=train_ratio,
        existing_list_path=existing_list_path,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
        seed=seed,
    )

    return train_filenames, validation_filenames
