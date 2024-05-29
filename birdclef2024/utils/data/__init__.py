import csv
import os
from typing import List, Optional

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
    "select_birdclef2024_samples",
]


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
