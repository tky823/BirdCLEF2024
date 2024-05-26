import csv
import glob
import os
from typing import Any, Dict, Iterator, List, Optional

import torch
import torchaudio
from audyn.utils.data.birdclef.birdclef2024 import decode_csv_line
from audyn.utils.data.birdclef.birdclef2024.dataset import (
    BirdCLEF2024AudioDataset as _BirdCLEF2024AudioDataset,
)
from torch.utils.data import IterableDataset, get_worker_info

from .sampler import BirdCLEF2024WeightedRandomSampler

__all__ = [
    "BirdCLEF2024AudioDataset",
    "WeightedBirdCLEF2024PrimaryLabelDataset",
]


class BirdCLEF2024AudioDataset(_BirdCLEF2024AudioDataset):
    """Dataset for inference of bird classification model.

    Args:
        list_path (str): Path to list file. Each entry represents path to audio file
            without extension such as ``soundscape_ABCDE``.
        feature_dir (str): Path to dataset containing ``sample_submission.csv`` file,
            ``test_soundscapes`` directory, and so on.
        audio_key (str): Key of audio.
        sample_rate_key (str): Key of sampling rate.
        filename_key (str): Key of filename in given sample.
        decode_audio_as_waveform (bool, optional): If ``True``, audio is decoded as waveform
            tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
            waveform tensor and sampling rate. Default: ``True``.
        decode_audio_as_monoral (bool, optional): If ``True``, decoded audio is treated as
            monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
            shape of waveform is (num_channels, num_samples), which is returned by
            ``torchaudio.load``. Default: ``True``.

    .. note::

        If test set is unavailable, training set of unlabeled dataset is used.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        audio_key: str = "audio",
        sample_rate_key: str = "sample_rate",
        filename_key: str = "filename",
        decode_audio_as_waveform: Optional[bool] = None,
        decode_audio_as_monoral: Optional[bool] = None,
    ) -> None:
        super().__init__(
            list_path,
            feature_dir,
            audio_key=audio_key,
            sample_rate_key=sample_rate_key,
            filename_key=filename_key,
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        audio_paths = sorted(glob.glob(os.path.join(self.audio_root, "*.ogg")))

        if len(audio_paths) == 0:
            # fall back to unlabeled dataset
            self.fallback_to_unlabeled_audio(list_path, feature_dir)

    def fallback_to_unlabeled_audio(self, list_path: str, feature_dir: str) -> None:
        list_dir = os.path.dirname(list_path)

        audio_root = os.path.join(feature_dir, "unlabeled_soundscapes")
        list_path = os.path.join(list_dir, "unlabeled_validation.txt")

        filenames = []

        with open(list_path) as f:
            for line in f:
                filename = line.strip()
                filenames.append(filename)

        self.audio_root = audio_root
        self.list_path = list_path
        self.filenames = filenames


class WeightedBirdCLEF2024PrimaryLabelDataset(IterableDataset):
    """Dataset for training of bird classification model using weighted random sampling.

    Args:
        list_path (str): Path to list file. Each entry represents path to audio file
            without extension such as ``asbfly/XC49755``.
        feature_dir (str): Path to dataset containing ``train_metadata.csv`` file,
            ``train_audio`` directory, and so on.
        audio_key (str): Key of audio.
        sample_rate_key (str): Key of sampling rate.
        label_name_key (str): Key of prmary label name in given sample.
        filename_key (str): Key of filename in given sample.
        replacement (bool): If ``True``, samples are taken with replacement.
        smooth (int): Offset to frequency of each class. In [#koutini2022efficient]_, ``1000``
            is used. Default: ``1``.
        decode_audio_as_waveform (bool, optional): If ``True``, audio is decoded as waveform
            tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
            waveform tensor and sampling rate. Default: ``True``.
        decode_audio_as_monoral (bool, optional): If ``True``, decoded audio is treated as
            monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
            shape of waveform is (num_channels, num_samples), which is returned by
            ``torchaudio.load``. Default: ``True``.

    .. note::

        This class does not support distributed training.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        audio_key: str = "audio",
        sample_rate_key: str = "sample_rate",
        label_name_key: str = "primary_label",
        filename_key: str = "filename",
        replacement: bool = True,
        smooth: float = 1,
        decode_audio_as_waveform: Optional[bool] = None,
        decode_audio_as_monoral: Optional[bool] = None,
    ) -> None:
        super().__init__()

        audio_root = os.path.join(feature_dir, "train_audio")
        csv_path = os.path.join(feature_dir, "train_metadata.csv")

        self.audio_root = audio_root
        self.csv_path = csv_path
        self.list_path = list_path

        self.audio_key = audio_key
        self.sample_rate_key = sample_rate_key
        self.label_name_key = label_name_key
        self.filename_key = filename_key

        if decode_audio_as_waveform is None:
            decode_audio_as_waveform = True

        if decode_audio_as_monoral is None:
            decode_audio_as_monoral = True

        self.decode_audio_as_waveform = decode_audio_as_waveform
        self.decode_audio_as_monoral = decode_audio_as_monoral

        filenames = []
        primary_label_mapping = {}

        with open(list_path) as f:
            for line in f:
                filename = line.strip()
                filenames.append(filename)

        with open(csv_path) as f:
            reader = csv.reader(f)

            for idx, line in enumerate(reader):
                if idx < 1:
                    continue

                data = decode_csv_line(line)
                filename = data["filename"]

                if filename in filenames:
                    primary_label = data["primary_label"]
                    primary_label_mapping[filename] = primary_label

        self.filenames = filenames
        self.primary_label_mapping = primary_label_mapping
        self.worker_id = None

        self.set_sampler(
            label_mapping=primary_label_mapping,
            length=len(filenames),
            replacement=replacement,
            smooth=smooth,
            filenames=self.filenames,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        audio_root = self.audio_root
        primary_label_mapping = self.primary_label_mapping

        if self.worker_id is None:
            # should be initialized
            worker_info = get_worker_info()

            if worker_info is None:
                self.worker_id = 0
                num_workers = 1
            else:
                self.worker_id = worker_info.id
                num_workers = worker_info.num_workers

            num_total_samples = self.sampler.num_samples
            num_samples_per_worker = num_total_samples // num_workers

            if self.worker_id < num_total_samples % num_workers:
                num_samples_per_worker += 1

            self.sampler.num_samples = num_samples_per_worker

        for idx in self.sampler:
            filename = self.filenames[idx]

            audio_path = os.path.join(audio_root, f"{filename}.ogg")
            waveform, sample_rate = torchaudio.load(audio_path)

            if self.decode_audio_as_monoral:
                waveform = waveform.mean(dim=0)

            if self.decode_audio_as_waveform:
                audio = waveform
            else:
                audio = waveform, sample_rate

            sample_rate = torch.tensor(sample_rate, dtype=torch.long)

            primary_label = primary_label_mapping[filename]

            feature = {
                self.audio_key: audio,
                self.sample_rate_key: sample_rate,
                self.label_name_key: primary_label,
                self.filename_key: filename,
            }

            yield feature

    def __len__(self) -> int:
        return self.sampler.num_samples

    def set_sampler(
        self,
        label_mapping: Dict[str, str],
        length: int,
        replacement: bool = True,
        smooth: float = 1,
        filenames: List[str] = None,
    ) -> None:
        if filenames is None:
            filenames = self.filenames

        self.sampler = BirdCLEF2024WeightedRandomSampler(
            label_mapping,
            length,
            replacement=replacement,
            smooth=smooth,
            filenames=filenames,
        )
