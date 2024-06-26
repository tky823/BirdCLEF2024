import csv
import glob
import json
import os
import re
import tarfile
from io import BytesIO
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type

import torch
import torchaudio
from audyn.utils.data.birdclef.birdclef2024.dataset import (
    BirdCLEF2024AudioDataset as _BirdCLEF2024AudioDataset,
)
from audyn.utils.data.birdclef.birdclef2024.dataset import (
    BirdCLEF2024PrimaryLabelDataset as _BirdCLEF2024PrimaryLabelDataset,
)
from audyn.utils.data.webdataset import (
    supported_audio_extensions,
    supported_json_extensions,
    supported_text_extensions,
    supported_torchdump_extensions,
)
from torch.utils.data import (
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
    get_worker_info,
)

from .sampler import BirdCLEF2024WeightedRandomSampler

__all__ = [
    "BirdCLEF2024PrimaryLabelDataset",
    "BirdCLEF2024AudioDataset",
    "WeightedBirdCLEF2024PrimaryLabelDataset",
    "BirdCLEF2024PrimaryLabelDistillationDataset",
    "BirdCLEF2024PrimaryLabelMultiDataset",
    "BirdCLEF2024PrimaryLabelMultiWebDataset",
]


class BirdCLEF2024PrimaryLabelDataset(_BirdCLEF2024PrimaryLabelDataset):
    """Alias of audyn.utils.data.birdclef.birdclef2024.dataset.BirdCLEF2024PrimaryLabelDataset"""


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

        from . import decode_csv_line

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


class BirdCLEF2024PrimaryLabelDistillationDataset(Dataset):
    """Dataset for training of bird classification model.

    Args:
        labeled_list_path (str): Path to list file with known primary label. Each entry represents
            path to audio file without extension such as ``abethr1/XC128013``.
        unlabeled_list_path (str): Path to list file with unknown primary label. Each entry
            represents path to audio file without extension such as ``460830``.
        feature_dir (str): Path to dataset containing ``train_metadata.csv`` file,
            ``train_audio`` directory, and so on.
        labeled_audio_key (str): Key of labeled audio.
        labeled_sample_rate_key (str): Key of sampling rate of labeled audio.
        label_name_key (str): Key of prmary label name in given sample.
        labeled_filename_key (str): Key of filename of labeled audio in given sample.
        unlabeled_audio_key (str): Key of unlabeled audio.
        unlabeled_sample_rate_key (str): Key of sampling rate of unlabeled audio.
        unlabeled_filename_key (str): Key of filename of unlabeled audio in given sample.
        seed (int): Random seed to sample unlabeled audio.
        decode_audio_as_waveform (bool, optional): If ``True``, audio is decoded as waveform
            tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
            waveform tensor and sampling rate. Default: ``True``.
        decode_audio_as_monoral (bool, optional): If ``True``, decoded audio is treated as
            monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
            shape of waveform is (num_channels, num_samples), which is returned by
            ``torchaudio.load``. Default: ``True``.

    """

    def __init__(
        self,
        labeled_list_path: str,
        unlabeled_list_path: str,
        feature_dir: str,
        labeled_audio_key: str = "labeled_audio",
        labeled_sample_rate_key: str = "labeled_sample_rate",
        label_name_key: str = "primary_label",
        labeled_filename_key: str = "labeled_filename",
        unlabeled_audio_key: str = "unlabeled_audio",
        unlabeled_sample_rate_key: str = "unlabeled_sample_rate",
        unlabeled_filename_key: str = "unlabeled_filename",
        duration: Optional[float] = 15,
        seed: int = 0,
        decode_audio_as_waveform: Optional[bool] = None,
        decode_audio_as_monoral: Optional[bool] = None,
    ) -> None:
        super().__init__()

        from . import decode_csv_line

        labeled_audio_root = os.path.join(feature_dir, "train_audio")
        unlabeled_audio_root = os.path.join(feature_dir, "unlabeled_soundscapes")
        csv_path = os.path.join(feature_dir, "train_metadata.csv")

        self.labeled_audio_root = labeled_audio_root
        self.unlabeled_audio_root = unlabeled_audio_root
        self.csv_path = csv_path
        self.labeled_list_path = labeled_list_path
        self.unlabeled_list_path = unlabeled_list_path

        self.labeled_audio_key = labeled_audio_key
        self.labeled_sample_rate_key = labeled_sample_rate_key
        self.label_name_key = label_name_key
        self.labeled_filename_key = labeled_filename_key
        self.unlabeled_audio_key = unlabeled_audio_key
        self.unlabeled_sample_rate_key = unlabeled_sample_rate_key
        self.unlabeled_filename_key = unlabeled_filename_key

        self.duration = duration
        self.seed = seed
        self.generator = None

        if decode_audio_as_waveform is None:
            decode_audio_as_waveform = True

        if decode_audio_as_monoral is None:
            decode_audio_as_monoral = True

        self.decode_audio_as_waveform = decode_audio_as_waveform
        self.decode_audio_as_monoral = decode_audio_as_monoral

        labeled_filenames = []
        unlabeled_filenames = []
        primary_label_mapping = {}

        with open(labeled_list_path) as f:
            for line in f:
                filename = line.strip()
                labeled_filenames.append(filename)

        with open(unlabeled_list_path) as f:
            for line in f:
                filename = line.strip()
                unlabeled_filenames.append(filename)

        with open(csv_path) as f:
            reader = csv.reader(f)

            for idx, line in enumerate(reader):
                if idx < 1:
                    continue

                data = decode_csv_line(line)
                filename = data["filename"]

                if filename in labeled_filenames:
                    primary_label = data["primary_label"]
                    primary_label_mapping[filename] = primary_label

        self.labeled_filenames = labeled_filenames
        self.unlabeled_filenames = unlabeled_filenames
        self.primary_label_mapping = primary_label_mapping

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        labeled_audio_root = self.labeled_audio_root
        labeled_filename = self.labeled_filenames[idx]
        primary_label_mapping = self.primary_label_mapping

        unlabeled_audio_root = self.unlabeled_audio_root
        unlabeled_filenames = self.unlabeled_filenames

        duration = self.duration

        if self.generator is None:
            # should be initialized
            worker_info = get_worker_info()

            if worker_info is None:
                worker_id = 0
            else:
                worker_id = worker_info.id

            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed + worker_id)

        g = self.generator

        unlabeled_idx = torch.randint(0, len(unlabeled_filenames), (), generator=g)
        unlabeled_idx = unlabeled_idx.item()
        unlabeled_filename = unlabeled_filenames[unlabeled_idx]

        # labeled
        audio_path = os.path.join(labeled_audio_root, f"{labeled_filename}.ogg")
        metadata = torchaudio.info(audio_path)

        if duration is not None:
            sample_rate = metadata.sample_rate
            num_frames = metadata.num_frames
            length = int(sample_rate * duration)

            if length < num_frames:
                frame_offset = torch.randint(0, num_frames - length, (), generator=g)
                frame_offset = frame_offset.item()
            else:
                frame_offset = 0
                length = -1

            waveform, sample_rate = torchaudio.load(
                audio_path, frame_offset=frame_offset, num_frames=length
            )
        else:
            waveform, sample_rate = torchaudio.load(audio_path)

        if self.decode_audio_as_monoral:
            waveform = waveform.mean(dim=0)

        if self.decode_audio_as_waveform:
            labeled_audio = waveform
        else:
            labeled_audio = waveform, sample_rate

        labeled_sample_rate = torch.tensor(sample_rate, dtype=torch.long)
        primary_label = primary_label_mapping[labeled_filename]

        # unlabeled
        audio_path = os.path.join(unlabeled_audio_root, f"{unlabeled_filename}.ogg")
        metadata = torchaudio.info(audio_path)

        if duration is not None:
            sample_rate = metadata.sample_rate
            num_frames = metadata.num_frames
            length = int(sample_rate * duration)

            if length < num_frames:
                frame_offset = torch.randint(0, num_frames - length, (), generator=g)
                frame_offset = frame_offset.item()
            else:
                frame_offset = 0
                length = -1

            frame_offset = torch.randint(0, num_frames - length, (), generator=g)
            frame_offset = frame_offset.item()
            waveform, sample_rate = torchaudio.load(
                audio_path, frame_offset=frame_offset, num_frames=length
            )
        else:
            waveform, sample_rate = torchaudio.load(audio_path)

        if self.decode_audio_as_monoral:
            waveform = waveform.mean(dim=0)

        if self.decode_audio_as_waveform:
            unlabeled_audio = waveform
        else:
            unlabeled_audio = waveform, sample_rate

        unlabeled_sample_rate = torch.tensor(sample_rate, dtype=torch.long)

        feature = {
            self.labeled_audio_key: labeled_audio,
            self.labeled_sample_rate_key: labeled_sample_rate,
            self.label_name_key: primary_label,
            self.labeled_filename_key: labeled_filename,
            self.unlabeled_audio_key: unlabeled_audio,
            self.unlabeled_sample_rate_key: unlabeled_sample_rate,
            self.unlabeled_filename_key: unlabeled_filename,
        }

        return feature

    def __len__(self) -> int:
        return len(self.labeled_filenames)


class BirdCLEF2024PrimaryLabelMultiDataset(Dataset):
    """Dataset for training of bird classification model using BirdCLEF2021-2024.

    Args:
        list_path (str): Path to list file. Each entry represents path to audio file
            without extension such as ``birdclef-2024/asbfly/XC49755``.
        feature_dir (str): Path to dataset containing ``train_metadata.csv`` file,
            ``train_audio`` directory, and so on.
        audio_key (str): Key of audio.
        sample_rate_key (str): Key of sampling rate.
        label_name_key (str): Key of prmary label name in given sample.
        filename_key (str): Key of filename in given sample.
        decode_audio_as_waveform (bool, optional): If ``True``, audio is decoded as waveform
            tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
            waveform tensor and sampling rate. Default: ``True``.
        decode_audio_as_monoral (bool, optional): If ``True``, decoded audio is treated as
            monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
            shape of waveform is (num_channels, num_samples), which is returned by
            ``torchaudio.load``. Default: ``True``.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        audio_key: str = "audio",
        sample_rate_key: str = "sample_rate",
        label_name_key: str = "primary_label",
        filename_key: str = "filename",
        decode_audio_as_waveform: Optional[bool] = None,
        decode_audio_as_monoral: Optional[bool] = None,
    ) -> None:
        super().__init__()

        from . import decode_csv_line

        self.list_path = list_path
        self.feature_dir = feature_dir

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

        files = []
        filenames_by_year = {}
        primary_label_mapping = {}

        with open(list_path) as f:
            for line in f:
                filename = line.strip()
                challenge, filename = filename.split("/", maxsplit=1)

                if challenge not in filenames_by_year:
                    filenames_by_year[challenge] = []

                filenames_by_year[challenge].append(filename)
                files.append(
                    {
                        "challenge": challenge,
                        "filename": filename,
                    }
                )

        for challenge in filenames_by_year.keys():
            csv_path = os.path.join(feature_dir, challenge, "train_metadata.csv")

            with open(csv_path) as f:
                reader = csv.reader(f)

                for idx, line in enumerate(reader):
                    if idx < 1:
                        continue

                    data = decode_csv_line(line)
                    filename = data["filename"]

                    if filename in filenames_by_year[challenge]:
                        primary_label = data["primary_label"]
                        primary_label_mapping[filename] = primary_label

        self.files = files
        self.primary_label_mapping = primary_label_mapping

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feature_dir = self.feature_dir
        file = self.files[idx]
        primary_label_mapping = self.primary_label_mapping

        challenge = file["challenge"]
        filename = file["filename"]
        audio_root = os.path.join(feature_dir, challenge)

        if challenge in ["birdclef-2021"]:
            audio_path = os.path.join(audio_root, "train_short_audio", f"{filename}.ogg")
        else:
            audio_path = os.path.join(audio_root, "train_audio", f"{filename}.ogg")

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

        return feature

    def __len__(self) -> int:
        return len(self.files)


class BirdCLEF2024PrimaryLabelMultiWebDataset(IterableDataset):
    """Dataset for training of bird classification model using BirdCLEF2021-2024.

    Args:
        list_path (str): Path to list file. Each entry represents path to audio file
            without extension such as ``birdclef-2024/asbfly/XC49755``.
        feature_dir (str): Path to dataset containing ``train_metadata.csv`` file,
            ``train_audio`` directory, and so on.
        decode_audio_as_waveform (bool, optional): If ``True``, audio is decoded as waveform
            tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
            waveform tensor and sampling rate. Default: ``True``.
        decode_audio_as_monoral (bool, optional): If ``True``, decoded audio is treated as
            monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
            shape of waveform is (num_channels, num_samples), which is returned by
            ``torchaudio.load``. Default: ``True``.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        shuffle: bool = False,
        decode_audio_as_waveform: Optional[bool] = None,
        decode_audio_as_monoral: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.list_path = list_path
        self.feature_dir = feature_dir

        if decode_audio_as_waveform is None:
            decode_audio_as_waveform = True

        if decode_audio_as_monoral is None:
            decode_audio_as_monoral = True

        self.shuffle = shuffle
        self.decode_audio_as_waveform = decode_audio_as_waveform
        self.decode_audio_as_monoral = decode_audio_as_monoral

        challenges = set()

        with open(list_path) as f:
            for line in f:
                filename = line.strip()
                challenge, filename = filename.split("/", maxsplit=1)
                challenges.add(challenge)

        challenges = sorted(list(challenges))

        filenames = set()
        mapping = {}
        files: Dict[str, _PicklableFile] = {}

        for challenge in challenges:
            subset = self._challenge_to_subset(challenge, training=shuffle)

            for url in sorted(glob.glob(os.path.join(feature_dir, subset, "*.tar"))):
                with tarfile.open(url) as f:
                    for tarinfo in f:
                        filename, key = tarinfo.name.split(".", maxsplit=1)

                        if filename not in filenames:
                            filenames.add(filename)
                            mapping[filename] = {
                                "__url__": url,
                                "data": {},
                            }

                        data = {
                            "offset_data": tarinfo.offset_data,
                            "size": tarinfo.size,
                        }
                        mapping[filename]["data"][key] = data

                files[url] = _PicklableFile(url)

        filenames = set(mapping.keys())

        with open(list_path) as f:
            assert len(filenames) == sum(1 for _ in f)

        self.mapping = mapping
        self.files = files
        self.worker_id = None
        self.filenames = sorted(list(filenames))

        # set sampler
        self.set_sampler(
            filenames=self.filenames,
            shuffle=self.shuffle,
            replacement=False,
        )

        self.close_all()

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        if self.worker_id is None:
            # should be initialized
            worker_info = get_worker_info()

            if worker_info is None:
                self.worker_id = 0
                num_workers = 1
            else:
                self.worker_id = worker_info.id
                num_workers = worker_info.num_workers

            num_total_samples = len(self.sampler)
            num_longer_workers = num_total_samples % num_workers
            num_samples_per_worker = num_total_samples // num_workers

            if self.worker_id < num_longer_workers:
                start_index = (num_samples_per_worker + 1) * self.worker_id
                end_index = (num_samples_per_worker + 1) * (self.worker_id + 1)
            else:
                start_index = (num_samples_per_worker + 1) * num_longer_workers
                start_index += num_samples_per_worker * (self.worker_id - num_longer_workers)
                end_index = start_index + num_samples_per_worker

            self.sampler.data_source = self.filenames[start_index:end_index]

            for url in self.files.keys():
                self.files[url].close()
                self.files[url] = _PicklableFile(url)

        yield from self._decode()

    def __len__(self) -> int:
        return self.sampler.num_samples

    def _decode(self) -> Iterator[Dict[str, Any]]:
        """Return decoding iterator called in __iter__."""
        for index in self.sampler:
            filename = self.filenames[index]
            mapping = self.mapping[filename]
            url = mapping["__url__"]
            data: Dict[str, Any] = mapping["data"]
            f = self.files[url]

            sample = {
                "__key__": filename,
                "__url__": url,
            }

            for key, value in data.items():
                if key.startswith("__"):
                    continue

                offset_data = value["offset_data"]
                size = value["size"]

                f.seek(offset_data)
                binary = f.read(size)
                ext = re.sub(r".*[.]", "", key)

                # based on
                # https://github.com/webdataset/webdataset/blob/f11fd66c163722c607ec99475a6f3cb880ec35b8/webdataset/autodecode.py#L156
                if ext in supported_json_extensions:
                    decoded = json.loads(binary)
                elif ext in supported_text_extensions:
                    decoded = binary.decode()
                elif ext in supported_torchdump_extensions:
                    binary = BytesIO(binary)
                    decoded = torch.load(binary)
                elif ext in supported_audio_extensions:
                    # NOTE: Decoding is applied in composer like ordinary webdataset.
                    decoded = binary
                else:
                    raise ValueError(f"Invalid key {key} is detected.")

                sample[key] = decoded

            yield sample

    def _challenge_to_subset(self, challenge: str, training: bool = False) -> str:
        if challenge == "birdclef-2024":
            if training:
                subset = "train"
            else:
                subset = "validation"
        elif challenge == "birdclef-2023":
            if training:
                subset = "train_2023"
            else:
                subset = "validation_2023"
        elif challenge == "birdclef-2022":
            if training:
                subset = "train_2022"
            else:
                subset = "validation_2022"
        elif challenge == "birdclef-2021":
            if training:
                subset = "train_2021"
            else:
                subset = "validation_2021"
        else:
            raise NotImplementedError(f"{challenge} is not supported as challenge.")

        return subset

    def set_sampler(
        self,
        filenames: Optional[str] = None,
        shuffle: bool = False,
        replacement: bool = False,
    ) -> None:
        if filenames is None:
            filenames = self.filenames

        if shuffle:
            self.sampler = RandomSampler(
                filenames,
                replacement=replacement,
                num_samples=len(filenames),
            )
        else:
            assert not replacement

            self.sampler = SequentialSampler(filenames)

    def close_all(self, *args, **kwargs) -> None:
        """Close all tar files."""
        for url in self.files.keys():
            self.files[url].close(*args, **kwargs)


class _PicklableFile:
    """Wrapper class of io.BufferedReader to pickle."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.file = open(path, mode="rb")

    def __reduce__(self) -> Tuple[Type, Tuple[str]]:
        self.file.close()
        return self.__class__, (self.path,)

    def seek(self, *args, **kwargs) -> int:
        """Wrapper of file.seek."""
        return self.file.seek(*args, **kwargs)

    def read(self, *args, **kwargs) -> bytes:
        """Wrapper of file.read."""
        return self.file.read(*args, **kwargs)

    def close(self, *args, **kwargs) -> None:
        """Wrapper of file.close."""
        return self.file.close(*args, **kwargs)
