"""Save features of training/validation samples in BirdCLEF2024.
- audio.ogg: Raw audio.
- primary_label.txt: Primary label.
- filename.txt: Filename w/o .ogg.
- sample_rate.pth: Sampling rate.
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue
from typing import Any, Dict, List

import audyn
import torch
import torchaudio
import webdataset as wds
from audyn.utils import setup_config
from audyn.utils.data.birdclef.birdclef2024 import decode_csv_line
from omegaconf import DictConfig
from tqdm import tqdm


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    feature_dir = config.preprocess.feature_dir
    audio_root = config.preprocess.audio_root
    csv_path = config.preprocess.csv_path
    subset = config.preprocess.subset
    max_workers = config.preprocess.max_workers
    max_shard_size = config.preprocess.max_shard_size

    assert list_path is not None, "Specify preprocess.list_path."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert audio_root is not None, "Specify preprocess.audio_root."
    assert csv_path is not None, "Specify preprocess.csv_path."
    assert subset is not None, "Specify preprocess.subset."
    assert max_workers is not None, "Specify preprocess.max_workers."

    os.makedirs(feature_dir, exist_ok=True)

    filenames = []
    files = {}

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
                data["root"] = audio_root
                files[filename] = data

    if dump_format == "torch":
        sorted_files = [files[filename] for filename in filenames]

        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for data in sorted_files:
                    filename = data["filename"]
                    feature_path = os.path.join(feature_dir, f"{filename}.pth")
                    future = executor.submit(
                        process_torch,
                        data,
                        feature_path=feature_path,
                    )
                    futures.append(future)

                for future in tqdm(futures):
                    future.result()
        else:
            for data in tqdm(sorted_files):
                filename = data["filename"]
                feature_path = os.path.join(feature_dir, f"{filename}.pth")
                process_torch(
                    data,
                    feature_path=feature_path,
                )
    elif dump_format == "webdataset":
        template_path = os.path.join(feature_dir, "%d.tar")

        if subset == "train":
            indices = torch.randperm(len(filenames)).tolist()
            filenames = [filenames[idx] for idx in indices]
        elif subset == "validation":
            pass
        else:
            raise ValueError(f"{subset} is not supported.")

        # reflect order of filenames
        sorted_files = [files[filename] for filename in filenames]
        subsets = [[] for _ in range(max_workers)]

        with open(list_path) as f:
            for idx, file in tqdm(enumerate(sorted_files)):
                subsets[idx % max_workers].append(file)

        queue = Queue()

        # load
        loading_processes: List[Process] = []

        for files in subsets:
            p = Process(
                target=process_webdataset,
                args=(queue,),
                kwargs={
                    "files": files,
                },
            )
            loading_processes.append(p)

        # write
        writing_process = Process(
            target=write_to_shards,
            args=(queue,),
            kwargs={
                "num_workers": max_workers,
                "tar_path": template_path,
                "max_shard_size": max_shard_size,
            },
        )

        # start multiprocessing
        for p in loading_processes:
            p.start()

        writing_process.start()

        # finish multiprocessing
        for p in loading_processes:
            p.join()

        writing_process.join()
    elif dump_format == "birdclef2024":
        pass
    else:
        raise ValueError(f"Invalid dump_format={dump_format} is detected.")


def process_torch(
    data: Dict[str, Any],
    feature_path: str,
) -> None:
    feature = {}

    filename = data["filename"]
    primary_label = data["primary_label"]
    audio_root = data["root"]
    audio_path = data["path"]

    audio_path = os.path.join(audio_root, audio_path)
    metadata = torchaudio.info(audio_path)

    with open(audio_path, mode="rb") as f:
        audio = f.read()

    feature["audio.ogg"] = audio
    feature["primary_label"] = primary_label
    feature["filename"] = filename
    feature["sample_rate"] = torch.tensor(
        metadata.sample_rate,
        dtype=torch.long,
    )

    feature_dir = os.path.dirname(feature_path)
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(feature, feature_path)


def process_webdataset(
    queue: Queue,
    files: List[Dict[str, Any]] = None,
) -> None:
    if files is not None:
        for data in files:
            feature = {}

            filename = data["filename"]
            primary_label = data["primary_label"]
            audio_root = data["root"]
            audio_path = data["path"]

            m4a_path = os.path.join(audio_root, audio_path)
            metadata = torchaudio.info(m4a_path)

            with open(m4a_path, mode="rb") as f:
                audio = f.read()

            feature["__key__"] = filename
            feature["audio.ogg"] = audio
            feature["primary_label.txt"] = primary_label
            feature["filename.txt"] = filename
            feature["sample_rate.pth"] = torch.tensor(
                metadata.sample_rate,
                dtype=torch.long,
            )

            queue.put(feature)

    queue.put(None)


def write_to_shards(
    queue: Queue,
    num_workers: int = 1,
    tar_path: str = None,
    max_shard_size: int = 1,
) -> None:
    num_working_processes = num_workers

    if tar_path is None:
        raise ValueError("Specify tar_path.")

    with wds.ShardWriter(tar_path, maxsize=max_shard_size) as sink:
        while True:
            feature = queue.get()

            if feature is None:
                num_working_processes -= 1
            else:
                sink.write(feature)

            if num_working_processes == 0:
                break


if __name__ == "__main__":
    main()
