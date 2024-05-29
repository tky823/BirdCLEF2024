import glob
import os

import audyn
import torch
from audyn.utils.data.birdclef.birdclef2024 import primary_labels
from natsort import natsorted
from omegaconf import DictConfig

from birdclef2024.utils import setup_config


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    list_path = config.preprocess.list_path
    feature_dir = config.preprocess.feature_dir
    submission_path = config.preprocess.submission_path

    assert list_path is not None, "Specify preprocess.list_path."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert submission_path is not None, "Specify preprocess.submission_path."

    filenames = []

    with open(list_path) as f:
        for line in f:
            filename = line.strip()
            filenames.append(filename)

    submission_dir = os.path.dirname(submission_path)

    if submission_dir:
        os.makedirs(submission_dir, exist_ok=True)

    with open(submission_path, mode="w") as f:
        line = "row_id,"
        f.write(line)

        line = ",".join(primary_labels)
        f.write(line + "\n")

        for filename in filenames:
            template_path = os.path.join(feature_dir, f"{filename}_*.pth")
            paths = natsorted(glob.glob(template_path))

            for path in paths:
                estimated = torch.load(path, map_location=lambda storage, loc: storage)
                estimated = estimated.tolist()
                estimated = [str(_estimated) for _estimated in estimated]

                endtime_filename = os.path.basename(path)
                endtime_filename, _ = os.path.splitext(endtime_filename)
                line = f"{endtime_filename},"
                f.write(line)
                line = ",".join(estimated)
                f.write(line + "\n")


if __name__ == "__main__":
    main()
