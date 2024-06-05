import os

import audyn
from omegaconf import DictConfig

from birdclef2024.utils import setup_config
from birdclef2024.utils.data import stratified_split_2022


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    list_path = config.preprocess.list_path
    csv_path = config.preprocess.csv_path
    subset = config.preprocess.subset
    train_ratio = config.preprocess.train_ratio

    assert list_path is not None, "Specify preprocess.list_path."
    assert subset is not None, "Specify preprocess.subset."
    assert csv_path is not None, "Specify preprocess.csv_path."
    assert train_ratio is not None, "Specify preprocess.train_ratio."

    train_filenames, validation_filenames = stratified_split_2022(
        csv_path,
        train_ratio=train_ratio,
        seed=config.system.seed,
    )

    if subset == "train":
        filenames = train_filenames
    elif subset == "validation":
        filenames = validation_filenames
    else:
        raise ValueError(f"{subset} is not supported.")

    with open(list_path, mode="w") as f:
        for filename in filenames:
            filename, _ = os.path.splitext(filename)
            f.write(filename + "\n")


if __name__ == "__main__":
    main()
