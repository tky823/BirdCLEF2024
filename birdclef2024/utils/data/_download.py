import os
from typing import List

from audyn.utils import audyn_cache_dir


def download_birdclef2024_pretrain_primary_labels() -> List[str]:
    primary_labels = set()

    for year in [2021, 2022, 2023]:
        path = os.path.join(audyn_cache_dir, "data", f"birdclef{year}", "primary-labels.txt")

        with open(path) as f:
            for line in f:
                line = line.strip()
                primary_labels.add(line)

    primary_labels = sorted(list(primary_labels))

    return primary_labels
