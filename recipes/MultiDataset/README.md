# BirdCLEF2024

This recipe is based on [baseline system](https://www.kaggle.com/code/awsaf49/birdclef24-kerascv-starter-train) of [BirdCLEF2024](https://www.kaggle.com/competitions/birdclef-2024) challenge.
Unlike original implementation, this recipe also uses datasets of BirdCLEF2021-2023 in addition to one of BirdCLEF2024.

## Stages

### Stage -1: Downloading dataset

Download dataset and place it as `../data/birdclef-2024.zip`.
Then, unzip the file.

```sh
recipes/BirdCLEF2024/
|- data/
    |- birdclef-2024.zip
    |- birdclef-2024/
        |- eBird_Taxonomy_v2021.csv
        |- train_metadata.csv
        |- sample_submission.csv
        |- train_audio/
        |- test_soundscapes/
        |- unlabeled_soundscapes/
```

In addition, please download datasets of BirdCLEF2022 and BirdCLEF2023 as `../data/birdclef-2023.zip` `../data/birdclef-2022.zip`, `../data/birdclef-2021.zip`, respectively.
Then, unzip the files.

### Stage 0: Preprocessing

```sh
# "../data" is used by default
# On kaggle environment, "/kaggle/input"
data_root=<PATH/TO/ROOT/OF/DATA>

# Only dump_format="birdclef2024" is supported.
dump_format="birdclef2024"

data="birdclef2021+2022+2023+2024"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data-root "${data_root}" \
--dump-format "${dump_format}" \
--data "${data}"
```

**NOTE**: `${data}/feature/train`, `${data}/feature/validation`, `${data}/feature/unlabeled_train`, `${data}/feature/unlabeled_validation`, `${data}/feature/train_2023`, `${data}/feature/train_2022`, `${data}/feature/unlabeled_validation`, and `${data}/feature/additional_train` directories are empty when `dump_format=birdclef2024`.
