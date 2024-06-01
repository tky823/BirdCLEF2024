# BirdCLEF2024

This model trains baseline model using teacher-student framework in [BirdCLEF2024](https://www.kaggle.com/competitions/birdclef-2024) challenge.

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

### Stage 0: Preprocessing

```sh
# "../data" is used by default
# On kaggle environment, "/kaggle/input"
data_root=<PATH/TO/ROOT/OF/DATA>

# "torch", "webdataset", or "birdclef2024"
dump_format="birdclef2024"

data="birdclef2024"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data-root "${data_root}" \
--dump-format "${dump_format}" \
--data "${data}"
```

**NOTE**: `${data}/feature/train`, `${data}/feature/validation`, `${data}/feature/unlabeled_train`, and `${data}/feature/unlabeled_validation` directories are empty when `dump_format=birdclef2024`.

### Stage 1: Training baseline model

Before training, set `HUGGINGFACE_TOKEN` and `HUGGINGFACE_REPO_ID` in `.env` file.

To train student model, run the following command:

```sh
tag=<TAG>

# "../data" is used by default
# On kaggle environment, "/kaggle/input"
data_root=<PATH/TO/ROOT/OF/DATA>

# "torch", "webdataset", or "birdclef2024"
dump_format="birdclef2024"

# NOTE: "train" config depends on "dump_format".
data="birdclef2024"
train="birdclef2024distillation_birdclef2024"
model="birdclef2024distillation"
optimizer="adam_student"
lr_scheduler="cos_anneal_student"
criterion="birdclef2024_distillation"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag "${tag}" \
--data-root "${data_root}" \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```
