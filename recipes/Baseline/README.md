# BirdCLEF2024

[Baseline recipe](https://www.kaggle.com/code/awsaf49/birdclef24-kerascv-starter-train) of [BirdCLEF2024](https://www.kaggle.com/competitions/birdclef-2024) challenge.

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

# NOTE: "train" config depends on "dump_format".
data="birdclef2024"
train="birdclef2024baseline_birdclef2024"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data-root "${data_root}" \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}"
```

**NOTE**: `${data}/feature/train`, `${data}/feature/validation`, `${data}/feature/unlabeled_train`, and `${data}/feature/unlabeled_validation` directories are empty when `dump_format=birdclef2024`.

### Stage 1: Training baseline model

Before training, set `HUGGINGFACE_TOKEN` and `HUGGINGFACE_REPO_ID` in `.env` file.

To train baseline model, run the following command:

```sh
tag=<TAG>

# "../data" is used by default
# On kaggle environment, "/kaggle/input"
data_root=<PATH/TO/ROOT/OF/DATA>

# "torch", "webdataset", or "birdclef2024"
dump_format="birdclef2024"

# NOTE: "train" config depends on "dump_format".
data="birdclef2024"
train="birdclef2024baseline_birdclef2024"
model="birdclef2024baseline"
optimizer="birdclef2024baseline"
lr_scheduler="cos_anneal"
criterion="birdclef2024"

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

### Stage 2: Preprocessing test dataset

```sh
# "../data" is used by default
# On kaggle environment, "/kaggle/input"
data_root=<PATH/TO/ROOT/OF/DATA>

# "torch", "webdataset", or "birdclef2024"
dump_format="birdclef2024"

# NOTE: "train" config depends on "dump_format".
data="birdclef2024"
train="birdclef2024baseline_birdclef2024"
test="birdclef2024baseline_birdclef2024"

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--data-root "${data_root}" \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--test "${test}"
```

### Stage 3: Inference by baseline model

To infer by baseline model, run the following command:

```sh
tag=<TAG>

checkpoint=<PATH/TO/TRAINED/MODEL>

# "../data" is used by default
# On kaggle environment, "/kaggle/input"
data_root=<PATH/TO/ROOT/OF/DATA>

# "torch", "webdataset", or "birdclef2024"
dump_format="birdclef2024"

# NOTE: "train" and "test" configs depend on "dump_format".
data="birdclef2024"
train="birdclef2024baseline_birdclef2024"
test="birdclef2024baseline_birdclef2024"
model="birdclef2024baseline"

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--tag "${tag}" \
--checkpoint "${checkpoint}" \
--data-root "${data_root}" \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--test "${test}" \
--model "${model}"
```

### Stage 4: Submit estimation

To submit estimation, run the following command:

```sh
# Tag is required in this stage.
tag=<TAG>

submission_path=<PATH/TO/SAVE/SUBMISSION.CSV>

# "torch", "webdataset", or "birdclef2024"
dump_format="birdclef2024"

# NOTE: "train" and "test" configs depend on "dump_format".
data="birdclef2024"
train="birdclef2024baseline_birdclef2024"
test="birdclef2024baseline_birdclef2024"
model="birdclef2024baseline"

. ./run.sh \
--stage 4 \
--stop-stage 4 \
--tag "${tag}" \
--submission-path "${submission_path}" \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--test "${test}" \
--model "${model}"
```
