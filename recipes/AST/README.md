# BirdCLEF2024

Finetune audio spectrogram transformer using BirdCLEF2024 dataset.

## Stages

### Stages -1 ~ 0: Downloading dataset

See `BirdCLEF2024/recipes/Baseline/README.md`.
Note that, you have to set `data="birdclef2024ast"` unlike baseline recipe.

### Stage 1: Training AST

Before training, set `HUGGINGFACE_TOKEN` and `HUGGINGFACE_REPO_ID` in `.env` file.

To train AST, run the following command:

```sh
tag=<TAG>

# "../data" is used by default
# On kaggle environment, "/kaggle/input"
data_root=<PATH/TO/ROOT/OF/DATA>

# "torch", "webdataset", or "birdclef2024"
dump_format="birdclef2024"

# NOTE: "train" config depends on "dump_format".
data="birdclef2024ast"
train="birdclef2024baseline_birdclef2024"  # or "birdclef2024baseline_weighted_birdclef2024"
model="birdclef2024ast"
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

**NOTE**: On kaggle, `batch_size=64` defined in `conf/train` may cause OOM error.

### Stage 2: Preprocessing test dataset

```sh
# "../data" is used by default
# On kaggle environment, "/kaggle/input"
data_root=<PATH/TO/ROOT/OF/DATA>

# "torch", "webdataset", or "birdclef2024"
dump_format="birdclef2024"

# NOTE: "train" config depends on "dump_format".
data="birdclef2024ast"
train="birdclef2024baseline_birdclef2024"  # or "birdclef2024baseline_weighted_birdclef2024"
test="birdclef2024ast_birdclef2024"  # or "birdclef2024ast_shared_birdclef2024"

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--data-root "${data_root}" \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--test "${test}"
```

### Stage 3: Inference by AST

To infer by AST, run the following command:

```sh
tag=<TAG>

checkpoint=<PATH/TO/TRAINED/MODEL>

# "../data" is used by default
# On kaggle environment, "/kaggle/input"
data_root=<PATH/TO/ROOT/OF/DATA>

# "torch", "webdataset", or "birdclef2024"
dump_format="birdclef2024"

# NOTE: "train" and "test" configs depend on "dump_format".
data="birdclef2024ast"
train="birdclef2024baseline_birdclef2024"  # or "birdclef2024baseline_weighted_birdclef2024"
test="birdclef2024ast_birdclef2024"  # or "birdclef2024ast_shared_birdclef2024"
model="birdclef2024ast"

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
data="birdclef2024ast"
train="birdclef2024baseline_birdclef2024"  # or "birdclef2024baseline_birdclef2024_weighted"
test="birdclef2024ast_birdclef2024"  # or "birdclef2024ast_shared_birdclef2024"
model="birdclef2024ast"

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
