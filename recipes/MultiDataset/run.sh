#!/bin/bash

set -eu
set -o pipefail

stage=-1
stop_stage=-1

tag=""
continue_from=""
checkpoint=""
submission_path=""

exp_root="./exp"
tensorboard_root="./tensorboard"

data_root="../data"
dump_root="dump"

dump_format="birdclef2024"

system="defaults"
preprocess="birdclef2021+2022+2023+2024"
data="birdclef2021+2022+2023+2024"
train="birdclef2024baseline_birdclef2024"
test="birdclef2024baseline_birdclef2024"
model="birdclef2024baseline"
optimizer="birdclef2024baseline"
lr_scheduler="cos_anneal"
criterion="birdclef2024"

. ../_common/parse_options.sh || exit 1;

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1: Download dataset"

    echo "Please download dataset from official page and place it to ${data_root}."
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preprocess dataset"

    (
        . ./preprocess_2024.sh \
        --stage 1 \
        --stop-stage 4 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Preprocess dataset of BirdCLEF2023"

    (
        . ./preprocess_2023.sh \
        --stage 1 \
        --stop-stage 2 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Preprocess dataset of BirdCLEF2022"

    (
        . ./preprocess_2022.sh \
        --stage 1 \
        --stop-stage 2 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Preprocess dataset of BirdCLEF2021"

    (
        . ./preprocess_2021.sh \
        --stage 1 \
        --stop-stage 2 \
        --data-root "${data_root}" \
        --dump-root "${dump_root}" \
        --dump-format "${dump_format}" \
        --preprocess "${preprocess}" \
        --data "${data}"
    )
fi
