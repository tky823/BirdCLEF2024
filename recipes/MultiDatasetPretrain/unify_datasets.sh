#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=2

data_root="../data"
dump_root="./dump"
log_dir="./log"

dump_format="birdclef2024"

preprocess="birdclef2021+2022+2023+2024"
data="birdclef2024"

. ../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
pretrain_list_path="${list_dir}/pretrain.txt"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Preprocess stage 0: Create empty list."

    :> "${pretrain_list_path}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Unify samples from in BirdCLEF2023, BirdCLEF2022, and BirdCLEF2021."

    for year in "2023" "2022" "2021"; do
        subset="train_${year}"
        list_path="${list_dir}/${subset}.txt"
        tmp_id=$(python -c "import uuid; print(str(uuid.uuid4()))")
        tmp_path="${tmp_id}.txt"

        cat "${list_path}" | sed "s/^/birdclef-${year}\//" > "${tmp_path}"
        cat "${tmp_path}" >> "${pretrain_list_path}"

        rm "${tmp_path}"
    done
fi
