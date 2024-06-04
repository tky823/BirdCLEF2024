#!/bin/bash

set -eu
set -o pipefail

stage=0
stop_stage=0

data_root="../data"
dump_root="./dump"
log_dir="./log"

dump_format="birdclef2024"

preprocess="birdclef2021+2022+2023+2024"
data="birdclef2024"

. ../_common/parse_options.sh || exit 1;

birdclef2022_dataroot="${data_root}/birdclef-2022"
csv_path="${birdclef2022_dataroot}/train_metadata.csv"
audio_root="${birdclef2022_dataroot}/train_audio"

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocess stage 1: Select samples unseen in BirdCLEF2024."

    mkdir -p "${list_dir}"

    subset="train_2021"
    list_path="${list_dir}/${subset}.txt"
    existing_list_path="${list_dir}/additional_train.txt"
    birdclef2024_train_list_path="${list_dir}/train.txt"
    birdclef2024_validation_list_path="${list_dir}/validation.txt"

    python ./local/select_samples.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${list_path}" \
    preprocess.csv_path="${csv_path}" \
    preprocess.audio_root="${audio_root}" \
    preprocess.subset="${subset}" \
    preprocess.existing_list_path="${existing_list_path}" \
    preprocess.birdclef2024_train_list_path="${birdclef2024_train_list_path}" \
    preprocess.birdclef2024_validation_list_path="${birdclef2024_validation_list_path}"

    cat "${list_path}" >> "${existing_list_path}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocess stage 2: Save features"

    subset="train_2021"
    subset_list_path="${list_dir}/${subset}.txt"
    subset_feature_dir="${feature_dir}/${subset}"

    python ../_common/local/save_features.py \
    --config-dir "./conf" \
    hydra.run.dir="${log_dir}/$(date +"%Y%m%d-%H%M%S")" \
    preprocess="${preprocess}" \
    data="${data}" \
    preprocess.dump_format="${dump_format}" \
    preprocess.list_path="${subset_list_path}" \
    preprocess.feature_dir="${subset_feature_dir}" \
    preprocess.csv_path="${csv_path}" \
    preprocess.audio_root="${audio_root}" \
    preprocess.subset="${subset}"
fi
