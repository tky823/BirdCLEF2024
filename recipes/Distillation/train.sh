#!/bin/bash

data_root="../data"
dump_root="./dump"
exp_root="./exp"
tensorboard_root="./tensorboard"

tag=""
continue_from=""

dump_format="birdclef2024"

system="defaults"
preprocess="birdclef2024"
data="birdclef2024"
train="birdclef2024baseline_birdclef2024"
model="birdclef2024baseline"
optimizer="birdclef2024baseline"
lr_scheduler="cos_anneal"
criterion="birdclef2024"

. ../_common/parse_options.sh || exit 1;

dump_dir="${dump_root}/${data}"
list_dir="${dump_dir}/list"
feature_dir="${dump_dir}/feature"

if [ -z "${tag}" ]; then
    tag=$(date +"%Y%m%d-%H%M%S")
fi

exp_dir="${exp_root}/${tag}"
tensorboard_dir="${tensorboard_root}/${tag}"

cmd=$(
    python ../_common/parse_run_command.py \
    --config-dir "./conf" \
    hydra.run.dir="./log/$(date +"%Y%m%d-%H%M%S")" \
    system="${system}"
)

train_labeled_list_path="${list_dir}/train.txt"
train_unlabeled_list_path="${list_dir}/unlabeled_train.txt"
validation_labeled_list_path="${list_dir}/validation.txt"
validation_unlabeled_list_path="${list_dir}/unlabeled_validation.txt"

if [ "${dump_format}" = "birdclef2024" ]; then
    train_feature_dir="${data_root}/birdclef-2024"
    validation_feature_dir="${data_root}/birdclef-2024"
else
    train_feature_dir="${feature_dir}/train"
    validation_feature_dir="${feature_dir}/validation"
fi

${cmd} ./local/train.py \
--config-dir "./conf" \
hydra.run.dir="${exp_dir}/log/$(date +"%Y%m%d-%H%M%S")" \
system="${system}" \
preprocess="${preprocess}" \
data="${data}" \
train="${train}" \
model="${model}" \
optimizer="${optimizer}" \
lr_scheduler="${lr_scheduler}" \
criterion="${criterion}" \
preprocess.dump_format="${dump_format}" \
train.dataset.train.labeled_list_path="${train_labeled_list_path}" \
train.dataset.train.unlabeled_list_path="${train_unlabeled_list_path}" \
train.dataset.train.feature_dir="${train_feature_dir}" \
train.dataset.validation.labeled_list_path="${validation_labeled_list_path}" \
train.dataset.validation.unlabeled_list_path="${validation_unlabeled_list_path}" \
train.dataset.validation.feature_dir="${validation_feature_dir}" \
train.resume.continue_from="${continue_from}" \
train.output.exp_dir="${exp_dir}" \
train.output.tensorboard_dir="${tensorboard_dir}"
