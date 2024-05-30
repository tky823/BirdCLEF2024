#!/bin/bash

stage=1
stop_stage=4

data_root="../data"
dump_root="dump"

dump_format="birdclef2024"

preprocess="birdclef2021+2022+2023+2024"
data="birdclef2021+2022+2023+2024"

. ../_common/parse_options.sh || exit 1;

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preprocessing stage 1: Preprocess dataset"

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

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preprocessing stage 2: Preprocess dataset of BirdCLEF2023"

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

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Preprocessing stage 3: Preprocess dataset of BirdCLEF2022"

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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Preprocessing stage 4: Preprocess dataset of BirdCLEF2021"

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
