#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MAX_JOBS=20

REPETITIONS=10

mkdir -p jobs

for repetition in `seq $REPETITIONS`; do
    cat "$DIR/selected-params.tsv" | while read l; do
	model=$(echo "$l" | cut -f 2)
	data_dir=$(echo "$l" | cut -f 4)
	seq_len=$(echo "$l" | cut -f 6)
	batch_size=$(echo "$l" | cut -f 8)
	learning_rate=$(echo "$l" | cut -f 10)
	epochs=$(echo "$l" | cut -f 12)
	while true; do
	    jobs=$(ls jobs | wc -l)
	    if [ $jobs -lt $MAX_JOBS ]; then break; fi
	    echo "Too many jobs ($jobs), sleeping ..."
	    sleep 60
	done
	echo "Submitting job with params $model $data_dir $seq_len $batch_size $learning_rate $epochs"
	job_id=$(sbatch "$DIR/test-pos.sh" \
	    $model \
	    $data_dir \
	    $seq_len \
	    $batch_size \
	    $learning_rate \
	    $epochs \
	    | perl -pe 's/Submitted batch job //'
	)
	echo "Submitted batch job $job_id"
	touch jobs/$job_id
	sleep 10
    done
done
