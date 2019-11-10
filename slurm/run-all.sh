#!/bin/bash

mkdir -p logs

for repetition in `seq 10`; do
    for model in \
	"models/multi_cased_L-12_H-768_A-12/bert_model.ckpt" \
	"models/multilingual_L-12_H-768_A-12/bert_model.ckpt" \
	"models/bert-base-finnish-cased/bert-base-finnish-cased" \
	"models/bert-base-finnish-uncased/bert-base-finnish-uncased"; do
	for datadir in data/{tdt,ftb,pud}; do
	    sbatch slurm/pos.sh $model $datadir
	    sleep 200
	done
    done
done
