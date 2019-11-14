#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2001710
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

function on_exit {
    rm -f out-$SLURM_JOBID.tsv
    rm -f combined-$SLURM_JOBID.conllu
    rm -f jobs/$SLURM_JOBID
}
trap on_exit EXIT

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 model data_dir seq_len batch_size learning_rate epochs"
    exit 1
fi

MODEL="$1"
DATA_DIR="$2"
MAX_SEQ_LENGTH="$3"
BATCH_SIZE="$4"
LEARNING_RATE="$5"
EPOCHS="$6"

VOCAB="$(dirname "$MODEL")/vocab.txt"
CONFIG="$(dirname "$MODEL")/bert_config.json"

if [[ $MODEL =~ "uncased" ]]; then
    caseparam="--do_lower_case"
elif [[ $MODEL =~ "multilingual" ]]; then
    caseparam="--do_lower_case"
else
    caseparam=""
fi

rm -f latest.out latest.err
ln -s logs/$SLURM_JOBID.out latest.out
ln -s logs/$SLURM_JOBID.err latest.err

module purge
module load tensorflow
source $HOME/venv/keras-bert/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID: $(date)"

srun python3 train.py \
    --vocab_file "$VOCAB" \
    --bert_config_file "$CONFIG" \
    --init_checkpoint "$MODEL" \
    --data_dir "$DATA_DIR" \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --train_batch_size $BATCH_SIZE \
    --predict test \
    --output out-$SLURM_JOBID.tsv \
    $caseparam

python scripts/mergepos.py "$DATA_DIR/test.conllu" out-$SLURM_JOBID.tsv \
    > combined-$SLURM_JOBID.conllu
result=$(python scripts/conll18_ud_eval.py -v "$DATA_DIR/gold-test.conllu" \
    combined-$SLURM_JOBID.conllu \
    | egrep UPOS)

echo -n 'TEST-RESULT'$'\t'
echo -n 'init_checkpoint'$'\t'"$MODEL"$'\t'
echo -n 'data_dir'$'\t'"$DATA_DIR"$'\t'
echo -n 'max_seq_length'$'\t'"$MAX_SEQ_LENGTH"$'\t'
echo -n 'train_batch_size'$'\t'"$BATCH_SIZE"$'\t'
echo -n 'learning_rate'$'\t'"$LEARNING_RATE"$'\t'
echo -n 'num_train_epochs'$'\t'"$EPOCHS"$'\t'
echo "$result"

seff $SLURM_JOBID

echo "END $SLURM_JOBID: $(date)"
