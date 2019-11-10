#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p gpu
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2001710
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 MODEL DATA"
    exit 1
fi

model="$1"
datadir="$2"

modeldir=$(dirname "$model")

if [[ $model =~ "uncased" ]]; then
    caseparam="--do_lower_case"
elif [[ $model =~ "multilingual" ]]; then
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

echo "START: $(date)"

srun python3 train.py \
    --vocab_file "$modeldir/vocab.txt" \
    --bert_config_file "$modeldir/bert_config.json" \
    --init_checkpoint "$model" \
    --data_dir "$datadir" \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --predict test \
    --output out-$SLURM_JOBID.tsv \
    $caseparam

python scripts/mergepos.py "$datadir/test.conllu" out-$SLURM_JOBID.tsv \
    > combined-$SLURM_JOBID.conllu
python scripts/conll18_ud_eval.py -v "$datadir/gold-test.conllu" \
    combined-$SLURM_JOBID.conllu \
    | perl -pe 's/^/'"$(basename $modeldir)"'\t'"$(basename $datadir)"'\t/'

rm out-$SLURM_JOBID.tsv
rm combined-$SLURM_JOBID.conllu

seff $SLURM_JOBID

echo "END: $(date)"
