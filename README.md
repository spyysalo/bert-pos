# BERT POS

Part-of-speech tagging using BERT

## Quickstart

Download BERT models

```
./scripts/getmodels.sh
```

Experiment with FinBERT cased and TDT data

```
MODELDIR="models/bert-base-finnish-cased"
DATADIR="data/tdt"

python3 train.py \
    --vocab_file "$MODELDIR/vocab.txt" \
    --bert_config_file "$MODELDIR/bert_config.json" \
    --init_checkpoint "$MODELDIR/bert-base-finnish-cased" \
    --data_dir "$DATADIR" \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output pred.tsv

python scripts/mergepos.py "$DATADIR/test.conllu" pred.tsv > pred.conllu
python scripts/conll18_ud_eval.py -v "$DATADIR/gold-test.conllu" pred.conllu
```

## CoNLL'18 UD data

Manually annotated data

(A small part of this data is found in `data/ud-treebanks-v2.2/`)

```
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2837/ud-treebanks-v2.2.tgz

tar xvzf ud-treebanks-v2.2.tgz
```

Predictions from CoNLL'18 participants

(A small part of this data is found in `data/official-submissions/`)

```
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2885/conll2018-test-runs.tgz

tar xvzf conll2018-test-runs.tgz
```

Evaluation script

```
wget https://universaldependencies.org/conll18/conll18_ud_eval.py \
    -O scripts/conll18_ud_eval.py
```

## Reformat

Gold data

```
for t in tdt ftb pud; do
    mkdir data/$t
    for f in data/ud-treebanks-v2.2/*/fi_${t}-ud-*.conllu; do
        s=$(echo "$f" | perl -pe 's/.*\/.*-ud-(.*)\.conllu/$1/')
	egrep '^([0-9]+'$'\t''|[[:space:]]*$)' $f | cut -f 2,4 \
            > data/$t/$s.tsv
    done
    cut -f 2 data/$t/test.tsv | egrep -v '^[[:space:]]*$' | sort | uniq \
        > data/$t/labels.txt
    mv data/$t/test.tsv data/$t/gold-test.tsv
    cp data/ud-treebanks-v2.2/*/fi_${t}-ud-test.conllu data/$t/gold-test.conllu
done
```

PUD doesn't have train and dev, use TDT

```
for s in train dev; do
    cp data/tdt/$s.tsv data/pud
done
```

Test data with predicted tokens

```
for t in tdt ftb pud; do
    cp data/official-submissions/Uppsala-18/fi_$t.conllu data/$t/test.conllu
    egrep '^([0-9]+'$'\t''|[[:space:]]*$)' data/$t/test.conllu \
        | cut -f 2 | perl -pe 's/(\S+)$/$1\tX/' > data/$t/test.tsv
done
```

## Reference results

Best UPOS result for each Finnish treebank in CoNLL'18
from https://universaldependencies.org/conll18/results-upos.html

```
fi_ftb: 1. HIT-SCIR (Harbin): 96.70
fi_pud: 1. LATTICE (Paris)  : 97.65
fi_tdt: 1. HIT-SCIR (Harbin): 97.30
```

## BERT model comparison for Finnish POS tagging

The scripts run here are specific to a particular Slurm system configuration.
You will need to edit them to match your setup if you want to rerun this.

```
./slurm/run-parameter-selection.sh
python3 slurm/select_params.py logs/*.out | cut -f 1-12 > slurm/selected-params.tsv
./slurm/run-selected-params.sh
python3 slurm/summarize_test.py logs/*.out | cut -f 2,4,11-14 > results.tsv
```

This should give approximately the following results:

```
Model             Corpus Mean
FinBERT cased     FTB    98.39
FinBERT uncased   FTB    98.28
M-BERT  cased     FTB    95.87
M-BERT  uncased   FTB    96.00
FinBERT cased     PUD    98.08
FinBERT uncased   PUD    97.94
M-BERT  cased     PUD    97.58
M-BERT  uncased   PUD    97.48
FinBERT cased     TDT    98.23
FinBERT uncased   TDT    98.12
M-BERT  cased     TDT    96.97
M-BERT  uncased   TDT    96.59
```
