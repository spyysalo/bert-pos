#!/usr/bin/env python

import sys
import os

import numpy as np

from datetime import datetime
from collections import Counter

from keras.layers import Dense
from keras.models import Model
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras import backend as K
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint
from keras_bert import Tokenizer


CLS_TOKEN = '[CLS]'
UNK_TOKEN = '[UNK]'
SEP_TOKEN = '[SEP]'
PAD_TOKEN = '[PAD]'


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--data_dir', required=True,
                    help='Input data directory')
    ap.add_argument('--vocab_file', required=True,
                    help='Vocabulary file that BERT model was trained on')
    ap.add_argument('--bert_config_file', required=True,
                    help='Configuration for pre-trained BERT model')
    ap.add_argument('--init_checkpoint', required=True,
                    help='Initial checkpoint for pre-trained BERT model')
    ap.add_argument('--max_seq_length', type=int, default=128,
                    help='Maximum input sequence length in WordPieces')
    ap.add_argument('--do_lower_case', default=False, action='store_true',
                    help='Lower case input text (for uncased models)')
    ap.add_argument('--learning_rate', type=float, default=5e-5,
                    help='Initial learning rate')
    ap.add_argument('--train_batch_size', type=int, default=32,
                    help='Batch size for training')
    ap.add_argument('--output', default=None)    # TODO rethink
    return ap


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        if any(len(i) != len(input_ids)
               for i in (input_mask, segment_ids, label_ids)):
            raise ValueError('length mismatch')
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class EvaluationCallback(Callback):
    def __init__(self, title, X, Y):
        self.title = title
        self.X = X
        self.Y = Y

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X)
        pred = np.argmax(pred, axis=-1)
        acc = (np.ravel(pred)==np.ravel(self.Y)).mean()
        print('*'*20, self.title, acc, '*'*20)
        # TODO track best, save checkpoints


def load_pretrained(options):
    model = load_trained_model_from_checkpoint(
        options.bert_config_file,
        options.init_checkpoint,
        training=False,
        trainable=True,
        seq_len=options.max_seq_length,
    )
    vocab = load_vocabulary(options.vocab_file)
    print('vocab size', len(vocab))
    return model, vocab


def load_conll(fn, token_idx=0, tag_idx=-1, separator=None):
    sentences, current = [], []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            if l and not l.isspace():
                fields = l.split(separator)
                current.append((fields[token_idx], fields[tag_idx]))
            elif current:
                # empty line, marks end of sentence
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)
    print('loaded {} sentences from {}'.format(len(sentences), fn))
    return sentences


def load_labels(options):
    fn = os.path.join(options.data_dir, 'labels.txt')
    labels = []
    with open(fn) as f:
        for l in f:
            if l.isspace():
                continue
            labels.append(l.strip())
    print('loaded {} labels from {}: {}'.format(len(labels), fn, labels))
    return labels


def load_data(options):
    data = [load_labels(options)]
    for fn in ('train.tsv', 'dev.tsv', 'test.tsv'):
        data.append(load_conll(os.path.join(options.data_dir, fn)))
    return data


def split_sentence(sentence, options):
    sentences = []
    max_len = options.max_seq_length-2    # fit [CLS] and [SEP]
    while len(sentence) > max_len:
        # TODO: try to find a good spot to split, e.g. avoid mid-word splits
        sentences.append(sentence[:max_len])
        sentence = sentence[max_len:]
    sentences.append(sentence)
    if len(sentences) > 1:
        print('SPLIT:',
              ' /// '.join(' '.join(t for t, _ in s) for s in sentences))
    return sentences


def tokenize_sentence(sentence, tokenizer, options):
    tokenized = []
    for token, tag in sentence:
        pieces = tokenizer._tokenize(token)   # tokenize() adds [CLS] and [SEP]
        tokenized.append((pieces[0], tag))
        for piece in pieces[1:]:
            # TODO fix tag
            tokenized.append((piece, tag))
    return tokenized


def index_sentence(sentence, vocab, label_map, oov_count):
    indexed = []
    for token, tag in sentence:
        if token in vocab:
            token_id = vocab[token]
        else:
            token_id = vocab[UNK_TOKEN]
            oov_count[token] += 1
        label_id = label_map[tag]
        indexed.append((token_id, label_id))
    return indexed


def create_example(sentence, pad_token_id, pad_label_id, options):
    total_len = options.max_seq_length
    input_ids = [t_id for t_id, l_id in sentence]
    label_ids = [l_id for t_id, l_id in sentence]
    input_len, pad_len = len(input_ids), total_len-len(input_ids)
    input_ids += [pad_token_id]*pad_len
    label_ids += [pad_label_id]*pad_len
    segment_ids = [0] * total_len
    input_mask = [1]*input_len + [0]*pad_len
    return InputFeatures(input_ids, input_mask, segment_ids, label_ids)


def create_examples(sentences, tokenizer, labels, options):
    tok_sents = []
    for s in sentences:
        tok_sents.append(tokenize_sentence(s, tokenizer, options))
    print(tok_sents[0])

    split_sents = []
    for s in tok_sents:
        split_sents.extend(split_sentence(s, options))
    print(split_sents[0])

    # TODO: optional merge

    pad_label = labels[0]    # TODO clarify assumption
    wrapped_sents = []
    for s in split_sents:
        wrapped = [(CLS_TOKEN, pad_label)] + s + [(SEP_TOKEN, pad_label)]
        wrapped_sents.append(wrapped)

    vocab = tokenizer._token_dict
    label_map = { l: i for i, l in enumerate(labels) }
    oov_count = Counter()

    indexed_sents = []
    for s in wrapped_sents:
        indexed_sents.append(index_sentence(s, vocab, label_map, oov_count))

    pad_token_id = vocab[PAD_TOKEN]
    pad_label_id = label_map[pad_label]
    examples = []
    for s in indexed_sents:
        example = create_example(s, pad_token_id, pad_label_id, options)
        examples.append(example)

    for i in [0]:
        tokens = [token for token, tag in wrapped_sents[i]]
        print(tokens)
        print(examples[i].input_ids)
        print(examples[i].input_mask)
        print(examples[i].segment_ids)
        print(examples[i].label_ids)
    return examples


# Workaround for issue https://github.com/keras-team/keras/issues/11749 from
# https://github.com/keras-team/keras/issues/11749#issuecomment-498709628
def accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


def write_predictions(token_ids, pred, vocab, labels, filename):
    pred_ids = np.argmax(pred, axis=-1)
    inv_label_map = { i: l for i, l in enumerate(labels) }
    inv_vocab = { v: k for k, v in vocab.items() }
    #print('x', len(token_ids), token_ids.shape)
    #print('y', len(pred_ids), pred_ids.shape)
    total = 0
    with open(filename, 'w') as out:
        for t_ids, p_ids in zip(token_ids, pred_ids):
            for t_id, p_id in zip(t_ids, p_ids):
                token, tag = inv_vocab[t_id], inv_label_map[p_id]
                if token == PAD_TOKEN:
                    continue
                print('{}\t{}'.format(token, tag), file=out)
                total += 1
            print(file=out)
    print('saved {} predictions in {}'.format(total, filename))


def main(argv):
    args = argparser().parse_args(argv[1:])
    bert, vocab = load_pretrained(args)
    tokenizer = Tokenizer(vocab, cased=not args.do_lower_case)
    labels, train_sents, dev_sents, test_sents = load_data(args)

    train_data = create_examples(train_sents, tokenizer, labels, args)
    dev_data = create_examples(dev_sents, tokenizer, labels, args)
    test_data = create_examples(dev_sents, tokenizer, labels, args)

    output = Dense(len(labels), activation='softmax')(bert.output)
    model = Model(inputs=bert.inputs, outputs=output)
    model.summary(line_length=80)

    optimizer = Adam(lr=args.learning_rate)
    model.compile(
        loss='sparse_categorical_crossentropy',
        sample_weight_mode='temporal',
        metrics=[accuracy],
        optimizer=optimizer
    )

    train_input = np.array([e.input_ids for e in train_data])
    train_mask = np.array([e.input_mask for e in train_data])
    train_segments = np.array([e.segment_ids for e in train_data])
    train_output = np.expand_dims(np.array([e.label_ids for e in train_data]),-1)

    dev_input = np.array([e.input_ids for e in dev_data])
    dev_mask = np.array([e.input_mask for e in dev_data])
    dev_segments = np.array([e.segment_ids for e in dev_data])
    dev_output = np.expand_dims(np.array([e.label_ids for e in dev_data]),-1)

    print(len(train_input))
    print(len(train_mask))
    print(len(train_segments))
    print(len(train_output))
    print('start training at', datetime.now())
    callbacks = [
        EvaluationCallback(
            'train', [train_input, train_segments], train_output),
        EvaluationCallback(
            'dev', [dev_input, dev_segments], dev_output),
    ]
    model.fit(
        [train_input, train_segments],
        train_output,
        sample_weight=train_mask,
        batch_size=args.train_batch_size,
        epochs=1,
        verbose=1,
        callbacks=callbacks
    )
    print('done training', datetime.now())
    if args.output is not None:
        test_input = np.array([e.input_ids for e in test_data])
        test_segments = np.array([e.segment_ids for e in test_data])
        pred = model.predict(
            [test_input, test_segments],
            verbose=1
        )
        write_predictions(test_input, pred, vocab, labels, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
