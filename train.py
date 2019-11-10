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
from keras_bert import AdamWarmup, calc_train_steps


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
    ap.add_argument('--num_train_epochs', type=int, default=1,
                    help='Number of training epochs')
    ap.add_argument('--continuation_label', default=None,
                    help='Label to assign to continuation word pieces')
    ap.add_argument('--predict', default=None, choices=['dev', 'test'])
    ap.add_argument('--output', default=None)    # TODO rethink
    return ap


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids,
                 head_flags):
        if any(len(i) != len(input_ids)
               for i in (input_mask, segment_ids, label_ids)):
            raise ValueError('length mismatch')
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.head_flags = head_flags


class EvaluationCallback(Callback):
    def __init__(self, title, input_ids, input_segments, Y, output_mask):
        self.title = title
        self.input_ids = input_ids
        self.input_segments = input_segments
        self.X = [input_ids, input_segments]
        self.Y = Y
        self.flat_input = np.ravel(input_ids)
        self.flat_Y = np.ravel(Y)
        self.flat_mask = np.ravel(output_mask)
        self.total_unmasked = np.sum(self.flat_mask)
        self.best, self.best_epoch = None, None

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X)
        pred = np.argmax(pred, axis=-1)
        flat_pred = np.ravel(pred)
        correct = flat_pred == self.flat_Y
        correct_unmasked = correct * self.flat_mask
        acc = np.sum(correct_unmasked)/self.total_unmasked
        if self.best is None or acc > self.best:
            self.best = acc
            self.best_epoch = epoch
        print('#'*20, self.title, acc, '#'*20)


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
    if (options.continuation_label is not None and
        options.continuation_label not in labels):
        labels = [options.continuation_label] + labels
        print('added continuation label {}'.format(options.continuation_label))
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
              ' /// '.join(' '.join(t[0] for t in s) for s in sentences))
    return sentences


def tokenize_sentence(sentence, tokenizer, options):
    tokenized = []
    for token, tag in sentence:
        pieces = tokenizer._tokenize(token)   # tokenize() adds [CLS] and [SEP]
        tokenized.append((pieces[0], tag, 1))
        for piece in pieces[1:]:
            if options.continuation_label is None:
                tokenized.append((piece, tag, 0))    # copy from head
            else:
                tokenized.append((piece, options.continuation_label, 0))
    return tokenized


def index_sentence(sentence, vocab, label_map, oov_count):
    indexed = []
    for token, tag, is_head in sentence:
        if token in vocab:
            token_id = vocab[token]
        else:
            token_id = vocab[UNK_TOKEN]
            oov_count[token] += 1
        label_id = label_map[tag]
        indexed.append((token_id, label_id, is_head))
    return indexed


def create_example(sentence, input_mask, pad_token_id, pad_label_id, options):
    total_len = options.max_seq_length
    input_ids = [t_id for t_id, l_id, is_head in sentence]
    label_ids = [l_id for t_id, l_id, is_head in sentence]
    head_flags = [is_head for t_id, l_id, is_head in sentence]
    input_len, pad_len = len(input_ids), total_len-len(input_ids)
    input_ids += [pad_token_id]*pad_len
    label_ids += [pad_label_id]*pad_len
    head_flags += [0] * pad_len
    input_mask += [0] * pad_len
    segment_ids = [0] * total_len
    return InputFeatures(input_ids, input_mask, segment_ids, label_ids,
                         head_flags)


def create_examples(sentences, tokenizer, labels, options):
    tok_sents = []
    for s in sentences:
        tok_sents.append(tokenize_sentence(s, tokenizer, options))
    print(tok_sents[0])

    split_sents = []
    for s in tok_sents:
        split_sents.extend(split_sentence(s, options))
    print(split_sents[0])

    pad_label = labels[0]    # TODO clarify assumption
    wrapped_sents = []
    for s in split_sents:
        wrapped = [(CLS_TOKEN, pad_label, 0)]+s+[(SEP_TOKEN, pad_label, 0)]
        wrapped_sents.append(wrapped)

    input_masks = []
    for s in wrapped_sents:
        mask = [t[2] for t in s]    # head
        input_masks.append(mask)
    print(input_masks[0])

    vocab = tokenizer._token_dict
    label_map = { l: i for i, l in enumerate(labels) }
    oov_count = Counter()

    indexed_sents = []
    for s in wrapped_sents:
        indexed_sents.append(index_sentence(s, vocab, label_map, oov_count))

    pad_token_id = vocab[PAD_TOKEN]
    pad_label_id = label_map[pad_label]
    examples = []
    for s, m in zip(indexed_sents, input_masks):
        example = create_example(s, m, pad_token_id, pad_label_id, options)
        examples.append(example)

    for i in [0]:
        tokens = [token for token, tag, is_head in wrapped_sents[i]]
        tags = [tag for token, tag, is_head in wrapped_sents[i]]
        head_flags = [is_head for token, tag, is_head in wrapped_sents[i]]
        print('tokens     ', i, tokens)
        print('tags       ', i, tags)
        print('heads      ', i, head_flags)
        print('input_ids  ', i, examples[i].input_ids)
        print('input_mask ', i, examples[i].input_mask)
        print('segment_ids', i, examples[i].segment_ids)
        print('label_ids  ', i, examples[i].label_ids)
        print('head_flags ', i, examples[i].head_flags)
    return examples


def flatten(list_of_lists):
    return [i for l in list_of_lists for i in l]


def write_predictions(sentences, token_ids, head_flags, pred, vocab, labels,
                      filename):
    pred_ids = np.argmax(pred, axis=-1)
    inv_label_map = { i: l for i, l in enumerate(labels) }
    inv_vocab = { v: k for k, v in vocab.items() }
    token_ids = flatten(token_ids)    
    pred_ids = flatten(pred_ids)
    head_flags = flatten(head_flags)
    if (len(token_ids) != len(pred_ids) or
        len(token_ids) != len(head_flags)):
        raise ValueError('length mismatch')

    parts, preds = [], []
    for t_id, p_id, is_head in zip(token_ids, pred_ids, head_flags):
        token, tag = inv_vocab[t_id], inv_label_map[p_id]
        if token in (CLS_TOKEN, PAD_TOKEN, SEP_TOKEN):
            continue
        if is_head:
            parts.append([])
            preds.append([])
        if token.startswith('##'):
            token = token[2:]
        parts[-1].append(token)
        preds[-1].append(tag)

    pred_idx = 0
    if filename is None:
        filename = '/dev/stdout'    # TODO
    with open(filename, 'w') as out:
        for sentence in sentences:
            for token in sentence:
                tag = preds[pred_idx][0]
                print('{}\t{}'.format(token, tag), file=out)
                pred_idx += 1
            print(file=out)
    print('saved {} tokens in {}'.format(pred_idx, filename))


def main(argv):
    args = argparser().parse_args(argv[1:])
    bert, vocab = load_pretrained(args)
    tokenizer = Tokenizer(vocab, cased=not args.do_lower_case)
    labels, train_sents, dev_sents, test_sents = load_data(args)

    train_data = create_examples(train_sents, tokenizer, labels, args)
    dev_data = create_examples(dev_sents, tokenizer, labels, args)
    test_data = create_examples(test_sents, tokenizer, labels, args)

    output = Dense(len(labels), activation='softmax')(bert.output)
    model = Model(inputs=bert.inputs, outputs=output)
    model.summary(line_length=80)

    train_input = np.array([e.input_ids for e in train_data])
    train_in_mask = np.array([e.input_mask for e in train_data])
    train_segments = np.array([e.segment_ids for e in train_data])
    train_output = np.expand_dims(
        np.array([e.label_ids for e in train_data]), -1)
    train_head_flags = np.array([e.head_flags for e in train_data])

    total_steps, warmup_steps = calc_train_steps(
        num_example=len(train_input),
        batch_size=args.train_batch_size,
        epochs=args.num_train_epochs,
        warmup_proportion=0.1,
    )

    optimizer = AdamWarmup(
        total_steps,
        warmup_steps,
        lr=args.learning_rate,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo'],
        min_lr=0    # TODO
    )

    model.compile(
        loss='sparse_categorical_crossentropy',
        sample_weight_mode='temporal',
        optimizer=optimizer
    )

    dev_input = np.array([e.input_ids for e in dev_data])
    dev_in_mask = np.array([e.input_mask for e in dev_data])
    dev_segments = np.array([e.segment_ids for e in dev_data])
    dev_output = np.expand_dims(np.array([e.label_ids for e in dev_data]),-1)
    dev_head_flags = np.array([e.head_flags for e in dev_data])

    train_start = datetime.now()
    print('start training at', train_start)
    train_cb = EvaluationCallback(
        'train', train_input, train_segments, train_output, train_head_flags)
    dev_cb = EvaluationCallback(
        'dev', dev_input, dev_segments, dev_output, dev_head_flags)
    callbacks = [train_cb, dev_cb]
    model.fit(
        [train_input, train_segments],
        train_output,
        sample_weight=train_in_mask,
        batch_size=args.train_batch_size,
        epochs=args.num_train_epochs,
        verbose=1,
        callbacks=callbacks
    )
    train_end = datetime.now()
    print('done training', train_end, 'time', train_end-train_start)

    if args.predict is not None:
        if args.predict == 'dev':
            pred_data, pred_sents = dev_data, dev_sents
        else:
            assert args.predict == 'test'
            pred_data, pred_sents = test_data, test_sents
        pred_input = np.array([e.input_ids for e in pred_data])
        pred_segments = np.array([e.segment_ids for e in pred_data])
        pred = model.predict(
            [pred_input, pred_segments],
            verbose=1
        )
        pred_tokens = [[t for t, _ in s] for s in pred_sents]
        pred_head_flags = np.array([e.head_flags for e in pred_data])
        write_predictions(pred_tokens, pred_input, pred_head_flags,
                          pred, vocab, labels, args.output)
    print('best dev result', dev_cb.best, 'for epoch', dev_cb.best_epoch)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
