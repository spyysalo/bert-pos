#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import sys


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('gold', help='gold data')
    ap.add_argument('pred', help='predictions')
    return ap


def load_data(fn):
    data = []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            if l.isspace() or not l:
                continue
            fields = l.split()
            token, tag = fields[0], fields[-1]
            data.append((token, tag))
    return data


def main(argv):
    args = argparser().parse_args(argv[1:])
    gold = load_data(args.gold)
    pred = load_data(args.pred)
    if len(gold) != len(pred):
        raise ValueError('data length mismatch')
    correct, total = 0, 0
    for (gold_token, gold_tag), (pred_token, pred_tag) in zip(gold, pred):
        if gold_token != pred_token:
            raise ValueError('token text mismatch')
        if gold_tag == pred_tag:
            correct += 1
        total += 1
    # print('Accuracy\t{:.2%}\t({}/{})'.format(correct/total, correct, total))
    print('accuracy\t{:.2%}'.format(correct/total))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
