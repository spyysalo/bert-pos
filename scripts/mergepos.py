#!/usr/bin/env python3

# Merge POS tags from TSV into CoNLL-U data.

import sys
import re

from logging import warning


# https://universaldependencies.org/format
CONLLU_FIELDS = [
    'ID',
    'FORM',
    'LEMMA',
    'UPOS',
    'XPOS',
    'FEATS',
    'HEAD',
    'DEPREL',
    'DEPS',
    'MISC',
]

FIELD_IDX = { k: i for i, k in enumerate(CONLLU_FIELDS) }


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-m', '--max-missing', default=0, type=int,
                    help='Maximum missing tokens in tag data (default 0)')
    ap.add_argument('conllu')
    ap.add_argument('tags')
    return ap


def read_tagged(fn):
    tagged_tokens = []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            if not l or l.isspace():
                continue
            fields = l.split('\t')
            token, tag = fields[0], fields[-1]
            tagged_tokens.append((token, tag))
    return tagged_tokens


def is_token_line(l):
    return re.match(r'^[0-9]+\t', l) is not None


def merge_tags(fn, tagged_tokens, options):
    form_idx, upos_idx = FIELD_IDX['FORM'], FIELD_IDX['UPOS']
    tok_idx, missing_count = 0, 0
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            if tok_idx < len(tagged_tokens):
                tok, upos = tagged_tokens[tok_idx]
            else:
                tok, upos = None, None
            l = l.rstrip('\n')
            if is_token_line(l):
                fields = l.split('\t')
                form = fields[form_idx]
                if form.startswith(tok) or tok == '[UNK]':
                    fields[upos_idx] = upos
                    tok_idx += 1
                elif tok is None:
                    warning('out of tags for "{}" on line {}'.format(
                        form, ln))
                    missing_count += 1
                else:
                    warning('missing tag for "{}" (current "{}") on line {}'.\
                            format(form, tok, ln))
                    missing_count += 1
                if missing_count > options.max_missing:
                    raise ValueError('Exceeded --max-missing')
                l = '\t'.join(fields)
            print(l)

                
def main(argv):
    args = argparser().parse_args(argv[1:])
    tagged_tokens = read_tagged(args.tags)
    merge_tags(args.conllu, tagged_tokens, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
