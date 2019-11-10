#!/usr/bin/env python3

import sys
import os
import re

import numpy as np

from collections import defaultdict, OrderedDict


model_name_map = {
    'bert-base-finnish-uncased': 'FinBERT uncased',
    'bert-base-finnish-cased': 'FinBERT cased',
    'multi_cased_L-12_H-768_A-12': 'M-BERT cased',
    'multilingual_L-12_H-768_A-12': 'M-BERT uncased',
}

result_re = re.compile(r'^\S+-RESULT\s(.*)\s(?:eval_)?accuracy\s(\S+?)\%?(\s|$)')


def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)


def model_name(model_path):
    d = os.path.basename(os.path.dirname(model_path))
    return model_name_map.get(d, d)


def parse_params(params):
    return OrderedDict([(k, v) for k, v in pairwise(params.split('\t'))])


def clean_params(params):
    cleaned = []
    for p, v in parse_params(params).items():
        # Filename cleanup
        if p == 'init_checkpoint':
            v = model_name(v)
        if p == 'data_dir':
            v = v.replace('data/', '').replace('-corpus', '')
        cleaned.extend([p,v])
    return '\t'.join(cleaned)
        

def data_and_model(params):
    parsed = parse_params(params)
    return (parsed['data_dir'], parsed['init_checkpoint'])


def read_logs(filenames, clean=True):
    missing = []
    results = defaultdict(list)
    for fn in filenames:
        found = False
        with open(fn) as f:
            for l in f:
                l = l.rstrip('\n')
                m = result_re.match(l)
                if not m:
                    continue
                if found:
                    print('multiple results in {}'.format(fn))
                params = m.group(1)
                result = float(m.group(2))
                if clean:
                    params = clean_params(params)
                results[params].append(result)
                found = True
        if not found:
            missing.append(fn)
    if missing:
        print('{} files without results'.format(len(missing)))
    return results


def main(argv):
    if len(argv) < 2:
        print('Usage: {} LOG [LOG[...]]'.format(os.path.basename(__file__)))
        return 1

    results = read_logs(argv[1:])

    # Figure out which parameters are fixed (always have the same value)
    param_values = OrderedDict()
    for params in results:
        for p, v in parse_params(params).items():
            if p not in param_values:
                param_values[p] = set()
            param_values[p].add(v)

    fixed_params = []
    for p, vals in param_values.items():
        if len(vals) == 1:
            fixed_params.append((p, vals.pop()))

    for p, v in fixed_params:
        print('{}\t{}'.format(p, v))


    for params, values in sorted(results.items(), key=lambda i: data_and_model(i[0])):
        # Don't repeat fixed parameter values
        nonfixed_params = []
        for p, v in parse_params(params).items():
            if (p, v) not in fixed_params:
                nonfixed_params.append((p, v))
        param_str = '\t'.join('{}\t{}'.format(p, v) for p, v in nonfixed_params)
        print('{}\tmean\t{}\tstd\t{}\tvalues\t{}'.format(
            param_str, np.mean(values), np.std(values), len(values)))

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
