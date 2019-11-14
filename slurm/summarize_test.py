#!/usr/bin/env python3

import sys
import os
import re

import numpy as np

from collections import defaultdict, OrderedDict

from summarize import read_logs, data_and_model, parse_params


model_name_map = {
    'bert-base-finnish-uncased': 'FinBERT uncased',
    'bert-base-finnish-cased': 'FinBERT cased',
    'multi_cased_L-12_H-768_A-12': 'M-BERT cased',
    'multilingual_L-12_H-768_A-12': 'M-BERT uncased',
}

result_re = re.compile(r'^TEST-RESULT\s(.*)\sUPOS(?:\s+\|\s+\S+){2}\s+\|\s+(\S+?)\s+\|\s+\S+\s*$')


def main(argv):
    if len(argv) < 2:
        print('Usage: {} LOG [LOG[...]]'.format(os.path.basename(__file__)))
        return 1

    results = read_logs(argv[1:], clean=True, regex=result_re)

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
