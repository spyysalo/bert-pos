#!/usr/bin/env python3

import sys
import os
import re

import numpy as np

from itertools import groupby
from summarize import read_logs, parse_params, data_and_model


def main(argv):
    results = read_logs(argv[1:], clean=False)
    means = [(p, np.mean(v)) for p, v in results.items()]
    means.sort(key=lambda i: data_and_model(i[0]))
    for _, group in groupby(means, key=lambda i: data_and_model(i[0])):
        group = list(group)
        max_p, max_v = max(group, key=lambda i:i[1])
        min_p, min_v = min(group, key=lambda i:i[1])
        print('{}\t{}\t({} values, range {}-{})'.format(
            max_p, max_v, len(group), min_v, max_v))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
