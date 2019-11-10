#!/usr/bin/env python3

import sys
import re

from collections import defaultdict
from numpy import mean, std


results = defaultdict(list)
for fn in sys.argv[1:]:
    with open(fn) as f:
        for l in f:
            l = l.rstrip('\n')
            m = re.match(r'^(\S+)\t(\S+)\t(UPOS).*\s(\S+)\s*\|', l)
            if not m:
                continue
            model, data, metric, result = m.groups()
            results[(model, data, metric)].append(float(result))

for key, values in results.items():
    print('{}\tmean {}\tstd {}\t{} values\t{}'.format(
        '\t'.join(key), mean(values), std(values), len(values), values)) 
