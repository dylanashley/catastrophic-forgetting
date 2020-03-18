#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

import datetime
import glob
import json
import os
import sys

outfile = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat() + '.json'
assert(not os.path.isfile(outfile))
files = glob.glob('*.json')
num_files = len(files)
max_num_digits = len(str(num_files))
progress_str_template = '{0:' + str(max_num_digits) + '} / ' + str(num_files)
data = []
progress_str = progress_str_template.format(0)
sys.stdout.write('Building {}: '.format(outfile) + progress_str)
for i, infile in enumerate(files):
    import time
    with open(infile, 'r') as f:
        data.append(json.load(f))
    back_str = '\b' * len(progress_str)
    progress_str = progress_str_template.format(i + 1)
    sys.stdout.write(back_str + progress_str)
sys.stdout.write('\n')
with open(outfile, 'w') as f:
    json.dump(data, f)
