#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import glob
import json
import os

outfile = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat() + '.json'
assert(not os.path.isfile(outfile))
data = []
for infile in glob.glob('*.json'):
    with open(infile, 'r') as f:
        data.append(json.load(f))
with open(outfile, 'w') as f:
    json.dump(data, f)
print(outfile)
