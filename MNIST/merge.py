#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import glob
import json
import os

filename = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat() + '.json'
assert(not os.path.isfile(filename))
data = []
for filename in glob.glob('*.json'):
    with open(filename, 'r') as infile:
        data.append(json.load(infile))
with open(filename, 'w') as outfile:
    json.dump(data, outfile)
print(filename)
