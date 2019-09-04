#!/bin/sh

rm results.json 2>/dev/null
python -c 'import glob, json; print(json.dumps([json.load(open(f, "r")) for f in glob.glob("*_*.json")]))' > results.json && find . -name "*_*.json" -print0 | xargs -0 rm
