#!/bin/sh

rm temp.csv 2>/dev/null
find . -name "123_*_results.csv" -print0 | xargs -0 cat | sort >> temp.csv
rm 123_results.csv 2>/dev/null
HEADER=$(tail -n 1 temp.csv)
echo "$HEADER" >> 123_results.csv
cat temp.csv | grep -xv $HEADER >> 123_results.csv
rm temp.csv
find . -name "123_*_results.csv" -print0 | xargs -0 rm
