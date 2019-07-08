#!/bin/sh

rm temp.csv 2>/dev/null
find . -name "test_*_results.csv" -print0 | xargs -0 cat | sort >> temp.csv
rm test_results.csv 2>/dev/null
HEADER=$(tail -n 1 temp.csv)
echo "$HEADER" >> test_results.csv
cat temp.csv | grep -xv $HEADER >> test_results.csv
rm temp.csv
find . -name "test_*_results.csv" -print0 | xargs -0 rm
