#!/bin/sh

# run everything
./tasks_0.sh

# amalgamate results
rm 123_results.csv 2>/dev/null
cat 123_*_results.csv | head -n 1 > 123_results.csv
for FILE in 123_*_results.csv; do
    tail -n +2 $FILE >> 123_results.csv
done
