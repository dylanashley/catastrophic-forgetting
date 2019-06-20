#!/bin/sh

# run everything
rm results.csv
for SEED in `seq 1 11`; do
for LR in '1e-1' '1e-2' '1e-3' '1e-4'; do
    ./run.py mountain_car $SEED sgd $LR 50000 0.1
    for BETA_1 in '0.75' '0.9' '0.999'; do
    for BETA_2 in '0.9' '0.999' '0.99999'; do
        ./run.py mountain_car $SEED adam $LR 50000 0.1 $BETA_1 $BETA_2
    done
    done
done
done

# clean output
head -n 1 results.csv > temp.csv
sort -u results.csv | sed '$ d' >> temp.csv
rm results.csv
mv temp.csv results.csv
