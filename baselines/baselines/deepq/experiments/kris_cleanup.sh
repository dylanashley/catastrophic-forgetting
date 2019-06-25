#!/bin/sh


rm temp.csv 2>/dev/null
find . -name "kris_*_results.csv" -print0 | xargs -0 cat | sort >> temp.csv
rm kris_results.csv 2>/dev/null
tail -n 1 temp.csv >> kris_results.csv
cat temp.csv | grep -xv 'seed,lr,beta_1,beta_2,buffer_size,train_freq,target_network_update_freq,exploration_fraction,mean_return' >> kris_results.csv
rm temp.csv
