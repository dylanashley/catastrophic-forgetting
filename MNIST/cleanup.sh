#!/bin/sh

rm temp.csv 2>/dev/null
find . -name "123_*_results.csv" -print0 | xargs -0 cat | sort >> temp.csv
rm 123_results.csv 2>/dev/null
tail -n 1 temp.csv >> 123_results.csv
cat temp.csv | grep -xv 'seed,test_fold,optimizer,t1_epochs,t2_epochs,learning_rate,momentum,beta_1,beta_2,rho,dataset,stage,accuracy' >> 123_results.csv
rm temp.csv
