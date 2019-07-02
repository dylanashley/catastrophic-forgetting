#!/bin/sh

HEADER='seed,test_fold,optimizer,learning_rate,momentum,beta_1,beta_2,rho,epochs,accuracies,final_accuracy,digit_predictions'
rm temp.csv 2>/dev/null
find . -name "test_*_results.csv" -print0 | xargs -0 cat | sort >> temp.csv
rm test_results.csv 2>/dev/null
tail -n 1 temp.csv >> test_results.csv
cat temp.csv | grep -xv $HEADER >> test_results.csv
rm temp.csv
