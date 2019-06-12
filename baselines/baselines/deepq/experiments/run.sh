#!/usr/local/bin/zsh

echo 'iteration,optimizer,learning_rate,performance'
for I in `seq 10`; do
    for O in 'adam' 'gradient_descent'; do
        for LR in '1e-4' '1e-3' '1e-2'; do
            echo -n "$I"','"$O"','"$LR"',' && ./run.py cartpole $O $LR 1 0.1
        done
    done
done
