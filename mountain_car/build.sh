#!/bin/bash

SCRIPT=$(dirname "$0")'/run.py'
TASKS_PER_FILE=60

# assert command line arguments valid
if [ "$#" -gt "0" ]
    then
        echo 'usage: ./build.sh'
        exit
    fi

# amalgamate all tasks
TASKS_PREFIX='tasks_'
rm "$TASKS_PREFIX"*.sh 2>/dev/null
rm tasks.sh 2>/dev/null
I=0
NUM_EPISODES=1000
for ENV_SEED in `seq 0 9`; do

    OPTIMIZER='constant'
    OUTFILE="$I"'.json'
    I=$((I + 1))
    ARGS=("--outfile=$OUTFILE"
          "--num-episodes=$NUM_EPISODES"
          "--env-seed=$ENV_SEED"
          "--optimizer=$OPTIMIZER")
    echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    NETWORK_SEED=$(((1<<16) - 1 - ENV_SEED))

    for LOSS in 'squared_error' 'TD'; do
        for LR in '1e-1' '1e-2' '1e-3' '1e-4' '1e-5' '1e-6'; do
            for MOMENTUM in '0.0' '0.9'; do
                OPTIMIZER='sgd'
                OUTFILE="$I"'.json'
                I=$((I + 1))
                ARGS=("--outfile=$OUTFILE"
                      "--num-episodes=$NUM_EPISODES"
                      "--env-seed=$ENV_SEED"
                      "--network-seed=$NETWORK_SEED"
                      "--optimizer=$OPTIMIZER"
                      "--lr=$LR"
                      "--momentum=$MOMENTUM"
                      "--loss=$LOSS")
                echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
            done

            for RHO in '0.9'; do
                OPTIMIZER='rms'
                OUTFILE="$I"'.json'
                I=$((I + 1))
                ARGS=("--outfile=$OUTFILE"
                      "--num-episodes=$NUM_EPISODES"
                      "--env-seed=$ENV_SEED"
                      "--network-seed=$NETWORK_SEED"
                      "--optimizer=$OPTIMIZER"
                      "--lr=$LR"
                      "--rho=$RHO"
                      "--loss=$LOSS")
                echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
            done

            for BETA_1 in '0.9'; do
                for BETA_2 in '0.999'; do
                    OPTIMIZER='adam'
                    OUTFILE="$I"'.json'
                    I=$((I + 1))
                    ARGS=("--outfile=$OUTFILE"
                          "--num-episodes=$NUM_EPISODES"
                          "--env-seed=$ENV_SEED"
                          "--network-seed=$NETWORK_SEED"
                          "--optimizer=$OPTIMIZER"
                          "--lr=$LR"
                          "--beta-1=$BETA_1"
                          "--beta-2=$BETA_2"
                          "--loss=$LOSS")
                    echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
                done
            done
        done
    done
done

# split tasks into files
perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' < tasks.sh > temp.sh
rm tasks.sh 2>/dev/null
split -l $TASKS_PER_FILE -a 3 temp.sh
rm temp.sh
AL=({a..z})
for i in `seq 0 25`; do
    for j in `seq 0 25`; do
        for k in `seq 0 25`; do
        FILE='x'"${AL[i]}${AL[j]}${AL[k]}"
        if [ -f $FILE ]; then
            ID=$((i * 26 * 26 + j * 26 + k))
            ID=${ID##+(0)}
            mv 'x'"${AL[i]}${AL[j]}${AL[k]}" "$TASKS_PREFIX""$ID"'.sh' 2>/dev/null
            chmod +x "$TASKS_PREFIX""$ID"'.sh' 2>/dev/null
        else
            break 3
        fi
        done
    done
done
echo $ID
