#!/bin/bash

SCRIPT=$(dirname "$0")'/run.py'
TASKS_PER_FILE=20

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
DATASET='mnist'
FOLD_COUNT='10'
TRAIN_FOLDS='0:1:2:3::0:1:2:3::4:5:6:7::4:5:6:7'
TEST_FOLDS='8:9'
PHASES='12:34:12:34'
LOG_FREQUENCY='1'
CRITERIA='online'
REQUIRED_ACCURACY='0.9'
TOLERANCE='2500'
MINIMUM_STEPS='0'
HOLD_STEPS='5'
for INIT_SEED in `seq 0 9`; do
    SHUFFLE_SEED=$(((1<<16) - 1 - INIT_SEED))
    for LR in '1e-1' '1e-2' '1e-3' '1e-4' '1e-5' '1e-6'; do
        for MOMENTUM in '0.0' '0.9'; do
            OPTIMIZER='sgd'
            OUTFILE="$I"'.json'
            I=$((I + 1))
            ARGS=("--outfile=$OUTFILE"
                  "--dataset=$DATASET"
                  "--fold-count=$FOLD_COUNT"
                  "--train-folds=$TRAIN_FOLDS"
                  "--test-folds=$TEST_FOLDS"
                  "--phases=$PHASES"
                  "--test-on-all-digits"
                  "--log-frequency=$LOG_FREQUENCY"
                  "--init-seed=$INIT_SEED"
                  "--shuffle-seed=$SHUFFLE_SEED"
                  "--criteria=$CRITERIA"
                  "--required-accuracy=$REQUIRED_ACCURACY"
                  "--tolerance=$TOLERANCE"
                  "--minimum-steps=$MINIMUM_STEPS"
                  "--hold-steps=$HOLD_STEPS"
                  "--optimizer=$OPTIMIZER"
                  "--lr=$LR"
                  "--momentum=$MOMENTUM")
            echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
        done

        for RHO in '0.9'; do
            OPTIMIZER='rms'
            OUTFILE="$I"'.json'
            I=$((I + 1))
            ARGS=("--outfile=$OUTFILE"
                  "--dataset=$DATASET"
                  "--fold-count=$FOLD_COUNT"
                  "--train-folds=$TRAIN_FOLDS"
                  "--test-folds=$TEST_FOLDS"
                  "--phases=$PHASES"
                  "--test-on-all-digits"
                  "--log-frequency=$LOG_FREQUENCY"
                  "--init-seed=$INIT_SEED"
                  "--shuffle-seed=$SHUFFLE_SEED"
                  "--criteria=$CRITERIA"
                  "--required-accuracy=$REQUIRED_ACCURACY"
                  "--tolerance=$TOLERANCE"
                  "--minimum-steps=$MINIMUM_STEPS"
                  "--hold-steps=$HOLD_STEPS"
                  "--optimizer=$OPTIMIZER"
                  "--lr=$LR"
                  "--rho=$RHO")
            echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
        done

        for BETA_1 in '0.9'; do
            for BETA_2 in '0.999'; do
                OPTIMIZER='adam'
                OUTFILE="$I"'.json'
                I=$((I + 1))
                ARGS=("--outfile=$OUTFILE"
                      "--dataset=$DATASET"
                      "--fold-count=$FOLD_COUNT"
                      "--train-folds=$TRAIN_FOLDS"
                      "--test-folds=$TEST_FOLDS"
                      "--phases=$PHASES"
                      "--test-on-all-digits"
                      "--log-frequency=$LOG_FREQUENCY"
                      "--init-seed=$INIT_SEED"
                      "--shuffle-seed=$SHUFFLE_SEED"
                      "--criteria=$CRITERIA"
                      "--required-accuracy=$REQUIRED_ACCURACY"
                      "--tolerance=$TOLERANCE"
                      "--minimum-steps=$MINIMUM_STEPS"
                      "--hold-steps=$HOLD_STEPS"
                      "--optimizer=$OPTIMIZER"
                      "--lr=$LR"
                      "--beta-1=$BETA_1"
                      "--beta-2=$BETA_2")
                echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
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
