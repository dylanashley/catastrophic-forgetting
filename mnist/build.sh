#!/bin/bash

SCRIPT=$(dirname "$0")'/run.py'
TASKS_PER_FILE=100

LEARNING_RATES=('0.125'
                '0.08838834764831845'
                '0.0625'
                '0.04419417382415922'
                '0.03125'
                '0.02209708691207961'
                '0.015625'
                '0.011048543456039806'
                '0.0078125'
                '0.005524271728019903'
                '0.00390625'
                '0.0027621358640099515'
                '0.001953125'
                '0.0013810679320049757'
                '0.0009765625'
                '0.0006905339660024879'
                '0.00048828125'
                '0.00034526698300124393'
                '0.000244140625'
                '0.00017263349150062197'
                '0.0001220703125'
                '8.631674575031098e-05'
                '6.103515625e-05'
                '4.315837287515549e-05'
                '3.0517578125e-05'
                '2.1579186437577746e-05'
                '1.52587890625e-05'
                '1.0789593218788873e-05'
                '7.62939453125e-06'
                '5.3947966093944364e-06'
                '3.814697265625e-06')

# assert command line arguments valid
if [ "$#" -gt "0" ]
    then
        echo 'usage: ./build.sh'
        exit
    fi

# begin amalgamating all tasks
TASKS_PREFIX='tasks_'
rm "$TASKS_PREFIX"*.sh 2>/dev/null
rm tasks.sh 2>/dev/null
I=0

# set shared parameters for all experiments
CRITERIA='online'
DATASET='mnist'
FOLD_COUNT='10'
HOLD_STEPS='5'
LOG_FREQUENCY='1'
MINIMUM_STEPS='0'
REQUIRED_ACCURACY='0.9'
TOLERANCE='2500'

##### Experiment 1 Train ######################################################

for INIT_SEED in `seq 0 49`; do
    SHUFFLE_SEED=$(((1<<16) - 1 - INIT_SEED))

    for LR in "${LEARNING_RATES[@]}"; do
        TRAIN_FOLDS='0:1'
        TEST_FOLDS='2:3'
        PHASES='1234:12:34'
        OPTIMIZER='sgd'
        MOMENTUM='0.0'
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
        echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

        TRAIN_FOLDS='0:1'
        TEST_FOLDS='2:3'
        PHASES='12:34'
        OPTIMIZER='sgd'
        MOMENTUM='0.0'
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
        echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

        TRAIN_FOLDS='0:1'
        TEST_FOLDS='2:3'
        PHASES='34:12'
        OPTIMIZER='sgd'
        MOMENTUM='0.0'
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
        echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
    done
done

##### Experiment 1 Test #######################################################

for INIT_SEED in `seq 50 549`; do
    SHUFFLE_SEED=$(((1<<16) - 1 - INIT_SEED))

    TRAIN_FOLDS='4'
    TEST_FOLDS='8:9'
    PHASES='1234:12:34'
    OPTIMIZER='sgd'
    LR='0.03125'
    MOMENTUM='0.0'
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
    echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    TRAIN_FOLDS='4::4::5::5'
    TEST_FOLDS='8:9'
    PHASES='12:34:12:34'
    OPTIMIZER='sgd'
    LR='0.0625'
    MOMENTUM='0.0'
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
    echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    TRAIN_FOLDS='4::4::5::5::6::6::7::7'
    TEST_FOLDS='8:9'
    PHASES='12:34:12:34:12:34:12:34'
    OPTIMIZER='sgd'
    LR='0.0625'
    MOMENTUM='0.0'
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
    echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    TRAIN_FOLDS='5::5::4::4'
    TEST_FOLDS='8:9'
    PHASES='34:12:34:12'
    OPTIMIZER='sgd'
    LR='0.0625'
    MOMENTUM='0.0'
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
    echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
done

#### Experiment 2 Train ######################################################

for INIT_SEED in `seq 0 49`; do
    SHUFFLE_SEED=$(((1<<16) - 1 - INIT_SEED))

    for LR in "${LEARNING_RATES[@]}"; do
        for MOMENTUM in '0.0' '0.81' '0.9' '0.99'; do
            TRAIN_FOLDS='0:1::0:1::2:3::2:3'
            TEST_FOLDS='4:5'
            PHASES='12:34:12:34'
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
            echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
        done

        for RHO in '0.81' '0.9' '0.99'; do
            TRAIN_FOLDS='0:1::0:1::2:3::2:3'
            TEST_FOLDS='4:5'
            PHASES='12:34:12:34'
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
            echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
        done

        for BETA_1 in '0.9'; do
            for BETA_2 in '0.999'; do
                TRAIN_FOLDS='0:1::0:1::2:3::2:3'
                TEST_FOLDS='4:5'
                PHASES='12:34:12:34'
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
                echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
            done
        done
    done
done

#### Experiment 2 Test #######################################################

for INIT_SEED in `seq 50 549`; do
    SHUFFLE_SEED=$(((1<<16) - 1 - INIT_SEED))

    TRAIN_FOLDS='6::6::7::7'
    TEST_FOLDS='8:9'
    PHASES='12:34:12:34'
    OPTIMIZER='sgd'
    MOMENTUM='0.0'
    LR='0.0625'
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
    echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    TRAIN_FOLDS='6::6::7::7'
    TEST_FOLDS='8:9'
    PHASES='12:34:12:34'
    OPTIMIZER='sgd'
    MOMENTUM='0.81'
    LR='0.0078125'
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
    echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    TRAIN_FOLDS='6::6::7::7'
    TEST_FOLDS='8:9'
    PHASES='12:34:12:34'
    OPTIMIZER='rms'
    RHO='0.99'
    LR='0.001953125'
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
    echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    TRAIN_FOLDS='6::6::7::7'
    TEST_FOLDS='8:9'
    PHASES='12:34:12:34'
    OPTIMIZER='adam'
    BETA_1='0.9'
    BETA_2='0.999'
    LR='0.001953125'
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
    echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
done

###############################################################################

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
