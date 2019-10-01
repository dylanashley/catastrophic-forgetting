#!/bin/sh

SCRIPT=$(dirname "$0")'/run.py'
TASKS_PER_FILE=100

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
for DATASET in 'mnist'; do
for FOLD_COUNT in '10'; do
for TEST_FOLDS in `seq 0 9`; do

if [ "$TEST_FOLDS" = '0' ]; then
    TRAIN_FOLDS='1:2:3:4:5:6:7:8:9'
elif [ "$TEST_FOLDS" = '1' ]; then
    TRAIN_FOLDS='0:2:3:4:5:6:7:8:9'
elif [ "$TEST_FOLDS" = '2' ]; then
    TRAIN_FOLDS='0:1:3:4:5:6:7:8:9'
elif [ "$TEST_FOLDS" = '3' ]; then
    TRAIN_FOLDS='0:1:2:4:5:6:7:8:9'
elif [ "$TEST_FOLDS" = '4' ]; then
    TRAIN_FOLDS='0:1:2:3:5:6:7:8:9'
elif [ "$TEST_FOLDS" = '5' ]; then
    TRAIN_FOLDS='0:1:2:3:4:6:7:8:9'
elif [ "$TEST_FOLDS" = '6' ]; then
    TRAIN_FOLDS='0:1:2:3:4:5:7:8:9'
elif [ "$TEST_FOLDS" = '7' ]; then
    TRAIN_FOLDS='0:1:2:3:4:5:6:8:9'
elif [ "$TEST_FOLDS" = '8' ]; then
    TRAIN_FOLDS='0:1:2:3:4:5:6:7:9'
elif [ "$TEST_FOLDS" = '9' ]; then
    TRAIN_FOLDS='0:1:2:3:4:5:6:7:8'
fi

for PHASES in '01:2:012:3:0123:4:01234'; do
for LOG_FREQUENCY in '100'; do
for CRITERIA in '0.95'; do
for TOLERANCE in '2500'; do
for ARCHITECTURE in '100'; do
for SEED in `seq 0 2`; do
for CRITERIA in 'steps' 'online'; do

if [ "$CRITERIA" = 'steps' ]; then
    STEPS='2000'
    REQUIRED_ACCURACY='None'
    TOLERANCE='None'
    VALIDATION_FOLDS='None'
    MINIMUM_STEPS='None'
elif [ "$CRITERIA" = 'online' ]; then
    STEPS='None'
    REQUIRED_ACCURACY='0.9'
    TOLERANCE='2000'
    VALIDATION_FOLDS='None'
    MINIMUM_STEPS='100'
fi

for LR in '1e-1' '1e-2' '1e-3' '1e-4' '1e-5'; do
    for MOMENTUM in '0.0' '0.75' '0.9' '0.999'; do
        OUTFILE="$I"'.json'
        I=$((I + 1))
        ARGS=("--outfile=$OUTFILE"
              "--dataset=$DATASET"
              "--fold-count=$FOLD_COUNT"
              "--train-folds=$TRAIN_FOLDS"
              "--test-folds=$TEST_FOLDS"
              "--phases=$PHASES"
              "--log-frequency=$LOG_FREQUENCY"
              "--architecture=$ARCHITECTURE"
              "--seed=$SEED"
              '--criteria=steps'
              '--steps=2000'
              '--optimizer=sgd'
              "--lr=$LR"
              "--momentum=$MOMENTUM")
        echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh

        OUTFILE="$I"'.json'
        I=$((I + 1))
        ARGS=("--outfile=$OUTFILE"
              "--dataset=$DATASET"
              "--fold-count=$FOLD_COUNT"
              "--train-folds=$TRAIN_FOLDS"
              "--test-folds=$TEST_FOLDS"
              "--phases=$PHASES"
              "--log-frequency=$LOG_FREQUENCY"
              "--architecture=$ARCHITECTURE"
              "--seed=$SEED"
              '--criteria=online'
              '--required-accuracy=0.9'
              '--tolerance=2000'
              '--minimum-steps=100'
              '--optimizer=sgd'
              "--lr=$LR"
              "--momentum=$MOMENTUM")
        echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
    done

    for RHO in '0.9' '0.999' '0.99999'; do
        OUTFILE="$I"'.json'
        I=$((I + 1))
        ARGS=("--outfile=$OUTFILE"
              "--dataset=$DATASET"
              "--fold-count=$FOLD_COUNT"
              "--train-folds=$TRAIN_FOLDS"
              "--test-folds=$TEST_FOLDS"
              "--phases=$PHASES"
              "--log-frequency=$LOG_FREQUENCY"
              "--architecture=$ARCHITECTURE"
              "--seed=$SEED"
              '--criteria=steps'
              '--steps=2000'
              '--optimizer=rms'
              "--lr=$LR"
              "--rho=$RHO")
        echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh

        OUTFILE="$I"'.json'
        I=$((I + 1))
        ARGS=("--outfile=$OUTFILE"
              "--dataset=$DATASET"
              "--fold-count=$FOLD_COUNT"
              "--train-folds=$TRAIN_FOLDS"
              "--test-folds=$TEST_FOLDS"
              "--phases=$PHASES"
              "--log-frequency=$LOG_FREQUENCY"
              "--architecture=$ARCHITECTURE"
              "--seed=$SEED"
              '--criteria=online'
              '--required-accuracy=0.9'
              '--tolerance=2000'
              '--minimum-steps=100'
              '--optimizer=rms'
              "--lr=$LR"
              "--rho=$RHO")
        echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
    done

    for BETA_1 in '0.75' '0.9' '0.999'; do
    for BETA_2 in '0.9' '0.999' '0.99999'; do
        OUTFILE="$I"'.json'
        I=$((I + 1))
        ARGS=("--outfile=$OUTFILE"
              "--dataset=$DATASET"
              "--fold-count=$FOLD_COUNT"
              "--train-folds=$TRAIN_FOLDS"
              "--test-folds=$TEST_FOLDS"
              "--phases=$PHASES"
              "--log-frequency=$LOG_FREQUENCY"
              "--architecture=$ARCHITECTURE"
              "--seed=$SEED"
              '--criteria=steps'
              '--steps=2000'
              '--optimizer=adam'
              "--lr=$LR"
              "--beta-1=$BETA_1"
              "--beta-2=$BETA_2")
        echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh

        OUTFILE="$I"'.json'
        I=$((I + 1))
        ARGS=("--outfile=$OUTFILE"
              "--dataset=$DATASET"
              "--fold-count=$FOLD_COUNT"
              "--train-folds=$TRAIN_FOLDS"
              "--test-folds=$TEST_FOLDS"
              "--phases=$PHASES"
              "--log-frequency=$LOG_FREQUENCY"
              "--architecture=$ARCHITECTURE"
              "--seed=$SEED"
              '--criteria=online'
              '--required-accuracy=0.9'
              '--tolerance=2000'
              '--minimum-steps=100'
              '--optimizer=adam'
              "--lr=$LR"
              "--beta-1=$BETA_1"
              "--beta-2=$BETA_2")
        echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
    done
    done
done
done
done
done
done
done
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
