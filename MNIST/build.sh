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
for DATASET in 'mnist' 'fashion_mnist'; do
for FOLDS in '10'; do
for VALIDATION_FOLD in `seq 0 9`; do
TEST_FOLD=$(((VALIDATION_FOLD + 1) % 10))
for PHASES in '01:2:012:3:0123:4:01234'; do
for LOG_FREQUENCY in '100'; do
for CRITERIA in '0.95'; do
for TOLERANCE in '1000'; do
for ARCHITECTURE in '100' '50:50'; do
for SEED in `seq 0 2`; do
for LR in '1e-1' '1e-2' '1e-3' '1e-4' '1e-5'; do
    for MOMENTUM in '0.0' '0.75' '0.9' '0.999'; do
        ARGS=("$DATASET"
              "$FOLDS"
              "$VALIDATION_FOLD"
              "$TEST_FOLD"
              "$PHASES"
              "$LOG_FREQUENCY"
              "$CRITERIA"
              "$TOLERANCE"
              "$ARCHITECTURE"
              "$SEED"
              'sgd'
              "$LR"
              "$MOMENTUM")
        OUTFILE=$(echo "${ARGS[@]}" | sed 's/ /_/g')'.json'
        echo 'python '"$SCRIPT"' '"$OUTFILE"' '"${ARGS[@]}" >> tasks.sh
    done

    for RHO in '0.9' '0.999' '0.99999'; do
        ARGS=("$DATASET"
              "$FOLDS"
              "$VALIDATION_FOLD"
              "$TEST_FOLD"
              "$PHASES"
              "$LOG_FREQUENCY"
              "$CRITERIA"
              "$TOLERANCE"
              "$ARCHITECTURE"
              "$SEED"
              'rms'
              "$LR"
              "$RHO")
        OUTFILE=$(echo "${ARGS[@]}" | sed 's/ /_/g')'.json'
        echo 'python '"$SCRIPT"' '"$OUTFILE"' '"${ARGS[@]}" >> tasks.sh
    done

    for BETA_1 in '0.75' '0.9' '0.999'; do
    for BETA_2 in '0.9' '0.999' '0.99999'; do
        ARGS=("$DATASET"
              "$FOLDS"
              "$VALIDATION_FOLD"
              "$TEST_FOLD"
              "$PHASES"
              "$LOG_FREQUENCY"
              "$CRITERIA"
              "$TOLERANCE"
              "$ARCHITECTURE"
              "$SEED"
              'adam'
              "$LR"
              "$BETA_1"
              "$BETA_2")
        OUTFILE=$(echo "${ARGS[@]}" | sed 's/ /_/g')'.json'
        echo 'python '"$SCRIPT"' '"$OUTFILE"' '"${ARGS[@]}" >> tasks.sh
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
