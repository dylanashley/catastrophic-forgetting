#!/bin/sh

SCRIPT=$(dirname "$0")'/123_run.py'
TASKS_PER_FILE=200

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
for SEED in `seq 0 0`; do
for FOLD in `seq 0 9`; do
for T1_EPOCHS in '5' '10' '25'; do
for T2_EPOCHS in '5' '10' '25'; do
for LR in '1e-1' '1e-2' '1e-3' '1e-4'; do
        for MOMENTUM in '0.0' '0.75' '0.9' '0.999'; do
            PREFIX='123_'"$FOLD"'_sgd_'"$T1_EPOCHS"'_'"$T2_EPOCHS"'_'"$LR"'_'"$MOMENTUM"'_'
            ARGS=("$PREFIX"
                  "$SEED"
                  "$FOLD"
                  'sgd'
                  "$T1_EPOCHS"
                  "$T2_EPOCHS"
                  "$LR"
                  "$MOMENTUM")
            echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
        done

        for RHO in '0.9' '0.999' '0.99999'; do
            PREFIX='123_'"$FOLD"'_rms_'"$T1_EPOCHS"'_'"$T2_EPOCHS"'_'"$LR"'_'"$RHO"'_'
            ARGS=("$PREFIX"
                  "$SEED"
                  "$FOLD"
                  'rms'
                  "$T1_EPOCHS"
                  "$T2_EPOCHS"
                  "$LR"
                  "$RHO")
            echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
        done

        for BETA_1 in '0.75' '0.9' '0.999'; do
        for BETA_2 in '0.9' '0.999' '0.99999'; do
            PREFIX='123_'"$FOLD"'_adam_'"$T1_EPOCHS"'_'"$T2_EPOCHS"'_'"$LR"'_'"$BETA_1"'_'"$BETA_2"'_'
            ARGS=("$PREFIX"
                  "$SEED"
                  "$FOLD"
                  'adam'
                  "$T1_EPOCHS"
                  "$T2_EPOCHS"
                  "$LR"
                  "$BETA_1"
                  "$BETA_2")
            echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
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
