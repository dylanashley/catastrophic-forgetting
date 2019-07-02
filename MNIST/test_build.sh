#!/bin/sh

SCRIPT=$(dirname "$0")'/test_run.py'
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
for SEED in `seq 0 0`; do
for DIGITS in '1;2;3'; do
for FOLD in `seq 0 9`; do
for EPOCHS in '5;10;25'; do
for LR in '1e-1' '1e-2' '1e-3' '1e-4'; do
        for MOMENTUM in '0.0' '0.75' '0.9' '0.999'; do
            PREFIX='test_'"$DIGITS"'_'"$FOLD"'_sgd_'"$EPOCHS"'_'"$LR"'_'"$MOMENTUM"'_'
            ARGS=("'$PREFIX'"
                  "'$DIGITS'"
                  "$SEED"
                  "$FOLD"
                  'sgd'
                  "'$EPOCHS'"
                  "$LR"
                  "$MOMENTUM")
            echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
        done

        for RHO in '0.9' '0.999' '0.99999'; do
            PREFIX='test_'"$DIGITS"'_'"$FOLD"'_rms_'"$EPOCHS"'_'"$LR"'_'"$RHO"'_'
            ARGS=("'$PREFIX'"
                  "'$DIGITS'"
                  "$SEED"
                  "$FOLD"
                  'rms'
                  "'$EPOCHS'"
                  "$LR"
                  "$RHO")
            echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
        done

        for BETA_1 in '0.75' '0.9' '0.999'; do
        for BETA_2 in '0.9' '0.999' '0.99999'; do
            PREFIX='test_'"$DIGITS"'_'"$FOLD"'_adam_'"$EPOCHS"'_'"$LR"'_'"$BETA_1"'_'"$BETA_2"'_'
            ARGS=("'$PREFIX'"
                  "'$DIGITS'"
                  "$SEED"
                  "$FOLD"
                  'adam'
                  "'$EPOCHS'"
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
