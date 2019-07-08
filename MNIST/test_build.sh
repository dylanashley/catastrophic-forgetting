#!/bin/sh

SCRIPT=$(dirname "$0")'/test_run.py'
TASKS_PER_FILE=50

# assert command line arguments valid
if [ "$#" -gt "0" ]
    then
        echo 'usage: ./build.sh'
        exit
    fi

# amalgamate all tasks
TASKS_PREFIX='test_tasks_'
rm "$TASKS_PREFIX"*.sh 2>/dev/null
rm tasks.sh 2>/dev/null
for SEED in `seq 0 2`; do
for DIGITS in '1:2:3'; do
for FOLD in `seq 0 9`; do
for ARCH in '100' '50:50'; do
for EPOCHS in '5:10:20:40:80'; do
for LR in '1e-1' '1e-2' '1e-3' '1e-4'; do
        for MOMENTUM in '0.0' '0.75' '0.9' '0.999'; do
            PREFIX='test_'"$SEED"'_'"$DIGITS"'_'"$FOLD"'_'"$ARCH"'_sgd_'"$EPOCHS"'_'"$LR"'_'"$MOMENTUM"'_'
            ARGS=("$PREFIX"
                  "$DIGITS"
                  "$SEED"
                  "$FOLD"
          "$ARCH"
                  'sgd'
                  "$EPOCHS"
                  "$LR"
                  "$MOMENTUM")
            echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
        done

        for RHO in '0.9' '0.999' '0.99999'; do
            PREFIX='test_'"$SEED"'_'"$DIGITS"'_'"$FOLD"'_'"$ARCH"'_rms_'"$EPOCHS"'_'"$LR"'_'"$RHO"'_'
            ARGS=("$PREFIX"
                  "$DIGITS"
                  "$SEED"
                  "$FOLD"
          "$ARCH"
                  'rms'
                  "$EPOCHS"
                  "$LR"
                  "$RHO")
            echo 'python '"$SCRIPT"' '"${ARGS[@]}" >> tasks.sh
        done

        for BETA_1 in '0.75' '0.9' '0.999'; do
        for BETA_2 in '0.9' '0.999' '0.99999'; do
            PREFIX='test_'"$SEED"'_'"$DIGITS"'_'"$FOLD"'_'"$ARCH"'_adam_'"$EPOCHS"'_'"$LR"'_'"$BETA_1"'_'"$BETA_2"'_'
            ARGS=("$PREFIX"
              "$DIGITS"
                  "$SEED"
                  "$FOLD"
          "$ARCH"
                  'adam'
                  "$EPOCHS"
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
