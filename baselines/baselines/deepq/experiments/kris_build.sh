#!/bin/sh

SCRIPT=$(dirname "$0")'/kris_run.py'
TASKS_PER_FILE=50

# assert command line arguments valid
if [ "$#" -gt "0" ]
    then
        echo 'usage: ./build.sh'
        exit
    fi

# amalgamate all tasks
TASKS_PREFIX='kris_tasks_'
rm "$TASKS_PREFIX"*.sh 2>/dev/null
rm tasks.sh 2>/dev/null
for SEED in `seq 1 10`; do
for LR in '1e-1' '1e-2' '1e-3' '1e-4'; do
for BETA_1 in '0.75' '0.9' '0.999'; do
for BETA_2 in '0.9' '0.999' '0.99999'; do
for TRAIN_FREQ in '1' '10' '100' '500'; do
for BUFFER_SIZE in 1 50000; do
    PREFIX='kris_'"$SEED"'_'"$LR"'_'"$BETA_1"'_'"$BETA_2"'_'"$BUFFER_SIZE"'_'"$TRAIN_FREQ"'_'
    ARGS=("$SEED"
          "$LR"
          "$BETA_1"
          "$BETA_2"
          "$BUFFER_SIZE"
          "$TRAIN_FREQ"
          "500"
          "0.1")
    echo 'python '"$SCRIPT"' '"${ARGS[@]}"' >> '"$PREFIX"'results.csv' >> tasks.sh
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
