#!/bin/bash

SCRIPT=$(dirname "$0")'/run.py'
TASKS_PER_FILE=200

POWERS_OF_TWO=('1.0'
               '0.5'
               '0.25'
               '0.125'
               '0.0625'
               '0.03125'
               '0.015625'
               '0.0078125'
               '0.00390625'
               '0.001953125'
               '0.0009765625'
               '0.00048828125'
               '0.000244140625'
               '0.0001220703125'
               '0.00006103515625'
               '0.000030517578125'
               '0.0000152587890625'
               '0.00000762939453125'
               '0.000003814697265625')

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
NUM_EPISODES=500
ENV_RANGE='classic'
LOSS='TD'
TARGET_UPDATE='1'
for ENV_SEED in `seq 50 299`; do
# for ENV_SEED in `seq 0 49`; do

    APPROXIMATOR='constant'
    OUTFILE="$I"'.json'
    I=$((I + 1))
    ARGS=("--outfile=$OUTFILE"
          "--num-episodes=$NUM_EPISODES"
          "--env-range=$ENV_RANGE"
          "--env-seed=$ENV_SEED"
          "--approximator=$APPROXIMATOR")
    echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    APPROXIMATOR='neural_network'
    NETWORK_SEED=$(((1<<16) - 1 - ENV_SEED))

    OPTIMIZER='sgd'
    OUTFILE="$I"'.json'
    I=$((I + 1))
    MOMENTUM='0.0'
    LR='0.00048828125'
    ARGS=("--outfile=$OUTFILE"
          "--num-episodes=$NUM_EPISODES"
          "--env-range=$ENV_RANGE"
          "--env-seed=$ENV_SEED"
          "--approximator=$APPROXIMATOR"
          "--network-seed=$NETWORK_SEED"
          "--loss=$LOSS"
          "--target-update=$TARGET_UPDATE"
          "--optimizer=$OPTIMIZER"
          "--lr=$LR"
          "--momentum=$MOMENTUM")
    echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    OPTIMIZER='sgd'
    OUTFILE="$I"'.json'
    I=$((I + 1))
    MOMENTUM='0.81'
    LR='0.0001220703125'
    ARGS=("--outfile=$OUTFILE"
          "--num-episodes=$NUM_EPISODES"
          "--env-range=$ENV_RANGE"
          "--env-seed=$ENV_SEED"
          "--approximator=$APPROXIMATOR"
          "--network-seed=$NETWORK_SEED"
          "--loss=$LOSS"
          "--target-update=$TARGET_UPDATE"
          "--optimizer=$OPTIMIZER"
          "--lr=$LR"
          "--momentum=$MOMENTUM")
    echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    OPTIMIZER='rms'
    OUTFILE="$I"'.json'
    I=$((I + 1))
    RHO='0.99'
    LR='0.00006103515625'
    ARGS=("--outfile=$OUTFILE"
          "--num-episodes=$NUM_EPISODES"
          "--env-range=$ENV_RANGE"
          "--env-seed=$ENV_SEED"
          "--approximator=$APPROXIMATOR"
          "--network-seed=$NETWORK_SEED"
          "--loss=$LOSS"
          "--target-update=$TARGET_UPDATE"
          "--optimizer=$OPTIMIZER"
          "--lr=$LR"
          "--rho=$RHO")
    echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    OPTIMIZER='adam'
    OUTFILE="$I"'.json'
    I=$((I + 1))
    BETA_1='0.9'
    BETA_2='0.999'
    LR='0.03125'
    ARGS=("--outfile=$OUTFILE"
          "--num-episodes=$NUM_EPISODES"
          "--env-range=$ENV_RANGE"
          "--env-seed=$ENV_SEED"
          "--approximator=$APPROXIMATOR"
          "--network-seed=$NETWORK_SEED"
          "--loss=$LOSS"
          "--target-update=$TARGET_UPDATE"
          "--optimizer=$OPTIMIZER"
          "--lr=$LR"
          "--beta-1=$BETA_1"
          "--beta-2=$BETA_2")
    echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    # for LR in "${POWERS_OF_TWO[@]}"; do
    #     for MOMENTUM in '0.0' '0.9'; do
    #         OPTIMIZER='sgd'
    #         OUTFILE="$I"'.json'
    #         I=$((I + 1))
    #         ARGS=("--outfile=$OUTFILE"
    #               "--num-episodes=$NUM_EPISODES"
    #               "--env-range=$ENV_RANGE"
    #               "--env-seed=$ENV_SEED"
    #               "--approximator=$APPROXIMATOR"
    #               "--network-seed=$NETWORK_SEED"
    #               "--loss=$LOSS"
    #               "--target-update=$TARGET_UPDATE"
    #               "--optimizer=$OPTIMIZER"
    #               "--lr=$LR"
    #               "--momentum=$MOMENTUM")
    #         echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
    #     done
    #
    #     for RHO in '0.9' '0.99'; do
    #         OPTIMIZER='rms'
    #         OUTFILE="$I"'.json'
    #         I=$((I + 1))
    #         ARGS=("--outfile=$OUTFILE"
    #               "--num-episodes=$NUM_EPISODES"
    #               "--env-range=$ENV_RANGE"
    #               "--env-seed=$ENV_SEED"
    #               "--approximator=$APPROXIMATOR"
    #               "--network-seed=$NETWORK_SEED"
    #               "--loss=$LOSS"
    #               "--target-update=$TARGET_UPDATE"
    #               "--optimizer=$OPTIMIZER"
    #               "--lr=$LR"
    #               "--rho=$RHO")
    #         echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
    #     done
    #
    #     for BETA_1 in '0.9'; do
    #         for BETA_2 in '0.999'; do
    #             OPTIMIZER='adam'
    #             OUTFILE="$I"'.json'
    #             I=$((I + 1))
    #             ARGS=("--outfile=$OUTFILE"
    #                   "--num-episodes=$NUM_EPISODES"
    #                   "--env-range=$ENV_RANGE"
    #                   "--env-seed=$ENV_SEED"
    #                   "--approximator=$APPROXIMATOR"
    #                   "--network-seed=$NETWORK_SEED"
    #                   "--loss=$LOSS"
    #                   "--target-update=$TARGET_UPDATE"
    #                   "--optimizer=$OPTIMIZER"
    #                   "--lr=$LR"
    #                   "--beta-1=$BETA_1"
    #                   "--beta-2=$BETA_2")
    #             echo 'python '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
    #         done
    #     done
    # done
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
