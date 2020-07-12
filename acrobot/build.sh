#!/bin/bash

SCRIPT=$(dirname "$0")'/run.py'
TASKS_PER_FILE=25


LEARNING_RATES=('0.125'
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
                '6.103515625e-05'
                '3.0517578125e-05'
                '1.52587890625e-05'
                '7.62939453125e-06'
                '3.814697265625e-06')

# LEARNING_RATES=('0.125'
#                 '0.08838834764831845'
#                 '0.0625'
#                 '0.04419417382415922'
#                 '0.03125'
#                 '0.02209708691207961'
#                 '0.015625'
#                 '0.011048543456039806'
#                 '0.0078125'
#                 '0.005524271728019903'
#                 '0.00390625'
#                 '0.0027621358640099515'
#                 '0.001953125'
#                 '0.0013810679320049757'
#                 '0.0009765625'
#                 '0.0006905339660024879'
#                 '0.00048828125'
#                 '0.00034526698300124393'
#                 '0.000244140625'
#                 '0.00017263349150062197'
#                 '0.0001220703125'
#                 '8.631674575031098e-05'
#                 '6.103515625e-05'
#                 '4.315837287515549e-05'
#                 '3.0517578125e-05'
#                 '2.1579186437577746e-05'
#                 '1.52587890625e-05'
#                 '1.0789593218788873e-05'
#                 '7.62939453125e-06'
#                 '5.3947966093944364e-06'
#                 '3.814697265625e-06')

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
NUM_EPISODES=2500
LOSS='TD'
TARGET_UPDATE='1'

##### Train ###################################################################

for ENV_SEED in `seq 0 49`; do

    APPROXIMATOR='constant'
    OUTFILE="$I"'.json'
    I=$((I + 1))
    ARGS=("--outfile=$OUTFILE"
          "--num-episodes=$NUM_EPISODES"
          "--env-seed=$ENV_SEED"
          "--approximator=$APPROXIMATOR")
    echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

    APPROXIMATOR='neural_network'
    NETWORK_SEED=$(((1<<16) - 1 - ENV_SEED))

    for LR in "${LEARNING_RATES[@]}"; do
        # for MOMENTUM in '0.0' '0.81' '0.9' '0.99'; do
        for MOMENTUM in '0.0'; do
            OPTIMIZER='sgd'
            OUTFILE="$I"'.json'
            I=$((I + 1))
            ARGS=("--outfile=$OUTFILE"
                  "--num-episodes=$NUM_EPISODES"
                  "--env-seed=$ENV_SEED"
                  "--approximator=$APPROXIMATOR"
                  "--network-seed=$NETWORK_SEED"
                  "--loss=$LOSS"
                  "--target-update=$TARGET_UPDATE"
                  "--optimizer=$OPTIMIZER"
                  "--lr=$LR"
                  "--momentum=$MOMENTUM")
            echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
        done

        # for RHO in '0.81' '0.9' '0.99'; do
        #     OPTIMIZER='rms'
        #     OUTFILE="$I"'.json'
        #     I=$((I + 1))
        #     ARGS=("--outfile=$OUTFILE"
        #           "--num-episodes=$NUM_EPISODES"
        #           "--env-seed=$ENV_SEED"
        #           "--approximator=$APPROXIMATOR"
        #           "--network-seed=$NETWORK_SEED"
        #           "--loss=$LOSS"
        #           "--target-update=$TARGET_UPDATE"
        #           "--optimizer=$OPTIMIZER"
        #           "--lr=$LR"
        #           "--rho=$RHO")
        #     echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
        # done

        # for BETA_1 in '0.9'; do
        #     for BETA_2 in '0.999'; do
        #         OPTIMIZER='adam'
        #         OUTFILE="$I"'.json'
        #         I=$((I + 1))
        #         ARGS=("--outfile=$OUTFILE"
        #               "--num-episodes=$NUM_EPISODES"
        #               "--env-seed=$ENV_SEED"
        #               "--approximator=$APPROXIMATOR"
        #               "--network-seed=$NETWORK_SEED"
        #               "--loss=$LOSS"
        #               "--target-update=$TARGET_UPDATE"
        #               "--optimizer=$OPTIMIZER"
        #               "--lr=$LR"
        #               "--beta-1=$BETA_1"
        #               "--beta-2=$BETA_2")
        #         echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
        #     done
        # done
    done
done

# ##### Test ####################################################################

# for ENV_SEED in `seq 50 549`; do

#     APPROXIMATOR='constant'
#     OUTFILE="$I"'.json'
#     I=$((I + 1))
#     ARGS=("--outfile=$OUTFILE"
#           "--num-episodes=$NUM_EPISODES"
#           "--env-seed=$ENV_SEED"
#           "--approximator=$APPROXIMATOR")
#     echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

#     APPROXIMATOR='neural_network'
#     NETWORK_SEED=$(((1<<16) - 1 - ENV_SEED))

#     OPTIMIZER='sgd'
#     OUTFILE="$I"'.json'
#     I=$((I + 1))
#     MOMENTUM='0.0'
#     LR=''
#     ARGS=("--outfile=$OUTFILE"
#           "--num-episodes=$NUM_EPISODES"
#           "--env-seed=$ENV_SEED"
#           "--approximator=$APPROXIMATOR"
#           "--network-seed=$NETWORK_SEED"
#           "--loss=$LOSS"
#           "--target-update=$TARGET_UPDATE"
#           "--optimizer=$OPTIMIZER"
#           "--lr=$LR"
#           "--momentum=$MOMENTUM")
#     echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

#     OPTIMIZER='sgd'
#     OUTFILE="$I"'.json'
#     I=$((I + 1))
#     MOMENTUM=''
#     LR=''
#     ARGS=("--outfile=$OUTFILE"
#           "--num-episodes=$NUM_EPISODES"
#           "--env-seed=$ENV_SEED"
#           "--approximator=$APPROXIMATOR"
#           "--network-seed=$NETWORK_SEED"
#           "--loss=$LOSS"
#           "--target-update=$TARGET_UPDATE"
#           "--optimizer=$OPTIMIZER"
#           "--lr=$LR"
#           "--momentum=$MOMENTUM")
#     echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

#     OPTIMIZER='rms'
#     OUTFILE="$I"'.json'
#     I=$((I + 1))
#     RHO=''
#     LR=''
#     ARGS=("--outfile=$OUTFILE"
#           "--num-episodes=$NUM_EPISODES"
#           "--env-seed=$ENV_SEED"
#           "--approximator=$APPROXIMATOR"
#           "--network-seed=$NETWORK_SEED"
#           "--loss=$LOSS"
#           "--target-update=$TARGET_UPDATE"
#           "--optimizer=$OPTIMIZER"
#           "--lr=$LR"
#           "--rho=$RHO")
#     echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

#     OPTIMIZER='adam'
#     OUTFILE="$I"'.json'
#     I=$((I + 1))
#     BETA_1='0.9'
#     BETA_2='0.999'
#     LR=''
#     ARGS=("--outfile=$OUTFILE"
#           "--num-episodes=$NUM_EPISODES"
#           "--env-seed=$ENV_SEED"
#           "--approximator=$APPROXIMATOR"
#           "--network-seed=$NETWORK_SEED"
#           "--loss=$LOSS"
#           "--target-update=$TARGET_UPDATE"
#           "--optimizer=$OPTIMIZER"
#           "--lr=$LR"
#           "--beta-1=$BETA_1"
#           "--beta-2=$BETA_2")
#     echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
# done

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
