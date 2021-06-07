#!/bin/sh

if [[ "$1" == "run_train" ]]; then
    for FOLDER in 'acrobot' 'fashion_mnist' 'mnist' 'mountain_car'; do
        cd $FOLDER
        rm *.json 2>/dev/null
        sbatch --array=0-$(./build.sh train) ./run_cc.sh
        cd ..
    done
fi

if [[ "$1" == "merge_train" ]]; then
    for FOLDER in 'acrobot' 'fashion_mnist' 'mnist' 'mountain_car'; do
        cd $FOLDER
        sbatch ./merge_cc.sh '../'"$FOLDER"'_validation.json'
        cd ..
    done
fi

if [[ "$1" == "run_test" ]]; then
    for FOLDER in 'acrobot' 'fashion_mnist' 'mnist' 'mountain_car'; do
        cd $FOLDER
        rm *.json 2>/dev/null
        sbatch --array=0-$(./build.sh test) ./run_cc.sh
        cd ..
    done
fi

if [[ "$1" == "merge_test" ]]; then
    for FOLDER in 'acrobot' 'fashion_mnist' 'mnist' 'mountain_car'; do
        cd $FOLDER
        sbatch ./merge_cc.sh '../'"$FOLDER"'_test.json'
        cd ..
    done
fi
