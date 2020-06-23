#!/bin/sh

# test policy
python envs.py > policy_steps.txt

# build test states
python build_test_states.py 'test_states.npz' '50000000' '2500' '--trajectory-outfile' 'test_states_trajectory.npy' '--interference-outfile' 'interference_test_states.npz' '--interference-sample-size' '180'
openssl md5 test_states.npz > test_states.md5
openssl md5 test_states_trajectory.npy > test_states_trajectory.md5
openssl md5 interference_test_states.npz > interference_test_states.md5

# remove excessively large files
rm test_states_trajectory.npy
