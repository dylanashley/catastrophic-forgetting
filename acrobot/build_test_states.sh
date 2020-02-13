#!/bin/sh

python build_test_states.py 'test_states.npz' '10000000' '500' '--trajectory-outfile' 'test_states_trajectory.npy' '--interference-outfile' 'interference_test_states.npz' '--interference-sample-size' '100'
openssl md5 test_states.npz > test_states.md5
openssl md5 test_states_trajectory.npy > test_states_trajectory.md5
openssl md5 interference_test_states.npz > interference_test_states.md5
rm test_states_trajectory.npy  # file is too big to keep around
