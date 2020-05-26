#!/bin/sh

# build test states
python build_test_states.py 'test_states.npz' 'classic' '10000000' '500' '--trajectory-outfile' 'test_states_trajectory.npy' '--interference-outfile' 'interference_test_states.npz'
openssl md5 test_states.npz > test_states.md5
openssl md5 test_states_trajectory.npy > test_states_trajectory.md5
openssl md5 interference_test_states.npz > interference_test_states.md5

# build alternate test states
python build_test_states.py 'alternate_test_states.npz' 'full' '10000000' '500' '--trajectory-outfile' 'alternate_test_states_trajectory.npy' '--interference-outfile' 'interference_alternate_test_states.npz'
openssl md5 alternate_test_states.npz > alternate_test_states.md5
openssl md5 alternate_test_states_trajectory.npy > alternate_test_states_trajectory.md5
openssl md5 interference_alternate_test_states.npz > interference_alternate_test_states.md5

# build plots
python plot_test_states.py 'test_states.npz' 'test_states_trajectory.npy' 'test_states.pdf'
python plot_test_states.py 'alternate_test_states.npz' 'alternate_test_states_trajectory.npy' 'alternate_test_states.pdf'
python plot_state_values.py

# remove excessively large files
rm test_states_trajectory.npy
rm alternate_test_states_trajectory.npy
