# Ping Pong Benchmark

1) mkdir build && cd build
2) cmake -Dcaliper_DIR="<path-to-caliper>/share/cmake/caliper" -DUSE_CALIPER=on ..
3) make

To run: srun --ntasks size -i num_ranks -p which_ranks
