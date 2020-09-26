# Random-Maps-Generation
Map generation of random distributions with different methods

## CMB: 
Implementation of 2d random gaussian field generation. Resulting map is cosmic microwave background-like in small-angle case. Also here are implemented some methods of python code acceleration: 
* 1) numpy matrices instead of nested raw for-loops;
* 2) numba just-in-time compilation
* 3) numba just-in-time compilation plus parallelization on CPU
* 4) numba CUDA jit compilation with different types of threads grid organization: 1d, 2d, 3d

Also these methods are compared in performance for different output 2d map size NxN
