# Random-Maps-Generation
Map generation of random distributions with different methods

## CMB: 
Implementation of 2d random gaussian field generation. Resulting map is cosmic microwave background-like in small-angle case with different number of modes: 3,5, 10, 20.
![](https://github.com/alt2019/Random-Maps-Generation/blob/master/CMB/CMB-like-map-N1024-K3.png)
![](https://github.com/alt2019/Random-Maps-Generation/blob/master/CMB/CMB-like-map-N1024-K5.png)
![](https://github.com/alt2019/Random-Maps-Generation/blob/master/CMB/CMB-like-map-N1024-K10.png)
![](https://github.com/alt2019/Random-Maps-Generation/blob/master/CMB/CMB-like-map-N1024-K20.png)

Also here are implemented some methods of python code acceleration: 
* numpy matrices instead of nested raw for-loops;
* numba just-in-time compilation
* numba just-in-time compilation plus parallelization on CPU
* numba CUDA jit compilation with different types of threads grid organization: 1d, 2d, 3d

Also these methods are compared in performance for different output 2d map size NxN

* For small grid (N from 10 to 100):
![alt tag](https://github.com/alt2019/Random-Maps-Generation/blob/master/CMB/methods-comparison-smallN.png)

* For medium and large grid (N varies in 32, 64, 128, 256, 512, 1024, 2048):
![](https://github.com/alt2019/Random-Maps-Generation/blob/master/CMB/methods-comparison-medium-to-largeN.png)
