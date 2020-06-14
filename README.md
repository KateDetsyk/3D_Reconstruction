# 3D Reconstruction

### Build && Run :
```
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release  -G"Unix Makefiles" ..
$ make
```
Then to run program
```
$ ./reconstruct
```
To run parallel version 
```
$ ./reconstruct_t
```

To run mpi vers, you should **uncomment** `# for mpi` lines (5, 6, 7, 36, 37 lines) and **rebuild** project.
Run mpi like this
```
$ mpirun -np [number of processes] ./mpi
```

### Configurations :
You can specify a configuration in the file `conf.txt`
```
    with_calibration: true
    find_points: true
    save_points: true
    visualize: true
    sgbm: true
    downscale: false
    surf: true
    maxThreadsNumber: 4
    lambda: 8000.0
    sigma: 1.5
    vis_mult: 1.0
    minHessian: 600
    preFilterCap: 63
    disparityRange: 13
    minDisparity: -1
    uniquenessRatio: 5
    windowSize: 3
    smoothP1: 8
    smoothP2: 32
    disparityMaxDiff: 1
    speckleRange: 0
    speckleWindowSize: 0
```
If you already calibrate you camera and have `cameraMatrix` and `distortionCoefficient` stored in 
`CalibrationMatrices.yml`, you can make parameter `with_calibration` **false** and the program will 
only look for points.
If you don't want to find points or to save them, you can also make this params **false**. 