# 3D Reconstruction

You can specify a configuration in the file `conf.txt`
```
   with_calibration: true
   find_points: true
   save_points: true
   sgbm: true
   downscale: true
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