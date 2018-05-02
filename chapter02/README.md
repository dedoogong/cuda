./sumMatrixOnGPU Starting...
Using Device 0: GeForce GTX 960
Matrix size: nx 1024 ny 1024
Matrix initialization elapsed 0.045114 sec
sumMatrixOnHost elapsed 0.001137 sec
sumMatrixOnGPU2D <<<  (32,32), (32,32)  >>> elapsed 0.000255 sec
sumMatrixOnGPU2D <<<  (64,32), (16,32)  >>> elapsed 0.000160 sec
sumMatrixOnGPU2D <<<  (32,64), (32,16)  >>> elapsed 0.000162 sec
sumMatrixOnGPU2D <<<  (64,64), (16,16)  >>> elapsed 0.000160 sec
sumMatrixOnGPU2D <<<  (16,64), (64,16)  >>> elapsed 0.000160 sec
sumMatrixOnGPU2D <<<  (64,16), (16,64)  >>> elapsed 0.000160 sec
sumMatrixOnGPU1D <<<  (32,1), (32,1)  >>> elapsed 0.000438 sec
sumMatrixOnGPU1D <<<  (16,1), (64,1)  >>> elapsed 0.000438 sec
sumMatrixOnGPU1D <<<  (8,1), (128,1)  >>> elapsed 0.000434 sec
sumMatrixOnGPUMix <<<  (32,1024), (32,1)  >>> elapsed 0.000160 sec
sumMatrixOnGPUMix <<<  (16,1024), (64,1)  >>> elapsed 0.000159 sec
sumMatrixOnGPUMix <<<  (8,1024), (128,1)  >>> elapsed 0.000157 sec
sumMatrixOnGPUMix <<<  (4,1024), (256,1)  >>> elapsed 0.000160 sec
sumMatrixOnGPUMix <<<  (2,1024), (512,1)  >>> elapsed 0.000161 sec
Arrays match.
