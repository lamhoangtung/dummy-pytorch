# Pytorch Dot Product `@` Bug
This repo is used to reproduce issue [#70162: Dot product return completely incorrect result when using pip but not when using conda](https://github.com/pytorch/pytorch/issues/70162) of pytorch.

## Torch 1.8.1 - CUDA 11.1 - Conda
Setup:
```bash
mamba create -y -n torch_181_conda python=3.6
conda activate torch_181_conda
mamba install -y pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c pytorch -c nvidia
python3 test_dot_product.py
```
Output: Correct
```console
....
        [[ 23.9812,   0.6946,  -2.6633],
         [ -0.9827,  16.9505,   9.5244],
         [  0.0000,   0.0000,   1.0000]],

        [[ 23.4596,  -1.7143, -12.2290],
         [  2.4253,  16.5818,  -2.7034],
         [  0.0000,   0.0000,   1.0000]]], device='cuda:0',
       dtype=torch.float64)
True
```
Environment:
```console
PyTorch version: 1.8.1
Is debug build: False
CUDA used to build PyTorch: 11.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.3 LTS (x86_64)
GCC version: (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Clang version: Could not collect
CMake version: version 3.19.7
Libc version: glibc-2.17

Python version: 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59)  [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.4.0-91-generic-x86_64-with-debian-bullseye-sid
Is CUDA available: True
CUDA runtime version: 11.1.105
GPU models and configuration: 
GPU 0: GeForce GTX 1080 Ti
GPU 1: GeForce RTX 3090

Nvidia driver version: 460.84
cuDNN version: Probably one of the following:
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.1.1
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip3] numpy==1.19.2
[pip3] torch==1.8.1
[pip3] torchaudio==0.8.0a0+e4e171a
[pip3] torchvision==0.9.1
[conda] blas                      1.0                         mkl  
[conda] cudatoolkit               11.1.74              h6bb024c_0    nvidia
[conda] ffmpeg                    4.3                  hf484d3e_0    pytorch
[conda] mkl                       2020.2                      256  
[conda] mkl-service               2.3.0            py36he904b0f_0  
[conda] mkl_fft                   1.3.0            py36h54f3939_0  
[conda] mkl_random                1.1.1            py36h0573a6f_0  
[conda] numpy                     1.19.2           py36h54aff64_0  
[conda] numpy-base                1.19.2           py36hfa32c7d_0  
[conda] pytorch                   1.8.1           py3.6_cuda11.1_cudnn8.0.5_0    pytorch
[conda] torchaudio                0.8.1                      py36    pytorch
[conda] torchvision               0.9.1                py36_cu111    pytorch
```

## Torch 1.8.1 - CUDA 11.1 - Pip
Setup:
```bash
mamba create -y -n torch_181_pip python=3.6
conda activate torch_181_pip
mamba install -y cudatoolkit=11.1 -c nvidia
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
python3 test_dot_product.py
```
Output: Incorrect
```console
....
        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0', dtype=torch.float64)
False
```
Environment:
```console
PyTorch version: 1.8.1+cu111
Is debug build: False
CUDA used to build PyTorch: 11.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.3 LTS (x86_64)
GCC version: (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Clang version: Could not collect
CMake version: version 3.19.7
Libc version: glibc-2.17

Python version: 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59)  [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.4.0-91-generic-x86_64-with-debian-bullseye-sid
Is CUDA available: True
CUDA runtime version: 11.1.105
GPU models and configuration: 
GPU 0: GeForce GTX 1080 Ti
GPU 1: GeForce RTX 3090

Nvidia driver version: 460.84
cuDNN version: Probably one of the following:
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.1.1
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip3] numpy==1.19.5
[pip3] torch==1.8.1+cu111
[pip3] torchaudio==0.8.1
[pip3] torchvision==0.9.1+cu111
[conda] cudatoolkit               11.1.74              h6bb024c_0    nvidia
[conda] numpy                     1.19.5                   pypi_0    pypi
[conda] torch                     1.8.1+cu111              pypi_0    pypi
[conda] torchaudio                0.8.1                    pypi_0    pypi
[conda] torchvision               0.9.1+cu111              pypi_0    pypi
```

## Torch 1.8.2 - CUDA 11.1 - Conda
Setup:
```bash
mamba create -y -n torch_182_conda python=3.6
conda activate torch_182_conda
mamba install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
python3 test_dot_product.py
```
Output: Correct
```console
        [[ 23.9812,   0.6946,  -2.6633],
         [ -0.9827,  16.9505,   9.5244],
         [  0.0000,   0.0000,   1.0000]],

        [[ 23.4596,  -1.7143, -12.2290],
         [  2.4253,  16.5818,  -2.7034],
         [  0.0000,   0.0000,   1.0000]]], device='cuda:0',
       dtype=torch.float64)
True
```
Environment:
```console
PyTorch version: 1.8.2
Is debug build: False
CUDA used to build PyTorch: 11.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.3 LTS (x86_64)
GCC version: (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Clang version: Could not collect
CMake version: version 3.19.7
Libc version: glibc-2.17

Python version: 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59)  [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.4.0-91-generic-x86_64-with-debian-bullseye-sid
Is CUDA available: True
CUDA runtime version: 11.1.105
GPU models and configuration: 
GPU 0: GeForce GTX 1080 Ti
GPU 1: GeForce RTX 3090

Nvidia driver version: 460.84
cuDNN version: Probably one of the following:
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.1.1
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip3] numpy==1.19.2
[pip3] torch==1.8.2
[pip3] torchaudio==0.8.2
[pip3] torchvision==0.9.2
[conda] blas                      1.0                         mkl  
[conda] cudatoolkit               11.1.74              h6bb024c_0    nvidia
[conda] mkl                       2020.2                      256  
[conda] mkl-service               2.3.0            py36he904b0f_0  
[conda] mkl_fft                   1.3.0            py36h54f3939_0  
[conda] mkl_random                1.1.1            py36h0573a6f_0  
[conda] numpy                     1.19.2           py36h54aff64_0  
[conda] numpy-base                1.19.2           py36hfa32c7d_0  
[conda] pytorch                   1.8.2           py3.6_cuda11.1_cudnn8.0.5_0    pytorch-lts
[conda] torchaudio                0.8.2                      py36    pytorch-lts
[conda] torchvision               0.9.2                py36_cu111    pytorch-lts
```

## Torch 1.8.2 - CUDA 11.1 - Pip
Setup:
```bash
mamba create -y -n torch_182_pip python=3.6
conda activate torch_182_pip
mamba install -y cudatoolkit=11.1 -c nvidia
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
python3 test_dot_product.py
```
Output: Incorrect
```
....
        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0', dtype=torch.float64)
False
```
Environment:
```console
PyTorch version: 1.8.2+cu111
Is debug build: False
CUDA used to build PyTorch: 11.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.3 LTS (x86_64)
GCC version: (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Clang version: Could not collect
CMake version: version 3.19.7
Libc version: glibc-2.17

Python version: 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59)  [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.4.0-91-generic-x86_64-with-debian-bullseye-sid
Is CUDA available: True
CUDA runtime version: 11.1.105
GPU models and configuration: 
GPU 0: GeForce GTX 1080 Ti
GPU 1: GeForce RTX 3090

Nvidia driver version: 460.84
cuDNN version: Probably one of the following:
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.1.1
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip3] numpy==1.19.5
[pip3] torch==1.8.2+cu111
[pip3] torchaudio==0.8.2
[pip3] torchvision==0.9.2+cu111
[conda] cudatoolkit               11.1.74              h6bb024c_0    nvidia
[conda] numpy                     1.19.5                   pypi_0    pypi
[conda] torch                     1.8.2+cu111              pypi_0    pypi
[conda] torchaudio                0.8.2                    pypi_0    pypi
[conda] torchvision               0.9.2+cu111              pypi_0    pypi
```

## Torch 1.10.1 - CUDA 11.3 - Conda
Setup:
```bash
mamba create -y -n torch_1101_conda python=3.6
conda activate torch_1101_conda
mamba install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
python3 test_dot_product.py
```

Output: Not an exact match, but correct
```console
....
        [[ 23.9812,   0.6946,  -2.6633],
         [ -0.9827,  16.9505,   9.5244],
         [  0.0000,   0.0000,   1.0000]],

        [[ 23.4596,  -1.7143, -12.2290],
         [  2.4253,  16.5818,  -2.7034],
         [  0.0000,   0.0000,   1.0000]]], device='cuda:0',
       dtype=torch.float64)
False
```
Environment:
```console
Collecting environment information...
PyTorch version: 1.10.1
Is debug build: False
CUDA used to build PyTorch: 11.3
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.3 LTS (x86_64)
GCC version: (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Clang version: Could not collect
CMake version: version 3.19.7
Libc version: glibc-2.17

Python version: 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59)  [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.4.0-91-generic-x86_64-with-debian-bullseye-sid
Is CUDA available: True
CUDA runtime version: 11.1.105
GPU models and configuration: 
GPU 0: GeForce GTX 1080 Ti
GPU 1: GeForce RTX 3090

Nvidia driver version: 460.84
cuDNN version: Probably one of the following:
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.1.1
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip3] numpy==1.16.6
[pip3] torch==1.10.1
[pip3] torchaudio==0.10.1
[pip3] torchvision==0.11.2
[conda] blas                      1.0                         mkl  
[conda] cudatoolkit               11.3.1               h2bc3f7f_2  
[conda] ffmpeg                    4.3                  hf484d3e_0    pytorch
[conda] mkl                       2021.4.0           h06a4308_640  
[conda] mkl-service               2.4.0            py36h7f8727e_0  
[conda] mkl_fft                   1.3.0            py36h42c9631_2  
[conda] mkl_random                1.2.2            py36h51133e4_0  
[conda] numpy                     1.16.6           py36h2d18471_3  
[conda] numpy-base                1.16.6           py36hdc34a94_3  
[conda] pytorch                   1.10.1          py3.6_cuda11.3_cudnn8.2.0_0    pytorch
[conda] pytorch-mutex             1.0                        cuda    pytorch
[conda] torchaudio                0.10.1               py36_cu113    pytorch
[conda] torchvision               0.11.2               py36_cu113    pytorch
```

## Torch 1.10.1 - CUDA 11.3 - Pip
Setup:
```bash
mamba create -y -n torch_1101_pip python=3.6
conda activate torch_1101_pip
mamba install -y cudatoolkit=11.3 -c nvidia
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
python3 test_dot_product.py
```
Output: Not an exact match, but correct
```console
....
        [[ 23.9812,   0.6946,  -2.6633],
         [ -0.9827,  16.9505,   9.5244],
         [  0.0000,   0.0000,   1.0000]],

        [[ 23.4596,  -1.7143, -12.2290],
         [  2.4253,  16.5818,  -2.7034],
         [  0.0000,   0.0000,   1.0000]]], device='cuda:0',
       dtype=torch.float64)
False
```
Environment:
```console
Collecting environment information...
PyTorch version: 1.10.1+cu113
Is debug build: False
CUDA used to build PyTorch: 11.3
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.3 LTS (x86_64)
GCC version: (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Clang version: Could not collect
CMake version: version 3.19.7
Libc version: glibc-2.17

Python version: 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59)  [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.4.0-91-generic-x86_64-with-debian-bullseye-sid
Is CUDA available: True
CUDA runtime version: 11.1.105
GPU models and configuration: 
GPU 0: GeForce GTX 1080 Ti
GPU 1: GeForce RTX 3090

Nvidia driver version: 460.84
cuDNN version: Probably one of the following:
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.1.1
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.1.1
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip3] numpy==1.19.5
[pip3] torch==1.10.1+cu113
[pip3] torchaudio==0.10.1+cu113
[pip3] torchvision==0.11.2+cu113
[conda] cudatoolkit               11.3.1               ha36c431_9    nvidia
[conda] numpy                     1.19.5                   pypi_0    pypi
[conda] torch                     1.10.1+cu113             pypi_0    pypi
[conda] torchaudio                0.10.1+cu113             pypi_0    pypi
[conda] torchvision               0.11.2+cu113             pypi_0    pypi
```