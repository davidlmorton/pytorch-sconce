# Setup Guide
This guide details the changes we need to make to a base Ubuntu 16.04 machine.
We will install cuda toolkit 9.1 and cudnn 7.0.5.

Cuda toolkit 9.1 only works with the 4.10 kernel that is default on 16.04.3.
Finally, we will install python 3.6.3 via pyenv to keep our stuff isolated from the system python.

## Cuda Toolkit 9.0 preinstallation checks:
```
$ lspci | grep -i nvidia
02:00.0 VGA compatible controller: NVIDIA Corporation GP106 [GeForce GTX 1060 6GB] (rev a1)
02:00.1 Audio device: NVIDIA Corporation GP106 High Definition Audio Controller (rev a1)
```

```
$ gcc --version
gcc (Ubuntu 5.4.0-6ubuntu1~16.04.5) 5.4.0 20160609
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

```
$ uname -r
4.10.0-28-generic
```

Ensure linux-headers for running kernel are installed.
```
$ dpkg -l | grep linux-headers
ii  linux-headers-4.10.0-28                    4.10.0-28.32~16.04.2                       all          Header files related to Linux kernel version 4.10.0
ii  linux-headers-4.10.0-28-generic            4.10.0-28.32~16.04.2                       amd64        Linux kernel headers for version 4.10.0 on 64 bit x86 SMP
```

Get the debian packages from nvidia.  Some of them require a login to the nvidia developer network.
```
$ ls -lah *.deb
-rw-rw-r-- 1 ubuntu ubuntu 2.8K Dec  1 19:29 cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
-rw-r--r-- 1 ubuntu ubuntu  66M Jan 15 10:43 libcudnn6_6.0.21-1+cuda8.0_amd64.deb
-rw-r--r-- 1 ubuntu ubuntu  58M Jan 15 10:43 libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb
-rw-r--r-- 1 ubuntu ubuntu  98M Jan 15 10:43 libcudnn7_7.0.3.11-1+cuda9.0_amd64.deb
-rw-r--r-- 1 ubuntu ubuntu  97M Jan 15 10:43 libcudnn7_7.0.5.15-1+cuda9.1_amd64.deb
-rw-r--r-- 1 ubuntu ubuntu  89M Jan 15 10:43 libcudnn7-dev_7.0.3.11-1+cuda9.0_amd64.deb
-rw-r--r-- 1 ubuntu ubuntu  88M Jan 15 10:43 libcudnn7-dev_7.0.5.15-1+cuda9.1_amd64.deb
-rw-r--r-- 1 ubuntu ubuntu 4.3M Jan 15 10:46 libcudnn7-doc_7.0.5.15-1+cuda9.1_amd64.deb
```

## Cuda  Installation
Install repository meta-data

```
$ sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
```

Installing the CUDA public GPG key

```
$ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
```

Install
```
$ sudo apt-get update
$ sudo apt-get install cuda
```

Failure, reboot and try again
```
$ nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
```
```
$ nvidia-smi
Tue Feb 27 20:43:53 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.30                 Driver Version: 390.30                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 106...  Off  | 00000000:02:00.0  On |                  N/A |
|  0%   20C    P8     5W / 156W |     71MiB /  6070MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1041      G   /usr/lib/xorg/Xorg                            69MiB |
+-----------------------------------------------------------------------------+
```

## Post installation steps

Add this to `~/.bashrc`:
```
export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### Ensure nvidia-persistenced is started on boot
As root, create this file
```
# cat nvidia-persistenced.service
[Unit]
Description=NVIDIA Persistence Daemon
Wants=syslog.target

[Service]
Type=forking
PIDFile=/var/run/nvidia-persistenced/nvidia-persistenced.pid
Restart=always
ExecStart=/usr/bin/nvidia-persistenced --verbose
ExecStopPost=/bin/rm -rf /var/run/nvidia-persistenced

[Install]
WantedBy=multi-user.target


Read more at: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ixzz54H6TQIIr
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook
```
Then issue this command
```
# systemctl enable nvidia-persistenced
```

Finally we need to comment out the line with `SUBSYSTEM=="memory"` as shown below:
```
# cat /lib/udev/rules.d/40-vm-hotadd.rules
# On Hyper-V and Xen Virtual Machines we want to add memory and cpus as soon as they appear
ATTR{[dmi/id]sys_vendor}=="Microsoft Corporation", ATTR{[dmi/id]product_name}=="Virtual Machine", GOTO="vm_hotadd_apply"
ATTR{[dmi/id]sys_vendor}=="Xen", GOTO="vm_hotadd_apply"
GOTO="vm_hotadd_end"

LABEL="vm_hotadd_apply"

# Memory hotadd request
#SUBSYSTEM=="memory", ACTION=="add", DEVPATH=="/devices/system/memory/memory[0-9]*", TEST=="state", ATTR{state}="online"

# CPU hotadd request
SUBSYSTEM=="cpu", ACTION=="add", DEVPATH=="/devices/system/cpu/cpu[0-9]*", TEST=="online", ATTR{online}="1"

LABEL="vm_hotadd_end"
```

## Verify setup of cuda toolkit
Install the samples in a directory with write permissions:
```
$ cuda-install-samples-9.0.sh ~/cuda-samples
```

Compile the samples
```
cd ~/cuda-samples/NVIDIA_CUDA-9.0_Samples
make
```

Run the `deviceQuery` sample and make sure it says it passed.
```
$ ./bin/x86_64/linux/release/deviceQuery
./bin/x86_64/linux/release/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 1060 6GB"
  CUDA Driver Version / Runtime Version          9.1 / 9.0
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 6070 MBytes (6364463104 bytes)
  (10) Multiprocessors, (128) CUDA Cores/MP:     1280 CUDA Cores
  GPU Max Clock rate:                            1848 MHz (1.85 GHz)
  Memory Clock rate:                             4004 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 1572864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 2 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.1, CUDA Runtime Version = 9.0, NumDevs = 1
Result = PASS
```

## Install cuDNN
### Get the debian packages
You'll need an account on the nvidia developer network.  Then you can find them [here](https://developer.nvidia.com/rdp/cudnn-download).
```
$ ls -lah *cudnn* | grep 9.1
-rw-r--r-- 1 ubuntu ubuntu  98M Jan 15 10:43 libcudnn7_7.0.3.11-1+cuda9.0_amd64.deb
-rw-r--r-- 1 ubuntu ubuntu  89M Jan 15 10:43 libcudnn7-dev_7.0.3.11-1+cuda9.0_amd64.deb
-rw-r--r-- 1 ubuntu ubuntu 4.3M Jan 15 16:02 libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb
```

Install using dpkg
```
$ ls *cudnn* | grep 9.1 | xargs -n 1 sudo dpkg -i
```

### Verify cuDNN Install
Copy the samples to a writable path, compile the samples and run the mnist sample.
```
cp -r /usr/src/cudnn_samples_v7/ $HOME
cd $HOME/cudnn_samples_v7/mnistCUDNN
make clean && make
```
Run the example, expect it to pass.
```
$ ./mnistCUDNN
cudnnGetVersion() : 5103 , CUDNN_VERSION from cudnn.h : 5103 (5.1.3)
Host compiler version : GCC 5.4.0
There are 1 CUDA capable devices on your machine :
device 0 : sms 10  Capabilities 6.1, SmClock 1847.5 Mhz, MemSize (Mb) 6069, MemClock 4004.0 Mhz, Ecc=0, boardGroupID=0
Using device 0

Testing single precision
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 1
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.022464 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.031744 time requiring 57600 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.052928 time requiring 100 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.065472 time requiring 205008 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.078464 time requiring 207360 memory
Resulting weights from Softmax:
0.0000000 0.9999399 0.0000000 0.0000000 0.0000561 0.0000000 0.0000012 0.0000017 0.0000010 0.0000000
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 0.9999288 0.0000000 0.0000711 0.0000000 0.0000000 0.0000000 0.0000000
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 0.9999820 0.0000154 0.0000000 0.0000012 0.0000006

Result of classification: 1 3 5

Test passed!

Testing half precision (math in single precision)
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 1
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.019040 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.030720 time requiring 100 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.031744 time requiring 28800 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.079872 time requiring 207360 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.084992 time requiring 205008 memory
Resulting weights from Softmax:
0.0000001 1.0000000 0.0000001 0.0000000 0.0000563 0.0000001 0.0000012 0.0000017 0.0000010 0.0000001
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 1.0000000 0.0000000 0.0000714 0.0000000 0.0000000 0.0000000 0.0000000
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 1.0000000 0.0000154 0.0000000 0.0000012 0.0000006

Result of classification: 1 3 5

Test passed!
```

## Install python 3.6.3 via pyenv

First we need to install some packages so we can compile a new python interpreter.
```
sudo apt-get install -y build-essential libncursesw5-dev libreadline6-dev \
    libssl-dev libgdbm-dev libc6-dev libsqlite3-dev libbz2-dev \
    libjpeg9 libjpeg9-dev libfreetype6 libfreetype6-dev zlib1g-dev \
    tk-dev
```

Now we download pyenv and update our login scripts so they setup the shims correctly.
```
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
```

Launch a new login shell, and then we can use pyenv to compile a new python version for us.
```
pyenv install 3.6.3
```

Finally we set this version of python to be the default interpreter whenever we're in this directory.
```
pyenv local 3.6.3
```

## Set up Jenkins (optional of course)

Use [this guide](https://www.digitalocean.com/community/tutorials/how-to-install-jenkins-on-ubuntu-16-04).
The steps are also copied below for postarity.

```
wget -q -O - https://pkg.jenkins.io/debian/jenkins-ci.org.key | sudo apt-key add -
echo deb https://pkg.jenkins.io/debian-stable binary/ | sudo tee /etc/apt/sources.list.d/jenkins.list
sudo apt-get update
sudo apt-get install jenkins
sudo systemctl start jenkins

# status should be active(exited)
sudo systemctl status jenkins

# service should be up at localhost:8080, connect and give password from:
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```

You'll also want to install pyenv as the jenkins user, just as we did for your normal user above.
