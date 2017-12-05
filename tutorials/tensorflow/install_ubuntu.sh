#!/usr/bin/env bash

## Building tensorflow # with GPU support ## with XLA support
## Ubuntu ## gcc > 5 ## using conda environment with python 3.5.4
# CUDA and cuDNN installed in separate folders.

# Add cuDNN to environment variables (not sure if all is needed):
export PATH="/home/notoraptor/.local/cudnn-7004-cuda-9/cuda/lib64:/home/notoraptor/Programmes/bin:$PATH"
export CPATH="/home/notoraptor/.local/cudnn-7004-cuda-9/cuda/include:/home/notoraptor/.local/include:$CPATH"
export LIBRARY_PATH="/home/notoraptor/.local/cudnn-7004-cuda-9/cuda/lib64:/home/notoraptor/.local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/notoraptor/.local/cudnn-7004-cuda-9/cuda/lib64:/home/notoraptor/.local/lib:$LD_LIBRARY_PATH"

# git clone https://github.com/tensorflow/tensorflow 
# cd tensorflow
# git checkout r1.4
# Install bazel: https://docs.bazel.build/versions/master/install.html

sudo apt-get install libcupti-dev
sudo apt-get install libibverbs-dev
sudo apt-get install librdmacm-dev

conda install numpy wheel

bazel clean
./configure
bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# for profiler
bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures //tensorflow/core/profiler:profiler

# Python package is in /tmp/tensorflow_pkg/ .
pip install /tmp/tensorflow_pkg/tensorflow-1.4.0-py2-none-any.whl


# nvprof python nv_f16_example.py --dtype float16 --nin 4096 --nbatch 4096 --nout 4096
# - Modify le script pour tester float16 et float32 a chaque appel.
# - sinter --mem=16000 --qos=high --gres=gpu:v100
# - http://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#example_tensorflow
# - Compile tensorflow master (J'ai testé tensorflow 1.4 from github) DOit être compiler from scratch on the computer with v100 avec cuda9 et cudnn7.
# - Si cela ne marche pas, esseyé leur docker sur la machine v100.
#   - activé docker sur kepler5, te donner les droits pour docker, apprendre docker, ...
# sinter --reservation=<reservation_name>
# sinter --reservation=lefransi_34
# sinter --mem=16000 --qos=high --gres=gpu:v100 --reservation=lefransi_34
