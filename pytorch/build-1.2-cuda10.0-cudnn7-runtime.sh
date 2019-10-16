#! /bin/bash

dir=$(dirname "$0")

docker build -t help-khidi-pytorch-1.2-cuda10.0-cudnn7-runtime -f $dir/Dockerfile-1.2-cuda10.0-cudnn7-runtime $dir