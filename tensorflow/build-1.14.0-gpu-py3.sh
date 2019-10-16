#! /bin/bash

dir=$(dirname "$0")

docker build -t help-khidi-tensorflow-1.14.0-gpu-py3 -f $dir/Dockerfile-1.14.0-gpu-py3 $dir