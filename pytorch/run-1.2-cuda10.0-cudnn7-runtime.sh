#! /bin/bash

function abs_path {
  (cd $(dirname $1) &>/dev/null && printf "%s/%s" "$PWD" "$(basename $1)")
}

dir=$(dirname "$0")
data_dir=$(abs_path $dir/../data)

docker run -e ID=pytorch-12cuda10-cudnn7-runtime \
  -v $data_dir:/data \
  -it --rm help-khidi-pytorch-1.2-cuda10.0-cudnn7-runtime \
  sh -c './train.sh; ./inference.sh'