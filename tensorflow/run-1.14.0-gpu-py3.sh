#! /bin/bash

function abs_path {
  (cd $(dirname $1) &>/dev/null && printf "%s/%s" "$PWD" "$(basename $1)")
}

dir=$(dirname "$0")
data_dir=$(abs_path $dir/../data)

docker run -e ID=tensorflow-1140-gpu-py3 \
  -v $data_dir:/data \
  -it --rm help-khidi-tensorflow-1.14.0-gpu-py3 \
  sh -c './train.sh; ./inference.sh'