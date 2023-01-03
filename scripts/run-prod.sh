#!/usr/bin/env bash

NAME=wav2lip

set -ex

docker build . -t $NAME
docker run -it --rm \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -p 5001:5000 \
  --gpus all \
  $NAME
