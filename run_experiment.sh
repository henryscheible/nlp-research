#!/usr/bin/env sh
docker_context="dsail3"
docker context use $docker_context
while read p; do
  echo "===============STARTING CONTAINER $p==============="
  docker image remove "$p:latest"
  docker build "./experiments/$p/" --build-arg TOKEN=$HF_TOKEN -t "$p:latest"
  docker run --gpus all -dit "$p:latest"
done < experiments.txt