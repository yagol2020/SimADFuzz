#!/bin/bash
if [ -z "$1" ]; then
  echo "请提供端口号作为参数，例如：mysh.sh 12345"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Open carla display?"
  exit 1
fi

port=$1
carla_name=carla-sim-$1
docker rm -f "$carla_name"

idle_gpu=${CUDA_VISIBLE_DEVICES:-0}


if [ "$2" == "True" ]; then
  # 启用显示
  carla_cmd="./CarlaUE4.sh -carla-rpc-port=$port -quality-level=Epic -ResX=1920 -ResY=1080 && /bin/bash"
  docker run --name=$carla_name \
    -d \
    --gpus "device=$idle_gpu" \
    --net=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    carlasim/carla:0.9.13 \
    $carla_cmd
else
  # 不启用显示
  carla_cmd="./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=$port -quality-level=Epic && /bin/bash"
  docker run --name=$carla_name \
    -d \
    --gpus "device=$idle_gpu" \
    --net=host \
    carlasim/carla:0.9.13 \
    $carla_cmd
fi