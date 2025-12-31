#!/bin/bash
xhost + local: &
echo $1
dockerName=$1
sudo docker run --rm -it \
  --env DISPLAY=$DISPLAY \
  --device=/dev/dri:/dev/dri \
  --device /dev/snd:/dev/snd \
  --volume "/tmp/.X11-unix/:/tmp/.X11-unix:rw" \
  --volume "/home/$USER/:/media/data" \
  --volume "/dev:/dev" \
  --volume ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
  --volume ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  --volume ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  --volume ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  --ipc host \
  --network=host \
  --cap-add=ALL \
  --privileged \
  --gpus all \
  $dockerName
