xhost +local:root
docker run -it --rm \
    --name sac_gymnasium_container \
    --net=host \
    --ipc=host \
    --privileged \
    --device /dev/dri:/dev/dri \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v "$(pwd)":/sac-gymnasium \
    sac-gymnasium:latest