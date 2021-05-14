# A script to run ompl-docker

docker run -it \
    -p $1:8800 \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v $XAUTH:/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /home/jacoblab/prob_planning/:/root/prob_planning \
    -v /home/jacoblab/prob_planning_data:/root/data \
    ompl:opengl-bionic \
    bash
