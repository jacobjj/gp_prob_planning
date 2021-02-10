#!/bin/bash
# A script to run ompl-docker
SAMPLES=5
for CID in {0..7..1}
do
	docker run -d \
	    --rm \
	    --name=data_$1_$CID \
	    --shm-size="2g"\
	    -e DISPLAY=$DISPLAY \
	    -e QT_X11_NO_MITSHM=1 \
	    -v $XAUTH:/root/.Xauthority \
	    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	    -v /home/jacoblab/prob_planning/:/root/prob_planning \
	    -v /home/jacoblab/prob_planning_data:/root/data \
	    ompl-pybullet \
	    python3 rrt_motion_dubin.py $((CID*SAMPLES)) $SAMPLES
done