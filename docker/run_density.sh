#!/bin/bash
# A script to collect density experiment

for CID in {0..5..1}
do
	docker run -d \
	    -v /home/jacoblab/prob_planning/:/root/prob_planning \
	    -v /home/jacoblab/prob_planning_data:/root/data \
	    ompl:opengl-bionic \
	    python3 rrt_motion_density.py\
	    	--envNum=$CID\
	    	--numObstacles=$((5+CID*2))\
	    	--seed=$((CID+3))
done