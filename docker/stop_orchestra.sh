#!/bin/bash

# Stop the orchestra containers

for CID in {0..7..1}
do
	docker stop data_$1_$CID
done