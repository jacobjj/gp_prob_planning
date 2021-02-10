#!/bin/bash

# Stop the orchestra containers

for CID in {0..$2..1}
do
	docker stop data_$1_$CID
done