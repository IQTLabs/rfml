#!/bin/bash

set -e

docker compose -f /data/gamutrf-deploy/offline.yml down --remove-orphans
docker compose -f /data/gamutrf-deploy/torchserve-cuda.yml down --remove-orphans
VOL_PREFIX=/data docker compose -f /data/gamutrf-deploy/torchserve-cuda.yml up -d

for i in am fm ; do
    sudo rm -rf /data/samples
    VOL_PREFIX=/data RECORDING=/logs/siggen/test/$i.sigmf-meta docker compose -f /data/gamutrf-deploy/offline.yml up gamutrf
    echo $i
    ./utils/count_labels.py /data/samples/*/*sigmf-meta $i
done

VOL_PREFIX=/data docker compose -f /data/gamutrf-deploy/torchserve-cuda.yml down
