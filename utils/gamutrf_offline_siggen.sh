#!/bin/bash

set -e

PYTHON="${PYTHON:=python3}"
TORCHSERVE="${TORCHSERVE:=torchserve.yml}"
OFFLINE="${OFFLINE:=offline.yml}"
DATA="${DATA:=/data}"


docker compose -f $DATA/gamutrf-deploy/$OFFLINE down --remove-orphans
docker compose -f $DATA/gamutrf-deploy/$TORCHSERVE down --remove-orphans
VOL_PREFIX=$DATA docker compose -f $DATA/gamutrf-deploy/$TORCHSERVE up -d

for i in am fm ; do
    sudo rm -rf $DATA/samples
    VOL_PREFIX=$DATA RECORDING=/logs/siggen/test/$i.sigmf-meta docker compose -f $DATA/gamutrf-deploy/$OFFLINE up gamutrf
    echo $i
    $PYTHON ./utils/count_labels.py $DATA/samples/*/*sigmf-meta $i
done

VOL_PREFIX=$DATA docker compose -f $DATA/gamutrf-deploy/$TORCHSERVE down
