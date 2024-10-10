#!/bin/bash

set -e

DATA="${DATA:=/data}"

if [ ! -w $DATA ] ; then
    echo $DATA must exist and be writable
    exit 1
fi

NUMBERS="${NUMBERS:=4}"
SRATE="${SRATE:=500000}"

rm -rf $DATA/siggen
mkdir -p $DATA/siggen
mkdir -p $DATA/siggen/test
mkdir -p $DATA/model_store

git clone https://github.com/iqtlabs/gamutrf-deploy $DATA/gamutrf-deploy

# generate both training and test data
docker build -f utils/Dockerfile.siggen utils -t iqtlabs/rfml-siggen
for i in am fm ; do
    for o in /data/siggen/$i /data/siggen/test/$i ; do
        docker run --rm -v $DATA/siggen:/data/siggen -t iqtlabs/rfml-siggen /siggen.py --samp_rate $SRATE --siggen $i --int_count $NUMBERS --sample_file $o
    done
done

sudo chown -R $(id -u):$(id -g) $DATA/siggen

# label data and produce model
poetry run python3 label_scripts/label_siggen.py $DATA/siggen
poetry run python3 experiments/siggen_experiments.py $DATA/siggen
cp models/siggen_experiment.mar $DATA/model_store/torchsig_model.mar
