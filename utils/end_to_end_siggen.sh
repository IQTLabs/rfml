#!/bin/bash

# end to end test that verifies 99% classification on AM/FM signals

set -e

if [ "$0" != "./utils/end_to_end_siggen.sh" ] ; then
    echo must be called as ./utils/end_to_end_siggen.sh
    exit 1
fi

if [ ! -w /data ] ; then
    echo /data must exist and be writable
    exit 1
fi

rm -rf /data/siggen
mkdir -p /data/siggen
mkdir -p /data/siggen/test
mkdir -p /data/model_store

git clone https://github.com/iqtlabs/gamutrf-deploy /data/gamutrf-deploy

# generate both training and test data
docker build -f utils/Dockerfile.siggen utils -t iqtlabs/rfml-siggen
for i in am fm ; do
    for o in /data/siggen/$i /data/siggen/test/$i ; do
        docker run -v /data/siggen:/data/siggen -u $(id -u ${USER}):$(id -g ${USER}) -t iqtlabs/rfml-siggen /siggen.py --samp_rate 1000000 --siggen $i --int_count 100 --sample_file $o
    done
done

# label data and produce model
python label_scripts/label_siggen.py
python experiments/siggen_experiments.py
cp models/siggen_experiment.mar /data/model_store/torchsig_model.mar

# run inference
./utils/gamutrf_offline_siggen.sh
