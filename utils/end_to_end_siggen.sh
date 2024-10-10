#!/bin/bash

# end to end test that verifies 99% classification on AM/FM signals

set -e

if [ "$0" != "./utils/end_to_end_siggen.sh" ] ; then
    echo must be called as ./utils/end_to_end_siggen.sh
    exit 1
fi

./utils/run_siggen.sh
./utils/gamutrf_offline_siggen.sh
