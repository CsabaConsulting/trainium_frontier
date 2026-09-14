#!/bin/bash
S=~/trainium_frontier/research/scratch/lnc_initmatrix.sh
bash $S lnc1-4p-nocompile      4 110 NEURON_LOGICAL_NC_CONFIG=1 -- --no-compile
bash $S lnc1-4p-noclipforeach  4 110 NEURON_LOGICAL_NC_CONFIG=1 -- --no-clip-foreach
bash $S lnc1-4p-gc0            4 110 NEURON_LOGICAL_NC_CONFIG=1 -- --grad-clip 0 --no-clip-foreach
echo PHASE2A_DONE > ~/lnc_logs/phase2a.done
bash ~/lnc_wsaxis.sh
