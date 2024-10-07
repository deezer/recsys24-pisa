#!/bin/bash

mkdir -p exp/logs/deezer_min250sess/pisa exp/logs/lfm1b_min50sess/pisa

#####################################################################################
# DEEZER
#####################################################################################

# training
# "negsam_strategy" = 0 for uniform and 1 for popularity-based sampling
echo "CUDA_VISIBLE_DEVICES=0 python -m pisa train --verbose -p configs/deezer.json"
CUDA_VISIBLE_DEVICES=0 python -m pisa train --verbose -p configs/deezer.json

# evaluation
CUDA_VISIBLE_DEVICES=0 python -m pisa eval --verbose -p configs/deezer.json


#####################################################################################
# LFM1B
#####################################################################################
# training
echo "CUDA_VISIBLE_DEVICES=0 python -m pisa train --verbose -p configs/lfm1b.json"
CUDA_VISIBLE_DEVICES=0 python -m pisa train --verbose -p configs/lfm1b.json

# evaluation
CUDA_VISIBLE_DEVICES=0 python -m pisa eval --verbose -p configs/lfm1b.json
