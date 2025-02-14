#!/bin/bash

mkdir -p exp/logs/deezer_min250sess/pisa_art exp/logs/lfm1b_min50sess/pisa_art

######################################################################################
# DEEZER
######################################################################################
echo "CUDA_VISIBLE_DEVICES=0 python -m pisa train --verbose -p configs/pisa_art/deezer.json"
CUDA_VISIBLE_DEVICES=0 python -m pisa train --verbose -p configs/pisa_art/deezer.json

# evaluation
CUDA_VISIBLE_DEVICES=0 python -m pisa eval --verbose -p configs/pisa_art/deezer.json

#####################################################################################
# LFM1B
#####################################################################################
# training
echo "CUDA_VISIBLE_DEVICES=0 python -m pisa train --verbose -p configs/pisa_art/lfm1b.json"
CUDA_VISIBLE_DEVICES=0 python -m pisa train --verbose -p configs/pisa_art/lfm1b.json

# evaluation
CUDA_VISIBLE_DEVICES=0 python -m pisa eval --verbose -p configs/pisa_art/lfm1b.json
