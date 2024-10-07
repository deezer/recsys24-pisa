#!/bin/bash

echo "CUDA_VISIBLE_DEVICES=0 python data_misc/lfm1b/05_pretrain_usertrack_embeddings.py"
CUDA_VISIBLE_DEVICES=0 python data_misc/lfm1b/05_pretrain_usertrack_embeddings.py
