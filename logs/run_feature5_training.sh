#!/bin/bash
# Wait for the BACH extraction workers, merge their indexes, then train the Feature 5 models.
cd /home/ravindu/rashmika/pathoasisst
P=~/venvs/pathoassist-torch/bin/python
while ps -eo args | awk '$1 ~ /bin\/python$/ && /prepare_bach/' | grep -q .; do sleep 30; done
$P training/region/prepare_bach.py --merge > logs/bach_merge.log 2>&1
CUDA_VISIBLE_DEVICES=1 $P training/region/train_magnification.py > logs/train_magnification.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $P training/region/train_region.py > logs/train_region.log 2>&1
wait
echo done > logs/feature5_training.done
