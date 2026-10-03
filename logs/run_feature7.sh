#!/bin/bash
cd /home/ravindu/rashmika/pathoasisst
while [ ! -f training/data/mitos_atypia/index.json ]; do
  ps -eo args | awk '$1 ~ /python$/ && /prepare_mitos_atypia/' | grep -q . || break
  sleep 30
done
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
~/miniconda3/envs/pathoassist/bin/python training/pleomorphism/train_pleomorphism.py --workers 8 > logs/train_pleomorphism.log 2>&1
echo done > logs/feature7.done
