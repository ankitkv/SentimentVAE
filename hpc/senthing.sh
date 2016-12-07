#!/bin/bash
module purge
module load tensorflow/python3.5.1/20161029

cd /scratch/vnb222/code/SentimentVAE
python3 -u main.py $DL_ARGS
