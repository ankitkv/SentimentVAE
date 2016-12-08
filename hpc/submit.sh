#!/bin/bash
export JOB_NAME=$1
shift
export DL_ARGS=$@

echo "Job name = $JOB_NAME"
echo "Arguments = $DL_ARGS"

USER_NAME=$(whoami)

qsub -N $JOB_NAME -v DL_ARGS -l nodes=1:ppn=2:gpus=1,walltime=47:59:00,pmem=8GB -m abe -M "$USER_NAME@nyu.edu" senthing.sh -o ./logs/$JOB_NAME.out -e ./logs/$JOB_NAME.err
