#!/bin/bash

#1) Create and activate environment
ENVS=$(conda info --envs | awk '{print $1}' )
if [[ $ENVS = *"CSI2PCloud"* ]]; then
   # shellcheck disable=SC1090
   source ~/anaconda3/etc/profile.d/conda.sh
   conda activate CSI2PCloud
else
   echo "Creating a new conda environment for CSI2PointCloud project..."
   conda env create -f environment.yml
   # shellcheck disable=SC1090
   source ~/anaconda3/etc/profile.d/conda.sh
   conda activate CSI2PCloud
   #python setup.py install
   #exit
fi;

