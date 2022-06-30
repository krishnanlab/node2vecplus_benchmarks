#!/bin/bash

homedir=$(dirname $(realpath $0))
echo homedir=${homedir}
cd $homedir

mkdir -p ../../data/networks/ppi/humanbase/downloads
sbatch download.sb
