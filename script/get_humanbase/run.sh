#!/bin/bash

homedir=$(dirname $(realpath $0))
echo homedir=${homedir}
cd $homedir

mkdir -p ../../data/networks/humanbase/downloads
sbatch download.sb
