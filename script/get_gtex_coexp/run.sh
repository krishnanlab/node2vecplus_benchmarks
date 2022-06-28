#!/bin/bash

homedir=$(dirname $(realpath $0))
echo homedir=${homedir}
cd $homedir

mkdir -p ../../data/networks/gtexcoexp/downloads/
sbatch download.sb
