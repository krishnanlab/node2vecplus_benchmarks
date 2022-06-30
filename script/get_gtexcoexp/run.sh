#!/bin/bash

homedir=$(dirname $(realpath $0))
echo homedir=${homedir}
cd $homedir

mkdir -p ../../data/networks/ppi/gtexcoexp/downloads/
sbatch download.sb
