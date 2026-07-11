#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
j1=$(qsub cats_dogs/qsub_templates/01_generate.pbs)
j2=$(qsub -W depend=afterok:${j1} cats_dogs/qsub_templates/02_postprocess.pbs)
j3=$(qsub -W depend=afterok:${j2} cats_dogs/qsub_templates/03_extract_vectors.pbs)
j4=$(qsub -W depend=afterok:${j3} cats_dogs/qsub_templates/04_greedy_ablation.pbs)
echo "submitted: gen=${j1} post=${j2} vec=${j3} ablate=${j4}"
