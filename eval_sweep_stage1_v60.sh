#!/bin/bash
set -e
cd ~/hybridbid
LOG=logs/eval_sweep_stage1_v60.log
echo "=== Stage 1 v6.0 Eval Sweep ===" | tee $LOG
date | tee -a $LOG

for step in 100000 150000 200000 250000 300000 350000 400000 450000 500000 600000 700000 800000 900000 1000000; do
    ckpt="checkpoints/stage1_v60/checkpoint_step${step}.pt"
    echo "--- step $step ---" | tee -a $LOG
    conda run --no-capture-output -n hybridbid python -u -m src.evaluation.evaluate_stage1 \
        --checkpoint "$ckpt" --device cuda --v60 2>&1 | tee -a $LOG
done

echo "=== Sweep complete ===" | tee -a $LOG
