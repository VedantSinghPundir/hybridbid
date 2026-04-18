#!/bin/bash
set -e
cd ~/hybridbid
LOG=logs/eval_sweep_stage2_v3a.log
echo "=== Stage 2 v3a Eval Sweep ===" | tee $LOG
date | tee -a $LOG

for step in 10000 20000 30000 50000 80000 100000 120000 150000 175000 200000; do
    ckpt="checkpoints/stage2_v3a/checkpoint_step${step}.pt"
    if [ ! -f "$ckpt" ]; then
        echo "--- step $step --- SKIPPED (no checkpoint)" | tee -a $LOG
        continue
    fi
    echo "--- step $step ---" | tee -a $LOG
    conda run --no-capture-output -n hybridbid python -u -m src.evaluation.evaluate_stage2 \
        --checkpoint "$ckpt" --device cuda --v3a 2>&1 | tee -a $LOG
done

echo "=== Sweep complete ===" | tee -a $LOG
