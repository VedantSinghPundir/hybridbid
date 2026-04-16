#!/bin/bash
set -e
cd ~/hybridbid
LOG=logs/eval_sweep_stage2_scratch.log
echo "=== Stage 2 Scratch Eval Sweep ===" | tee $LOG
date | tee -a $LOG

for step in 10000 20000 30000 48000 60000 75000 90000 105000 120000; do
    ckpt="checkpoints/stage2_scratch/checkpoint_step${step}.pt"
    echo "--- step $step ---" | tee -a $LOG
    conda run --no-capture-output -n hybridbid python -u -m src.evaluation.evaluate_stage2 --checkpoint "$ckpt" --device cuda 2>&1 | tee -a $LOG
done

echo "" | tee -a $LOG
echo "--- Stage 1 baseline (300k on post-RTC+B) ---" | tee -a $LOG
conda run --no-capture-output -n hybridbid python -u -m src.evaluation.evaluate_stage2 --stage1-baseline --device cuda 2>&1 | tee -a $LOG

echo "=== Sweep complete ===" | tee -a $LOG
