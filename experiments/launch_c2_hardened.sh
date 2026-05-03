#!/usr/bin/env bash
# Launch C2 hardened across 6Q × 6 GPUs in parallel.
# Run on chen server after P1 completes (GPUs 2-7 free again).
set -u
LOGDIR=/data/chenhongrui/business/logs
cd /data/chenhongrui/business/experiments
source /data/anaconda3/etc/profile.d/conda.sh
conda activate qiqi_rl_gpu
GPUS=(2 3 4 5 6 7)
for i in 0 1 2 3 4 5; do
  G=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$G nohup python -u dump_c2_hardened.py \
    --question_idx $i --device cuda:0 \
    > $LOGDIR/c2_hard_q${i}.log 2>&1 &
  echo "launched C2-hardened q=$i on GPU=$G pid=$!"
done
sleep 2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
