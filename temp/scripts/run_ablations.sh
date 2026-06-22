#!/usr/bin/env bash
# Mion / MiloM component ablations on the small LM (0.4B tokens, 2 seeds).
set -e
BASE="python tasks/lm_pretrain.py --data fineweb --model small --tokens 4e8 \
  --batch-size 16 --accum 2 --out-dir results/ablations"
for SEED in 1 2; do
  # -- Mion: Newton-Schulz iteration count (speed/quality tradeoff)
  for NS in 1 3 5; do
    $BASE --optimizer mion --lr 0.003 --seed $SEED --opt-kwargs "{\"ns_steps\":$NS}"
  done
  # -- Mion: target update RMS
  for RMS in 0.1 0.2 0.4; do
    $BASE --optimizer mion --lr 0.003 --seed $SEED --opt-kwargs "{\"rms_target\":$RMS}"
  done
  # -- Mion: grafting blend on the group path
  for SF in 0.0 0.2 0.5; do
    $BASE --optimizer mion --lr 0.003 --seed $SEED --opt-kwargs "{\"scale_factor\":$SF}"
  done
  # -- MiloM: structure-aligned vs flat (original) grouping
  for GM in row flat; do
    $BASE --optimizer milo_m --lr 0.003 --seed $SEED --opt-kwargs "{\"group_mode\":\"$GM\"}"
  done
  # -- MiloM: blend strength
  for SF in 0.0 0.2 0.5; do
    $BASE --optimizer milo_m --lr 0.003 --seed $SEED --opt-kwargs "{\"scale_factor\":$SF}"
  done
done
