#!/usr/bin/env bash
# ~10 minute end-to-end validation: every optimizer takes 60 LM steps on
# tiny-shakespeare; loss must be finite and decreasing. Run after setup.sh.
set -e
python scripts/download_data.py --source shakespeare
OPTS="${OPTS:-sgdm adamw nadamw adafactor lion schedulefree prodigy sophia muon soap ademamix milo_m mion milo shampoo kron}"
for OPT in $OPTS; do
  echo "=== smoke: $OPT"
  python tasks/lm_pretrain.py --data shakespeare --model small --optimizer "$OPT" \
    --lr 0.001 --steps 60 --batch-size 4 --accum 1 --ctx 256 \
    --eval-every 60 --out-dir results/smoke \
    || echo "!!! $OPT FAILED (missing package or original milo.py?)"
done
python analysis/aggregate.py results/smoke || true
