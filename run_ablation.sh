#!/usr/bin/env bash
# Usage: bash run_ablation.sh /path/to/PlantVillage/color [/path/to/field_images]
# Optional environment variables:
#   SEED=1          random seed (run seeds 42, 1, 2 for the paper)
#   OODMAP=map.json maps field-folder names to PlantVillage class names
#   EXTRA="--max_per_class 300 --epochs_head 4 --epochs_ft 4"   faster, reduced runs
#                   (delete runs/splits.json if you change EXTRA; state any reduction in the paper)
DATA=$1
OOD=${2:-}
SEED=${SEED:-42}
OODARG=""
[ -n "$OOD" ] && OODARG="--ood_dir $OOD"
[ -n "${OODMAP:-}" ] && OODARG="$OODARG --ood_map $OODMAP"
C="python train.py --data_dir $DATA --seed $SEED $OODARG ${EXTRA:-}"

$C --run_name A0_cnn_s$SEED          --model cnn
$C --run_name A1_mnv2_s$SEED         --model transfer
$C --run_name A2_cbam_s$SEED         --model transfer --cbam
$C --run_name A3_cbam_sbr_s$SEED     --model transfer --cbam --sbr
$C --run_name A4_cbam_sbr_mga_s$SEED --model transfer --cbam --sbr --attn_sup
$C --run_name A5_full_s$SEED         --model transfer --cbam --sbr --attn_sup --hier
