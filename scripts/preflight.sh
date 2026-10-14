#!/usr/bin/env bash
# preflight.sh — read-only checks before Phase 6. Modifies nothing.
# Run:  bash scripts/preflight.sh
# Gate: memory free >= 40%  (RESUME.md blocker, 17 Sep 2026)

set -u

FREE=$(memory_pressure 2>/dev/null | grep -i "free percentage" | grep -oE "[0-9]+" | head -1)
SWAP=$(sysctl -n vm.swapusage 2>/dev/null)
IMAGES=$(find ~/ms_cache/dataset -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.npy" \) 2>/dev/null | wc -l | tr -d ' ')

echo "== Phase 6 preflight =="
echo "Memory free : ${FREE:-?}%   (need >= 40)"
echo "Swap        : $SWAP"
echo "Dataset     : $IMAGES images  (expect 10569)"
echo

if [ -z "${FREE:-}" ]; then
  echo "FAIL: could not read memory_pressure"
  exit 1
fi

if [ "$FREE" -lt 40 ]; then
  echo "BLOCKED: free ${FREE}% < 40%. Close apps (keep only VS Code) or reboot, then rerun."
  ps axo rss,comm | awk 'NR>1 {n=split($2,a,"/"); printf "%8.0f MB  %s\n", $1/1024, a[n]}' | sort -rn | head -8
  exit 2
fi

if [ "$IMAGES" -ne 10569 ]; then
  echo "WARN: dataset count $IMAGES != 10569 — check ~/ms_cache/dataset / external drive mount"
  exit 3
fi

echo "PASS: memory and dataset OK. Run Phase 6:"
echo '  source .venv/bin/activate'
echo '  export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1'
echo '  python -u src/eval.py --ckpt checkpoints/full_best.pt --out reports/phase6_eval.json --grid reports/phase6_grid.png --desc-npz reports/phase6_desc_sample.npz'
echo '  python scripts/descriptors.py --npz reports/phase6_desc_sample.npz --eval reports/phase6_eval.json'
