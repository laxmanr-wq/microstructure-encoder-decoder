#!/usr/bin/env bash
# Resume entry point: run Phase 4 calibration (unless its report exists), check its gate,
# smoke-test every downstream entry point against the calibration checkpoint (so a bug is
# found and fixed while Phase 5 trains, not after it), then launch Phase 5.
# Stops at the first failure. Refuses to start below 40% free memory (FORCE=1 overrides).
set -u
ROOT="/Volumes/Hard Disk/Microstructure Encoder Decoder"
cd "$ROOT" || exit 1
exec > >(tee -a "$ROOT/reports/gap_phase5_chain.log") 2>&1
PY="$ROOT/.venv/bin/python"
SCR="/private/tmp/claude-501/-Volumes-Hard-Disk-Microstructure-Encoder-Decoder/7fbd8c6c-9b22-4828-88f3-8b177ceddc1a/scratchpad/smoke"
CK="checkpoints/calib_best.pt"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$SCR"

say()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { say "FAIL: $*"; exit 1; }

# Training at batch 8 needs ~2.7 GB; below ~40% free this machine thrashes (measured
# 1.9 -> 0.29 it/s) and every hour of Phase 5 is wasted.
free=$(memory_pressure | awk -F": " '/free percentage/{gsub("%","",$2); print int($2)}')
say "memory free: ${free}%"
if [ "${free:-0}" -lt 40 ] && [ "${FORCE:-0}" != 1 ]; then
  fail "only ${free}% memory free - close Chrome and other heavy apps first (FORCE=1 to override)"
fi

if [ ! -f reports/phase4_calib.json ]; then
  if pgrep -qf "src/train.py --steps 2000 --tag cali[b]"; then
    say "phase 4 already running - waiting for it to finish"
    while pgrep -qf "src/train.py --steps 2000 --tag cali[b]"; do sleep 10; done
  else
    say "phase 4: 2000-step calibration at batch 8"
    "$PY" src/train.py --steps 2000 --tag calib --report reports/phase4_calib.json \
        > reports/phase4_calib.log 2>&1 || fail "phase 4 exited non-zero (see reports/phase4_calib.log)"
  fi
fi

"$PY" - << 'PYCHK' || fail "phase 4 gate did not pass (see reports/phase4_calib.json)"
import json, sys
r = json.load(open("reports/phase4_calib.json"))
g = r.get("gate", {})
print("phase4 gate:", g, "| 5h budget:", r.get("recommended_budget", {}).get("steps_5h"))
sys.exit(0 if g.get("PASS") else 1)
PYCHK
[ -f "$CK" ] || fail "missing $CK"

IMG=$("$PY" -c "
import csv
for r in csv.DictReader(open('manifests/manifest.csv')):
    if r['split'] == 'test' and r['keep'] == '1':
        print(r['path']); break")
say "S1 infer.py on $(basename "$IMG")"
"$PY" src/infer.py "$IMG" -o "$SCR/infer" --ckpt "$CK" > "$SCR/s1.log" 2>&1 || { tail -6 "$SCR/s1.log"; fail "infer.py"; }
ls "$SCR"/infer/*_recon.png "$SCR"/infer/*_latent.npy "$SCR"/infer/*_metrics.json > /dev/null || fail "infer outputs missing"
grep -hE '"psnr_db"|"ratio"' "$SCR"/infer/*_metrics.json

say "S2 eval.py --limit 3 at full resolution"
"$PY" src/eval.py --ckpt "$CK" --limit 3 --desc-per-band 2 --out "$SCR/eval_smoke.json" \
    --grid "$SCR/grid_smoke.png" --desc-npz "$SCR/desc_smoke.npz" > "$SCR/s2.log" 2>&1 || { tail -8 "$SCR/s2.log"; fail "eval.py"; }
grep -E "^ +all|allocation failed" "$SCR/s2.log" | tail -2

say "S3 descriptors.py (MCRpy) on the smoke sample"
"$PY" scripts/descriptors.py --npz "$SCR/desc_smoke.npz" --eval "$SCR/eval_smoke.json" --per-band 1 \
    > "$SCR/s3.log" 2>&1 || { tail -8 "$SCR/s3.log"; fail "descriptors.py"; }
grep -E "^  band [0-9]:|overall descriptor" "$SCR/s3.log" | tail -4

say "S4 train_gan.py, 20 steps (adversarial term from step 10)"
"$PY" src/train_gan.py --init "$CK" --steps 20 --d-warmup 10 --val-every 10 --report "$SCR/gan_smoke.json" \
    > "$SCR/s4.log" 2>&1 || { tail -8 "$SCR/s4.log"; fail "train_gan.py"; }
grep -E "VAL|done:" "$SCR/s4.log" | tail -3
rm -f checkpoints/gan_best.pt checkpoints/gan_last.pt checkpoints/gan_*.tmp checkpoints/._gan_*

say "S5 app.py starts and serves (Gradio)"
MSAE_CKPT="$CK" PORT=7861 "$PY" src/app.py > "$SCR/s5.log" 2>&1 &
APP=$!
ok=0
for _ in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7861/ || true)
  [ "$code" = "200" ] && { ok=1; break; }
  kill -0 "$APP" 2>/dev/null || break
  sleep 2
done
kill "$APP" 2>/dev/null; wait "$APP" 2>/dev/null
[ "$ok" = 1 ] || { tail -8 "$SCR/s5.log"; fail "app.py did not serve HTTP 200"; }
say "app.py served HTTP 200 on :7861"

say "all smoke tests passed - launching phase 5"
"$PY" scripts/phase5_run.py
say "phase 5 orchestrator exited with $?"
