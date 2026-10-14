"""Phase 5 — full training run.

Resumes from the Phase 4 checkpoint and trains to the 5-hour step budget that
phase4_calib.json computed from measured throughput. Early on it deliberately
SIGTERMs the run and restarts it from the saved checkpoint, so resume is proven
on the real run rather than assumed. Orchestrates train.py in subprocesses and
never imports torch itself (8 GB machine)."""
import os, sys, json, time, signal, subprocess, re

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
REP = os.path.join(ROOT, "reports")
CKPT = os.path.join(ROOT, "checkpoints")
TRAIN = [sys.executable, os.path.join(ROOT, "src", "train.py")]
VALCK = [sys.executable, os.path.join(ROOT, "scripts", "val_ckpt.py")]
ENV = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
KILL_AFTER = 150            # steps past the resume point before the deliberate kill
CKPT_EVERY = 500            # BRIEF: the drive may be unplugged mid-run, checkpoint often
STEP_RE = re.compile(r"^step +(\d+)/\d+ loss ([0-9.]+)")


def logged_steps(path):
    out = []
    try:
        text = open(path, errors="ignore").read().replace("\r", "\n")
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        m = STEP_RE.match(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def run_segment(args, log_path, kill_at=None):
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    t0, killed = time.time(), False
    with open(log_path, "w") as log:
        p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
        while p.poll() is None:
            time.sleep(5)
            if kill_at and not killed:
                st = logged_steps(log_path)
                if st and st[-1][0] >= kill_at:
                    p.send_signal(signal.SIGTERM)
                    killed = True
                    print(f"  sent SIGTERM at logged step {st[-1][0]}", flush=True)
    return p.returncode, time.time() - t0, killed


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    p4 = json.load(open(os.path.join(REP, "phase4_calib.json")))
    budget = int(p4["recommended_budget"]["steps_5h"])
    start = int(p4["steps_done"])
    calib_last = p4["checkpoints"]["last"]
    # score step 2000 with EMA and raw weights; the gate must beat the better of the two
    r = subprocess.run(VALCK + [calib_last], capture_output=True, text=True, env=ENV)
    m = re.search(r"^VALJSON (.*)$", r.stdout, re.M)
    v2000 = json.loads(m.group(1)) if m else {"ema": p4["val_final"], "raw": None}
    cands = [v2000["ema"]["lpips"]] + ([v2000["raw"]["lpips"]] if v2000.get("raw") else [])
    base_lpips = float(min(cands))
    print(f"step-{start} val  EMA: {v2000['ema']}", flush=True)
    print(f"step-{start} val  raw: {v2000.get('raw')}", flush=True)
    full_last = os.path.join(CKPT, "full_last.pt")
    print(f"phase 5: resume {calib_last} at step {start}, train to {budget} "
          f"(5h budget from phase4, sustained {p4['recommended_budget']['sustained_it_s']} it/s); "
          f"gate baseline LPIPS {base_lpips:.4f} (better of EMA/raw)", flush=True)

    # ---- segment A: resume from calibration, then deliberately kill early ----
    logA, jsonA = os.path.join(REP, "phase5_segA.log"), os.path.join(REP, "phase5_segA.json")
    rcA, wallA, killed = run_segment(
        TRAIN + ["--steps", str(budget), "--tag", "full", "--resume", calib_last,
                 "--ckpt-every", str(CKPT_EVERY), "--report", jsonA],
        logA, kill_at=start + KILL_AFTER)
    textA = open(logA, errors="ignore").read()
    m = re.search(r"stopping early at step (\d+)", textA)
    killed_at = int(m.group(1)) if m else None
    print(f"segment A: rc={rc_str(rcA)} killed={killed} checkpoint step={killed_at} "
          f"({wallA/60:.1f} min)", flush=True)
    if not (killed and killed_at and os.path.exists(full_last)):
        print("RESUME TEST FAILED: segment A did not stop cleanly with a checkpoint", flush=True)
        sys.exit(1)

    # ---- segment B: restart from the checkpoint the kill left behind ----
    logB, jsonB = os.path.join(REP, "phase5_segB.log"), os.path.join(REP, "phase5_segB.json")
    rcB, wallB, _ = run_segment(
        TRAIN + ["--steps", str(budget), "--tag", "full", "--resume", full_last,
                 "--ckpt-every", str(CKPT_EVERY), "--report", jsonB],
        logB)
    textB = open(logB, errors="ignore").read()
    m = re.search(r"resumed .* at step (\d+)", textB)
    resumed_at = int(m.group(1)) if m else None

    la, lb = logged_steps(logA), logged_steps(logB)
    loss_before = mean([l for _, l in la[-3:]])
    loss_after = mean([l for _, l in lb[:3]])
    # a resume that silently failed restarts from init-level loss (~1.4); a real one continues
    continuity = loss_after < 1.5 * loss_before
    resume_ok = bool(killed_at is not None and resumed_at == killed_at and continuity)

    B = json.load(open(jsonB))
    A = json.load(open(jsonA)) if os.path.exists(jsonA) else {}
    vals = [h["val"] for h in B.get("history", []) if "val" in h and h["val"].get("step", 0) > start]
    best = min(vals, key=lambda v: v["lpips"]) if vals else None
    gate = {"val_lpips_better_than_step2000": bool(best and best["lpips"] < base_lpips),
            "resume_verified": resume_ok}
    gate["PASS"] = all(gate.values())

    rep = {
        "phase": 5, "budget_steps": budget, "start_step": start,
        "steps_done": B.get("steps_done"), "reached_budget": B.get("steps_done") == budget,
        "wall_clock_s": round(wallA + wallB, 1), "wall_clock_h": round((wallA + wallB) / 3600, 3),
        "segments": {"A": {"wall_s": round(wallA, 1), "rc": rcA, "killed_at": killed_at},
                     "B": {"wall_s": round(wallB, 1), "rc": rcB, "resumed_at": resumed_at}},
        "resume_verification": {"killed_at_step": killed_at, "resumed_at_step": resumed_at,
                                "loss_before_kill": round(loss_before, 5),
                                "loss_after_resume": round(loss_after, 5),
                                "continuity_ok": continuity, "verified": resume_ok},
        "step2000_val": v2000, "gate_baseline_lpips": base_lpips,
        "best_val": best, "val_final": B.get("val_final"),
        "peak_gb": max(A.get("peak_gb", 0) or 0, B.get("peak_gb", 0) or 0),
        "throttle_profile": (A.get("throttle_profile") or []) + (B.get("throttle_profile") or []),
        "it_s_first10": B.get("it_s_first10"), "it_s_last10": B.get("it_s_last10"),
        "throttle_pct": B.get("throttle_pct"),
        "val_curve": [h["val"] for h in B.get("history", []) if "val" in h],
        "train_curve": [{k: h[k] for k in ("step", "loss", "l1", "lpips", "ms_ssim", "it_s")}
                        for h in B.get("history", []) if "loss" in h],
        "checkpoints": B.get("checkpoints"), "gate": gate,
    }
    json.dump(rep, open(os.path.join(REP, "phase5_train.json"), "w"), indent=2)
    print(json.dumps({"resume": rep["resume_verification"], "best_val": best,
                      "wall_h": rep["wall_clock_h"], "steps_done": rep["steps_done"]}, indent=2))
    print("GATE:", "PASS" if gate["PASS"] else "FAIL", gate, flush=True)


def rc_str(rc):
    return "ok" if rc == 0 else str(rc)


if __name__ == "__main__":
    main()
