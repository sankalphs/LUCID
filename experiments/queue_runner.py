"""Sequential experiment queue runner.

Reads one JSON array-of-CLI-args per line from results/queue/queue.txt,
executes them IN ORDER with the repo python, moves finished lines to done.txt.
Lines may be appended at any time; the runner re-reads after each job.

Usage:
  Start-Process C:\Python313\python.exe -ArgumentList "experiments\queue_runner.py","--wait-for-pid-file","results\reproduction\train_pid.txt"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_DIR = PROJECT_ROOT / "results" / "queue"
PYTHON = sys.executable


def wait_for_pid_exit(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    pid = int(pid_file.read_text().strip())
    print(f"[queue] waiting for PID {pid} to exit ...", flush=True)
    while True:
        try:
            p = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                               capture_output=True, text=True)
            if str(pid) not in p.stdout:
                print(f"[queue] PID {pid} exited.", flush=True)
                return
        except Exception:
            return
        time.sleep(60)


def pop_job(queue_file: Path):
    raw = queue_file.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    lines = [ln.strip() for ln in raw.decode("utf-8-sig").splitlines()
             if ln.strip()]
    if not lines:
        return None, []
    job = json.loads(lines[0])          # parse FIRST; only consume on success
    queue_file.write_text("\n".join(lines[1:]) + ("\n" if len(lines) > 1
                          else ""), encoding="utf-8")
    return job, lines[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-for-pid-file", default=None)
    args = ap.parse_args()

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    queue_file = QUEUE_DIR / "queue.txt"
    done_file = QUEUE_DIR / "done.txt"
    if not queue_file.exists():
        queue_file.write_text("", encoding="utf-8")

    if args.wait_for_pid_file:
        wait_for_pid_exit(PROJECT_ROOT / args.wait_for_pid_file)

    while True:
        job, _rest = pop_job(queue_file)
        if job is None:
            print("[queue] empty; sleeping 120s", flush=True)
            time.sleep(120)
            continue
        argv = [PYTHON] + job["args"]
        name = job.get("name", job["args"][0])
        print(f"[queue] START {name}: {job['args']}", flush=True)
        t0 = time.time()
        r = subprocess.run(argv, cwd=str(PROJECT_ROOT))
        dt = (time.time() - t0) / 60
        status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
        with open(done_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"name": name, "status": status,
                                "minutes": round(dt, 1),
                                "args": job["args"]}) + "\n")
        print(f"[queue] DONE {name} [{status}] in {dt:.1f} min", flush=True)


if __name__ == "__main__":
    main()
