"""
Monitor baseline training progress.

Usage:
  python baselines/monitor.py unetplusplus
  python baselines/monitor.py deeplabv3plus
  python baselines/monitor.py all
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATUS_FILES = {
    "unetplusplus": ROOT / "status_unetplusplus.json",
    "deeplabv3plus": ROOT / "status_deeplabv3plus.json",
}


def fmt_time(secs):
    if secs is None:
        return "  --  "
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def render(arch, status_path):
    if not status_path.exists():
        print(f"[{arch}] status file not found: {status_path}")
        print(f"        (training has not started or status file was not yet created)")
        return False
    try:
        with open(status_path) as f:
            s = json.load(f)
    except json.JSONDecodeError:
        print(f"[{arch}] status file is being written... (retry)")
        return False

    phase = s.get("phase", "?")
    elapsed = fmt_time(s.get("elapsed_sec"))
    eta = fmt_time(s.get("eta_sec"))
    best_iou = s.get("best_val_iou", 0)
    best_ep = s.get("best_epoch", 0)

    print(f"[{arch}] phase={phase}  elapsed={elapsed}  ETA={eta}")
    print(f"        best val IoU so far: {best_iou:.4f} @ epoch {best_ep}")

    if phase == "train":
        epoch = s.get("epoch", 0)
        batch = s.get("batch", 0)
        bpe = s.get("batches_per_epoch", 0)
        pct = batch / max(bpe, 1) * 100
        running_loss = s.get("running_train_loss", 0)
        bar_w = 30
        filled = int(pct / 100 * bar_w)
        bar = "#" * filled + "-" * (bar_w - filled)
        print(f"        epoch {epoch}: batch {batch}/{bpe}  [{bar}] {pct:5.1f}%  loss={running_loss:.4f}")
    elif phase == "val_done":
        epoch = s.get("epoch", 0)
        train_iou = s.get("train_iou", 0)
        val_iou = s.get("val_iou", 0)
        val_dice = s.get("val_dice", 0)
        epoch_sec = s.get("epoch_sec", 0)
        es_counter = s.get("early_stopping_counter", 0)
        es_patience = s.get("early_stopping_patience", 0)
        print(f"        epoch {epoch} done in {fmt_time(epoch_sec)}: "
              f"train_iou={train_iou:.4f}  val_iou={val_iou:.4f}  val_dice={val_dice:.4f}")
        print(f"        early-stop counter: {es_counter}/{es_patience}")
    elif phase == "done":
        epoch = s.get("epoch", 0)
        metrics = s.get("metrics", {})
        train_time = s.get("train_time_sec", 0)
        print(f"        FINISHED at epoch {epoch} in {fmt_time(train_time)}")
        if metrics:
            print(f"        IoU={metrics.get('iou',0):.4f}  Dice={metrics.get('dice',0):.4f}  "
                  f"Acc={metrics.get('accuracy',0):.4f}  HD95={metrics.get('hd95',0):.4f}  "
                  f"BF1={metrics.get('bf1',0):.4f}")
    return True


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if "all" in args:
        targets = list(STATUS_FILES.keys())
    else:
        targets = [a for a in args if a in STATUS_FILES]
    if not targets:
        print(f"Unknown target(s): {args}")
        print(f"Available: {list(STATUS_FILES.keys())}")
        sys.exit(1)

    print(f"PSR baseline monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    any_rendered = False
    for arch in targets:
        if render(arch, STATUS_FILES[arch]):
            any_rendered = True
    if not any_rendered:
        print("No active training detected.")
    print("-" * 70)


if __name__ == "__main__":
    main()