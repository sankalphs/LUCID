import json
from pathlib import Path

p = Path("results/queue/queue.txt")
raw = p.read_bytes()
if raw.startswith(b"\xef\xbb\xbf"):
    raw = raw[3:]
lines = [l for l in raw.decode("utf-8-sig").splitlines() if l.strip()]
if not any("classical_A_strip" in l for l in lines):
    lines.insert(0, json.dumps({
        "name": "classical_A_strip",
        "args": ["experiments/run_classical_baselines.py",
                 "--split-preset", "A_strip"]}))
p.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
print("queue size:", len(lines), "| head:", lines[0][:50])
