import json, numpy as np
rows=json.load(open('results/all_results.json'))
seeds=[r for r in rows if r['exp_id'].startswith('multiseed') or r['exp_id']=='repro_seed42']
seeds=sorted(seeds, key=lambda r: r['seed'])
print('multiseed so far:')
for r in seeds:
    print(f"seed {r['seed']:4d} IoU {r['iou']:.4f} Dice {r['dice']:.4f} HD95 {r['hd95']:.4f} BF1 {r['bf1']:.4f} best_ep {r.get('best_epoch')}")
for m in ['iou','dice','accuracy','hd95','bf1']:
    v=np.array([r[m] for r in seeds])
    if len(v)>1:
        print(m, f"{v.mean():.4f} +- {v.std(ddof=1):.4f} (min {v.min():.4f} max {v.max():.4f})")
