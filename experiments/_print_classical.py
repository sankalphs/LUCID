import json

rows = json.load(open("results/all_results.json"))
cls = [r for r in rows if r.get("arch") == "classical"]
for tag in ("legacy", "A_strip", "B", "C"):
    print("---", tag)
    for r in cls:
        if r.get("split_preset") == tag:
            print("%-26s IoU %.4f Dice %.4f Acc %.4f HD95 %.4f BF1 %.4f"
                  % (r["exp_id"].split("/")[1], r["iou"], r["dice"],
                     r["accuracy"], r["hd95"], r["bf1"]))
