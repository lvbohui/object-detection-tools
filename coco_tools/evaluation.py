import argparse
import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parse = argparse.ArgumentParser("Calculate mAP")
parse.add_argument("--dt", help="detection result", 
                    default="coco_instances_results.json")
parse.add_argument("--gt", help="ground truth",
                    default="output/component/val.json")
args = parse.parse_args()

gt = COCO(args.gt)
dt = COCO(args.gt).loadRes(args.dt)

e = COCOeval(gt, dt, "bbox")
with open(args.gt, "r", encoding="utf-8") as f:
    label_info = json.load(f)

categoires = label_info["categories"]

for i in range(len(categoires)):
    e.params.catIds = [i+1]
    print(categoires[i]["name"])

    e.evaluate()
    e.accumulate()
    e.summarize()
    print("\n")

