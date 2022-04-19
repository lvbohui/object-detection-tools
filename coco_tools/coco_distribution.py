import argparse
import io
import json

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

parse = argparse.ArgumentParser()

parse.add_argument("--label-file", help="label file path", required=True)

args = parse.parse_args()

with open(args.label_file, "r", encoding="utf-8") as f:
    label_info = json.load(f)

    image_num = len(label_info["images"])

    categories = {}
    for c in label_info["categories"]:
        categories[c["id"]] = c["name"]
    
    count = {}
    for anno in label_info["annotations"]:
        count[anno["category_id"]] = count.get(anno["category_id"], 0) + 1
    
    result = {}
    summary = 0
    for k, v in count.items():
        result[categories[k]] = v
        summary += v

    over_standard = []
    cate = []
    for k, v in result.items():
        cate.append(k)
        print(f"{k}: {(v/summary):.2}")
        if v >= image_num:
            over_standard.append(k)
    print("平均每张图至少一个的目标: ", over_standard)
    print(cate)
        
    plt.figure(figsize=(20, 10))
    plt.title("HD component distribution")
    plt.xlabel("Category name")
    plt.ylabel("Numbers")
    plt.barh(list(result.keys()), list(result.values()))
    plt.savefig("distribution.jpg")
