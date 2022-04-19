import json
import os

import cv2

label_file = "train.json"

image_dir = "../PyTorch-YOLOv3/data/custom/images"
output_dir = "."

f = open(label_file, "r", encoding="utf-8")

label_info = json.load(f)

category_name_id = {}
for c in label_info["categories"]:
    category_name_id[c["id"]] = c["name"]

for image_label in label_info["images"]:
    image_id = image_label["id"]
    image = cv2.imread(os.path.join(image_dir, image_label["file_name"]))
    for anno in label_info["annotations"]:
        if anno["image_id"] == image_id:
            category_name = category_name_id[anno["category_id"]]
            bbox = anno["bbox"]
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(output_dir, image_label["file_name"]), image)
