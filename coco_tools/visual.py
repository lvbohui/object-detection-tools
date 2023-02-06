import argparse
import json
import os

import cv2
from pycocotools import coco

# Visualization of COCO data-set
parser = argparse.ArgumentParser()
parser.add_argument("--annotation-file", type=str, 
                    default="")
parser.add_argument("--image-dir", type=str,
                    default="")
parser.add_argument("--output-dir", type=str,
                    default="")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

coco = coco.COCO(args.annotation_file)

images = coco.imgs
image2anno = coco.imgToAnns

cnt = 0
for image_id in images.keys():
    image_file_name = images[image_id]["file_name"]
    if not os.path.exists(os.path.join(args.image_dir, image_file_name)):
        continue
    image = cv2.imread(os.path.join(args.image_dir, image_file_name))
    image_height, image_width, _ = image.shape

    annos = image2anno[image_id]
    for anno in annos:
        bbox = anno["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(args.output_dir, image_file_name), image)

    cnt += 1
    if cnt == 30:
        break
