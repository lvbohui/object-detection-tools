import json
import os
import argparse

from pycocotools.coco import COCO
import cv2

parser = argparse.ArgumentParser("Cut specific objects")
parser.add_argument("--annotations-dir", type=str, 
                    default="output/component/val.json")
parser.add_argument("--image-folder", type=str, 
                    default="output/component/valid")
parser.add_argument("--output-dir", type=str, 
                    default="output/crop/double_bed")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

select_category_name = ["doubleBed"]

# read coco annotations with COCO
label_info = COCO(args.annotations_dir)

image2anno = label_info.imgToAnns
images = label_info.imgs

for image_id, image_info in images.items():
    file_name = image_info["file_name"]
    image = cv2.imread(os.path.join(args.image_folder, file_name))
    annos = image2anno[image_id]
    cut_num = 0
    for anno in annos:
        category_name = label_info.loadCats(anno["category_id"])[0]["name"]

        if category_name in select_category_name:
            bbox = anno["bbox"]
            anno_id = anno["id"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
            padding = 5
            crop_image = image[y1-padding:y2+padding, x1-padding:x2+padding]
            # crop_image = image[y1:y2, x1:x2]
            gray_crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            
            name = file_name.split(".")
            crop_name = "{}_{}.{}".format(name[-2], str(anno_id), name[-1])

            cv2.imwrite(os.path.join(args.output_dir, crop_name), gray_crop_image)
            cut_num += 1
    print("{} cut {}".format(file_name, cut_num))