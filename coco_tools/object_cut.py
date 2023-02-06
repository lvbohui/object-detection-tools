import argparse
import hashlib
import json
import os

import cv2
from pycocotools.coco import COCO


def cut(annotations_file, image_dir, cut_categories, output_dir="output"):
    # read coco annotations with COCO
    label_info = COCO(annotations_file)

    image2anno = label_info.imgToAnns
    images = label_info.imgs
    
    if cut_categories == "all":
        cates = label_info.cats
        cut_categories = []
        for i, cate_info in cates.items():
            cut_categories.append(cate_info["name"])
    print(cut_categories)

    # make output dirs
    os.makedirs(output_dir, exist_ok=True)
    for category_name in cut_categories:
        os.makedirs(os.path.join(output_dir, category_name), exist_ok=True)


    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        image = cv2.imread(os.path.join(image_dir, file_name))
        annos = image2anno[image_id]
        cut_num = 0
        for anno in annos:
            category_name = label_info.loadCats(anno["category_id"])[0]["name"]

            if category_name in cut_categories:
                bbox = anno["bbox"]
                anno_id = anno["id"]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
                crop_image = image[y1:y2, x1:x2]
                name = hashlib.md5(file_name.encode("utf-8")).hexdigest()
                crop_name = "{}_{}.{}".format(name, str(anno_id), ".jpg")

                cv2.imwrite(os.path.join(output_dir, category_name, crop_name), crop_image)
                cut_num += 1
        print("{} cut {}".format(file_name, cut_num))



parser = argparse.ArgumentParser("Cut specific objects")
parser.add_argument("--annotations-dir", type=str, 
                    default="dataset/color_floorplan/annotations/instances_default.json")
parser.add_argument("--image-folder", type=str, 
                    default="dataset/color_floorplan/images")
parser.add_argument("--output-dir", type=str, 
                    default="output/crop/color_floorplan/bevel")
args = parser.parse_args()


cut_categories = ["singleDoor", "simpleWindow", "slidingDoor"]
cut(args.annotations_dir, args.image_folder, cut_categories, args.output_dir)
