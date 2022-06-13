import argparse
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
                padding = 5
                crop_image = image[y1-padding:y2+padding, x1-padding:x2+padding]
                gray_crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                
                name = file_name.split(".")
                crop_name = "{}_{}.{}".format(name[-2], str(anno_id), name[-1])

                cv2.imwrite(os.path.join(output_dir, category_name, crop_name), gray_crop_image)
                cut_num += 1
        print("{} cut {}".format(file_name, cut_num))


parser = argparse.ArgumentParser("Cut specific objects")
parser.add_argument("--annotations-dir", type=str, 
                    default="output/clean_component/train.json")
parser.add_argument("--image-folder", type=str, 
                    default="output/clean_component/train")
parser.add_argument("--output-dir", type=str, 
                    default="output/crop")
args = parser.parse_args()


cut(args.annotations_dir, args.image_folder, "all", args.output_dir)
