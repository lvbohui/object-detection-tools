import argparse
import json
import os
import tqdm

import cv2
from pycocotools import coco


class COCOAugmentation(object):
    def __init__(self, annotation_file, image_dir, output_dir):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.coco = coco.COCO(self.annotation_file)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.images = self.coco.imgs
        self.image_ids = self.coco.getImgIds()
        self.anno = self.coco.anns
        self.image2annos = self.coco.imgToAnns


    def rotation90(self):
        # Rotate the image
        rotated_images = []
        rotated_annos = []
        cnt = 0
        for image_id in self.image_ids:
            image_info = self.images[image_id]
            file_name = image_info["file_name"]
            image = cv2.imread(os.path.join(self.image_dir, file_name))
            H, W, _ = image.shape
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # cv2.imwrite(os.path.join(self.output_dir, file_name), rotated_image)
            new_H, new_W, _ = rotated_image.shape
            image_info["height"] = new_H
            image_info["width"] = new_W
            rotated_images.append(image_info)

            for anno in self.image2annos[image_id]:
                bbox = anno["bbox"]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
                rotated_x1, rotated_y1 = y1, W-x2 
                rotated_w, rotated_h = bbox[3], bbox[2]

                anno["bbox"] = [rotated_x1, rotated_y1, rotated_w, rotated_h]
                rotated_annos.append(anno)
            
            cnt += 1
            if cnt % 50 == 0:
                print("current processd images number:{}".format(cnt))

        # update annotations
        with open(self.annotation_file, "r") as f:
            label_info = json.load(f)
        label_info["annotations"] = rotated_annos
        label_info["images"] = rotated_images
        with open("rotated_doorwindow_valid.json", "w") as f:
            json.dump(label_info, f)


parser = argparse.ArgumentParser()
parser.add_argument("--annotation-file", type=str, required=True,
                    help="path to annotation file")
parser.add_argument("--image-dir", type=str, required=True,
                    help="directory of images")
parser.add_argument("--output-dir", type=str, required=True,
                    help="directory of output")
args = parser.parse_args()

os.makedirs(output_dir, exist_ok=True)

aug = COCOAugmentation(args.annotation_file, args.image_dir, args.output_dir)
aug.rotation90()

