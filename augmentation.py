import argparse
import json
import os
import shutil

import cv2

from utils.crop import anno_convert, crop_window, is_in_box

WINDOW_SIZE = 640 # crop window size

class DataAug(object):
    def __init__(self, image_folder, label_file, aug_categories, output_dir=".", copy_num=2):
        self.image_folder = image_folder
        with open(label_file, "r", encoding="utf-8") as f:
            self.label_info = json.load(f)

            self.image_dict = {}
            for image in self.label_info["images"]:
                self.image_dict[image["file_name"]] = image["id"]

            self.category_dict = {}
            for category in self.label_info["categories"]:
                self.category_dict[category["id"]] = category["name"]
        
        self.aug_categories = aug_categories
        self.output_dir = output_dir
        self.copy_num = copy_num

    def random_crop(self):
        annotations_copy = self.label_info["annotations"].copy()
        around_limiters = 10
        for img in os.listdir(self.image_folder):
            print("Aug image:{}".format(img))

            object_cnt = 0
            copy_cnt = 0

            image_id = self.image_dict[img]
            image_ext = img.split(".")[-1]
            image = cv2.imread(os.path.join(self.image_folder, img))

            for anno in self.label_info["annotations"]:
                if anno["image_id"] == self.image_dict[img]:
                    if self.category_dict[anno["category_id"]] in self.aug_categories: # 判定少数类别
                        x, y, w, h = anno["bbox"]
                        if w <= WINDOW_SIZE-around_limiters and h <= WINDOW_SIZE-around_limiters: # 判定目标大小小于设定框
                            print("Augmentation object:{}".format(self.category_dict[anno["category_id"]]))
                            object_cnt += 1

                            # 设定裁剪框的中心为当前目标的中心
                            x_center = x + w/2
                            y_center = y + h/2
                            crop_box = [int(x-(WINDOW_SIZE/2)), int(y-(WINDOW_SIZE/2)), WINDOW_SIZE, WINDOW_SIZE]
                            cropped_image = crop_window(image, crop_box)

                            # 裁剪框
                            cropped_image_name = "{}_aug_{}.{}".format(".".join(img.split(".")[:-1]), str(object_cnt), image_ext)
                            cv2.imwrite(os.path.join(self.output_dir, cropped_image_name), cropped_image)

                            # 找出当前裁剪框下的所有目标并转换坐标
                            cropped_label_box = []
                            for anno_copy in annotations_copy:
                                if anno_copy["image_id"] == image_id:
                                    if is_in_box(crop_box, anno_copy["bbox"]):
                                        cropped_label_box.append((anno_copy["category_id"]-1, anno_convert(crop_box, anno_copy["bbox"])))
                            convert_label_file = os.path.join(self.output_dir, cropped_image_name.replace(image_ext, "txt"))
                            with open(convert_label_file, "w", encoding="utf-8") as f_yolo:
                                for select_box in cropped_label_box:
                                    category_id, [box_x, box_y, box_w, box_h] = select_box
                                    x_center = (box_x + box_w/2)/WINDOW_SIZE
                                    y_center = (box_y + box_h/2)/WINDOW_SIZE
                                    f_yolo.writelines("{} {} {} {} {}\n".format(category_id, x_center, y_center, box_w/WINDOW_SIZE, box_h/WINDOW_SIZE))
                            
                            for copy_cnt in range(self.copy_num):
                                copy_file_without_ext = "{}_copy_{}.".format(".".join(cropped_image_name.split(".")[:-1]), str(copy_cnt))
                                
                                copy_file_img = os.path.join(self.output_dir, copy_file_without_ext + image_ext)
                                copy_file_label = os.path.join(self.output_dir, copy_file_without_ext + "txt")
                                
                                shutil.copyfile(os.path.join(self.output_dir, cropped_image_name), copy_file_img)
                                shutil.copyfile(convert_label_file, copy_file_label)
                        
                        else:
                            print("{} is oversize.".format(self.category_dict[anno["category_id"]]))



if __name__ == "__main__":
    parse = argparse.ArgumentParser("Sliding Window")
    parse.add_argument("--images_folder", default="../PyTorch-YOLOv3/images", 
                                        help="folder of labeled images")
    parse.add_argument("--label_file", default="../PyTorch-YOLOv3/annotations/instances_default.json", 
                                    help="COCO label file")
    parse.add_argument("--output", default="/home/xiaojiu/software/Yolo_mark-master/x64/Release/data/img",
                                help="output folder")
    args = parse.parse_args()

    images_folder = args.images_folder
    coco_label_file = args.label_file
    aug_category = ["double_door", "unequal_double_door", "sliding_door",
                    "shaped_window", "bay_window", "corner_window", "corner_bay_window"]
    output_dir = args.output

    data_aug = DataAug(images_folder, coco_label_file, aug_category, output_dir)
    data_aug.random_crop()