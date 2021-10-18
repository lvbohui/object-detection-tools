import argparse
import json
import os

import cv2

from utils.crop import anno_convert, crop_window, is_in_box

WINDOW_SIZE = 640 # sliding window size
STRIDE = 10 # sliding stride

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
output_dir = args.output

# Set label information
with open(coco_label_file, "r", encoding="utf-8") as f:
    label_info = json.load(f)

# Setting obj.names
with open(os.path.join(output_dir+"/..", "obj.names"), "w", encoding="utf-8") as f:
    for c in label_info["categories"]:
        f.writelines(c['name']+"\n")

images_dict = {}
for img in label_info["images"]:
    images_dict[img["file_name"]] = img["id"]

annotations_copy = label_info["annotations"].copy()

for img in os.listdir(images_folder):
    image = cv2.imread(os.path.join(images_folder, img))
    print("Sliding Image:{}".format(img))
    height, width, _ = image.shape

    image_id = images_dict[img]
    image_cnt = 0  # 记录当前图片上裁剪下来的sliding_window的数目，并作为新的训练数据的命名
    image_ext = img.split(".")[-1]

    for x in range(0, width-WINDOW_SIZE, STRIDE):
        for y in range(0, height-WINDOW_SIZE, STRIDE):
            sliding_window_box = [x, y, WINDOW_SIZE, WINDOW_SIZE]

            for anno in label_info["annotations"]:
                if anno["image_id"] == image_id:
                    if is_in_box(sliding_window_box, anno["bbox"]): # 当前sliding_window框到了一个目标
                        image_cnt += 1

                        # 裁剪下当前sliding_window
                        cropped_image = crop_window(image, sliding_window_box)
                        cropped_image_name = "{}_{}.{}".format(".".join(img.split(".")[:-1]), str(image_cnt), image_ext)
                        cv2.imwrite(os.path.join(output_dir, cropped_image_name), cropped_image)
                        
                        # 把所有在当前sliding_window内的目标的bounding box都转换坐标并保存
                        cropped_label_box = []
                        for anno_copy in annotations_copy:
                            if anno_copy["image_id"] == image_id:
                                if is_in_box(sliding_window_box, anno_copy["bbox"]):
                                    cropped_label_box.append((anno_copy["category_id"]-1, anno_convert(sliding_window_box, anno_copy["bbox"])))

                        convert_label_file = os.path.join(output_dir, cropped_image_name.replace(image_ext, "txt"))
                        with open(convert_label_file, "w", encoding="utf-8") as f_yolo:
                            for select_box in cropped_label_box:
                                category_id, [box_x, box_y, box_w, box_h] = select_box
                                x_center = (box_x + box_w/2)/WINDOW_SIZE
                                y_center = (box_y + box_h/2)/WINDOW_SIZE
                                f_yolo.writelines("{} {} {} {} {}\n".format(category_id, x_center, y_center, box_w/WINDOW_SIZE, box_h/WINDOW_SIZE))

                        # 移除已经被框过的标注框，防止重复
                        label_info["annotations"].remove(anno)

