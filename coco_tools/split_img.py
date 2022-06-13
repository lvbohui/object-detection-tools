"""
Split images into train, valid, test sets, according to the label files.
"""

import argparse
import json
import os
import shutil

parser = argparse.ArgumentParser("COCO split image")
parser.add_argument("--image-folder", type=str,
                     help="image folder")
parser.add_argument("--train-label-file", type=str,
                     help="train label file")
parser.add_argument("--valid-label-file", type=str,
                     help="valid label file")
parser.add_argument("--test-label-file", type=str,
                     help="test label file")
parser.add_argument("--output-dir", type=str,
                     help="output directory")

args = parser.parse_args()
train_folder = os.path.join(args.output_dir, "train")
valid_folder = os.path.join(args.output_dir, "valid")
test_folder = os.path.join(args.output_dir, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

train_valid_test_notfound = [0, 0, 0]
with open(args.train_label_file, "r") as f:
    train_label = json.load(f)
    for item in train_label["images"]:
        image_name = item["file_name"]
        if os.path.exists(os.path.join(args.image_folder, image_name)):
            old_image_path = os.path.join(args.image_folder, image_name)
            new_image_path = os.path.join(train_folder, image_name)
            shutil.copyfile(old_image_path, new_image_path)
        else:
            print("{} not found".format(image_name))
            train_valid_test_notfound[0] += 1

with open(args.valid_label_file, "r") as f2:
    valid_label = json.load(f2)
    for item in valid_label["images"]:
        image_name = item["file_name"]
        if os.path.exists(os.path.join(args.image_folder, image_name)):
            old_image_path = os.path.join(args.image_folder, image_name)
            new_image_path = os.path.join(valid_folder, image_name)
            shutil.copyfile(old_image_path, new_image_path)
        else:
            
            train_valid_test_notfound[1] += 1

with open(args.test_label_file, "r") as f3:
    test_label = json.load(f3)
    for item in test_label["images"]:
        image_name = item["file_name"]
        if os.path.exists(os.path.join(args.image_folder, image_name)):
            old_image_path = os.path.join(args.image_folder, image_name)
            new_image_path = os.path.join(test_folder, image_name)
            shutil.copyfile(old_image_path, new_image_path)
        else:
            train_valid_test_notfound[2] += 1

print("Not found number:\nTrain:{}, Valid:{}, Test:{}".format(train_valid_test_notfound[0], 
                                                              train_valid_test_notfound[1], 
                                                              train_valid_test_notfound[2]))
