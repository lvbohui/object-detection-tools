"""
Split COCO dataset with specific train, valid, test rate.
"""
import argparse
import json
import logging
import os
import os.path as osp
import random
import shutil
import time

import numpy as np

log_folder = "logs"
log_path = osp.join(log_folder, "coco_split.log")
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')


class MyEncoder(json.JSONEncoder):
    # 调整json文件存储形式
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def split_coco_dataset(dataset_dir, val_percent, test_percent, save_dir):
    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from pycocotools.coco import COCO
    if not osp.exists(osp.join(dataset_dir, "annotations.json")):
        logging.info("\'annotations.json\' is not found in {}!".format(
            dataset_dir))

    # annotation_file = osp.join(dataset_dir, "annotations.json")
    annotation_file = dataset_dir

    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    anno_ids = coco.getAnnIds()

    # 获取原标注中的license和info
    licenses = coco.dataset["licenses"]
    info = coco.dataset["info"]

    # 设置license和info
    localtime = time.asctime( time.localtime(time.time()))
    licenses[0]["url"] = "www.foxitsoftware.com"
    licenses[0]["name"] = "Foxit Software Copyright"
    info["description"] = "HD Color floorplan detection dataset"
    info["url"] = "www.foxitsoftware.com"
    info["version"] = "1.0"
    info["year"] = localtime.split(" ")[-1]
    info["contributor"] = "AIDATA"
    info["date_created"] = localtime
    info["author"] = "Bohui"

    val_num = int(len(img_ids) * val_percent)
    test_num = int(len(img_ids) * test_percent)
    train_num = len(img_ids) - val_num - test_num

    random.shuffle(img_ids)
    train_files_ids = img_ids[:train_num]
    val_files_ids = img_ids[train_num:train_num + val_num]
    test_files_ids = img_ids[train_num + val_num:]

    for img_id_list in [train_files_ids, val_files_ids, test_files_ids]:
        img_anno_ids = coco.getAnnIds(imgIds=img_id_list, iscrowd=0)
        imgs = coco.loadImgs(img_id_list)
        instances = coco.loadAnns(img_anno_ids)
        categories = coco.loadCats(cat_ids)
        img_dict = {
            "license": licenses,
            "info": info,
            "categories": categories,
            "images": imgs,
            "annotations": instances,
        }

        if img_id_list == train_files_ids:
            json_file = open(osp.join(save_dir, 'train.json'), 'w+')
            json.dump(img_dict, json_file, cls=MyEncoder)
        elif img_id_list == val_files_ids:
            json_file = open(osp.join(save_dir, 'valid.json'), 'w+')
            json.dump(img_dict, json_file, cls=MyEncoder)
        elif img_id_list == test_files_ids and len(test_files_ids):
            json_file = open(osp.join(save_dir, 'test.json'), 'w+')
            json.dump(img_dict, json_file, cls=MyEncoder)

    return train_num, val_num, test_num

def split_image(src_folder, output_dir, train_label_file, valid_label_file, test_label_file):

    train_folder = os.path.join(output_dir, "train")
    valid_folder = os.path.join(output_dir, "valid")
    test_folder = os.path.join(output_dir, "test")


    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    train_valid_test_notfound = [0, 0, 0]
    with open(train_label_file, "r") as f:
        train_label = json.load(f)
        for item in train_label["images"]:
            image_name = item["file_name"]
            if os.path.exists(os.path.join(src_folder, image_name)):
                old_image_path = os.path.join(src_folder, image_name)
                new_image_path = os.path.join(train_folder, image_name)
                shutil.copyfile(old_image_path, new_image_path)
            else:
                print("{} not found".format(image_name))
                train_valid_test_notfound[0] += 1

    with open(valid_label_file, "r") as f2:
        valid_label = json.load(f2)
        for item in valid_label["images"]:
            image_name = item["file_name"]
            if os.path.exists(os.path.join(src_folder, image_name)):
                old_image_path = os.path.join(src_folder, image_name)
                new_image_path = os.path.join(valid_folder, image_name)
                shutil.copyfile(old_image_path, new_image_path)
            else:
                
                train_valid_test_notfound[1] += 1

    with open(test_label_file, "r") as f3:
        test_label = json.load(f3)
        for item in test_label["images"]:
            image_name = item["file_name"]
            if os.path.exists(os.path.join(src_folder, image_name)):
                old_image_path = os.path.join(src_folder, image_name)
                new_image_path = os.path.join(test_folder, image_name)
                shutil.copyfile(old_image_path, new_image_path)
            else:
                train_valid_test_notfound[2] += 1

    print("Not found number:\nTrain:{}, Valid:{}, Test:{}".format(train_valid_test_notfound[0], 
                                                                train_valid_test_notfound[1], 
                                                                train_valid_test_notfound[2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("COCO split")
    parser.add_argument("--image-folder", type=str,
                         help="labeled image folder")
    parser.add_argument("--label-file", type=str, default=".",
                         help="label file folder")
    parser.add_argument("--valid-percent", type=float, default=0.2,
                         help="valid percent")
    parser.add_argument("--test-percent", type=float, default=0.1,
                         help="test percent")
    parser.add_argument("--save-dir", type=str, default="./output/color_floorplan",
                         help="save dir")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train_num, val_num, test_num = split_coco_dataset(args.label_file,
                                                      args.val_percent,
                                                      args.test_percent,
                                                      args.save_dir)
    logging.info("train num: {}, val num: {}, test num: {}".format(train_num, val_num, test_num))

    temp_list = ["train", "valid", "test"]

    train_label_file, valid_label_file, test_label_file = [os.path.join(args.save_dir, item) for item in temp_list]
    split_image(args.image_folder, args.save_dir, train_label_file, valid_label_file, test_label_file)
