import argparse
import json
import logging
import os
import os.path as osp
import random
import time

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

    annotation_file = osp.join(dataset_dir, "annotations.json")


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
    info["description"] = "HD Component detection dataset"
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
            json_file = open(osp.join(save_dir, 'val.json'), 'w+')
            json.dump(img_dict, json_file, cls=MyEncoder)
        elif img_id_list == test_files_ids and len(test_files_ids):
            json_file = open(osp.join(save_dir, 'test.json'), 'w+')
            json.dump(img_dict, json_file, cls=MyEncoder)

    return train_num, val_num, test_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser("COCO split")
    parser.add_argument("--label-file", type=str, default=".",
                         help="label file folder")
    parser.add_argument("--val-percent", type=float, default=0.2,
                         help="val percent")
    parser.add_argument("--test-percent", type=float, default=0.1,
                         help="test percent")
    parser.add_argument("--save-dir", type=str, default="./output",
                         help="save dir")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train_num, val_num, test_num = split_coco_dataset(args.label_file,
                                                      args.val_percent,
                                                      args.test_percent,
                                                      args.save_dir)
    logging.info("train num: {}, val num: {}, test num: {}".format(train_num, val_num, test_num))
