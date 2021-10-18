import argparse
import os

from terminaltables import AsciiTable


def count_labels(label_folder, category_dict):
    """
    Count the number of objects for each category in the data. (YOLO format only)

    :param label_folder: folder of label files.
    :type label_folder: str
    
    :param category_dict: dictionary of labels.
    :type category_dict: dict
    :eg category_dict: {0: "category1", 1: "category2"}
    """
    cnt_result = {}
    cnt_table = []
    temp_table = ["Category Name"]

    for category_name in category_dict.values():
        cnt_result[category_name] = 0
        temp_table.append(category_name)
    cnt_table.append(temp_table)

    label_files = [item for item in os.listdir(label_folder) if ".txt" in item]
    for label_file in label_files:
        f = open(os.path.join(label_folder, label_file), "r", encoding="utf-8")
        labels = f.readlines()
        for label in labels:
            category_num = int(label.split(" ")[0])
            cnt_result[category_dict[category_num]] += 1
    temp_table = ["Numbers"]
    for num in cnt_result.values():
        temp_table.append(num)
    cnt_table.append(temp_table)
    print(AsciiTable(cnt_table).table)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count numbers of each category.')
    parser.add_argument("-c", '--classes', help='class.names',default="data/custom/classes.names", required=True)
    parser.add_argument("-f", "--label_folder", help="label file folder", default="data/custom/labels", required=True)
    args = parser.parse_args()

    f = open(args.classes)
    class_cnt = f.readlines()
    f.close()

    cnt = 0
    category_dict = {}
    for c in class_cnt:
        category_dict[cnt] = c
        cnt += 1
    
    label_folder = args.label_folder

    count_labels(label_folder, category_dict)
