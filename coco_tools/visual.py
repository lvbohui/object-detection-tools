import json
import os

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization of model performance

category_id_to_name_file = "component_index_categories.json"
with open(category_id_to_name_file, "r") as f:
    category_id_to_name = json.load(f)

categories_name = []
for item in category_id_to_name.items():
    categories_name.append(item[1])

iou_score = [0.478, 0.555, 0.611, 0.716, 0.808, 
             0.821, 0.579, 0.916, 0.829, 0.287, 
             0.727, 0.260, 0.784, 0.999, 0.849]

average_iou_score = sum(iou_score) / len(iou_score)
print("Average IoU score: ", average_iou_score)
average_top_8_iou_score = sum(sorted(iou_score, reverse=True)[:8]) / 8
# print(sorted(iou_score, reverse=True))
print("Average top 8 IoU score: ", average_top_8_iou_score)

# Plot the IoU score with category name
# plt.figure(figsize=(20, 10))
# plt.title("IoU score with category name")
# plt.xlabel("Category name")
# plt.ylabel("IoU score")
# plt.grid(axis="y")
# plt.bar(categories_name, iou_score)
# plt.savefig("iou_score.jpg")
