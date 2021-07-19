import json
import os
import numpy as np
import cv2
import scipy.fftpack as fp
from scipy import stats
from matplotlib import pyplot as plt
import copy
import label_file

# get image and json filepaths
directory = "processed_hardcases4"
images = []
json_files = []
for filename in os.listdir(directory):
    if filename.endswith(".png") or filename.endswith(".jpg"): 
        images.append(os.path.join(directory, filename))
        continue
    elif filename.endswith(".json"): 
        json_files.append(os.path.join(directory, filename))
        continue

images.sort()
json_files.sort()

print(images)
print(json_files)



# get relation and element json objects with relative bbox coords and relation image slices
relations = []
relation_indicators = []
relation_elements = []
count = 0
for current_imagepath, json_file in zip(images,json_files):

    # read image and json
    current_image = cv2.imread(current_imagepath)
    with open(json_file) as f:
        data = json.load(f)

    if "shapes" not in data:
        continue

    # make sure each shape has line color
    filtered_shapes = []
    for json_obj in data["shapes"]:

        if not json_obj:
            continue

        if 'line_color' not in json_obj:
            json_obj['line_color'] = [255,0,0,128]
        
        if 'fill_color' not in json_obj:
            json_obj['fill_color'] = None

        filtered_shapes.append(json_obj)

    # save json and new image
    image_path = data["imagePath"]
    imageHeight = data["imageHeight"]
    imageWidth = data["imageWidth"]
    json_filename = json_files[count].split("\\")[-1]
    print(json_filename)
    target_dir = "processed_hardcases5"
    template_label_file = label_file.LabelFile()
    template_label_file.save(os.path.join(target_dir, json_filename),filtered_shapes,image_path,imageHeight,imageWidth,imageData=data["imageData"])

    count+=1

            