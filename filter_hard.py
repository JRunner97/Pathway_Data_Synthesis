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
directory = "test_subset3"
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

    # get relationships
    tmp_indicators = []
    tmp_relations = []
    tmp_elements = []
    relation_sizes = []
    for json_obj in data["shapes"]:

        if "activate_relation" in json_obj['label'] or "inhibit_relation" in json_obj['label']:

            pts = np.clip(np.array(json_obj['points']),0,np.inf).astype('int')
            width,height = cv2.minAreaRect(pts)[1]
            relation_sizes.append(width*height)

            tmp_relations.append(json_obj)
        
        elif "activate" in json_obj['label'] or "inhibit" in json_obj['label']:

            # save indicator json
            tmp_indicators.append(json_obj)

        else:
            tmp_elements.append(json_obj)

    print(len(tmp_relations))

    # get 3 largest relations
    # if len(relation_sizes) >=3:
    #     sorted_indices = np.argsort(np.array(relation_sizes))
    #     tmp_relations = np.array(tmp_relations)
    #     tmp_relations = tmp_relations[sorted_indices]
    #     tmp_relations = tmp_relations[-3:]

    # get 3 smallest relations
    if len(relation_sizes) >=6:
        sorted_indices = np.argsort(np.array(relation_sizes))
        tmp_relations = np.array(tmp_relations)
        tmp_relations = tmp_relations[sorted_indices]
        tmp_relations = tmp_relations[:6]


    # get elements for each relation
    shapes = []
    for relation in tmp_relations:

        print("relation")
        print(relation['label'])

        tmp_str = relation['label'].split("|")
        source_elements = tmp_str[0].split(":")[-1]
        if "[" in source_elements:
            tmp_str2 = source_elements[1:-1]
            source_elements = tmp_str2.split(",")
        else:
            source_elements = [source_elements]


        target_elements = tmp_str[-1]
        if "[" in target_elements:
            tmp_str2 = target_elements[1:-1]
            target_elements = tmp_str2.split(",")
        else:
            target_elements = [target_elements]

        # get all relation's elements
        elements_to_save = []
        for element in tmp_elements:
            compare_str = element['label'].split(":")[0]
            if compare_str in source_elements:
                elements_to_save.append(element)
            elif compare_str in target_elements:
                elements_to_save.append(element)

        # get relation's indicator
        indicator_json = None
        for indicator in tmp_indicators:
            temp_str = indicator['label'].split("|")
            indicator_source_el = temp_str[0].split(":")[-1]
            if "[" in indicator_source_el:
                tmp_str2 = indicator_source_el[1:-1]
                indicator_source_el = tmp_str2.split(",")
            else:
                indicator_source_el = [indicator_source_el]


            indicator_target_el = temp_str[-1]
            if "[" in indicator_target_el:
                tmp_str2 = indicator_target_el[1:-1]
                indicator_target_el = tmp_str2.split(",")
            else:
                indicator_target_el = [indicator_target_el]

            
            if indicator_source_el == source_elements and indicator_target_el == target_elements:
                indicator_json = indicator

        for element in elements_to_save:
            if element not in shapes:
                shapes.append(element)

        shapes.append(relation)
        shapes.append(indicator_json)

    # shapes = list(set(shapes))

    # save json and new image
    image_path = data["imagePath"]
    imageHeight = data["imageHeight"]
    imageWidth = data["imageWidth"]
    json_filename = json_files[count].split("\\")[-1]
    print(json_filename)
    target_dir = "processed_hardcases4"
    template_label_file = label_file.LabelFile()
    template_label_file.save(os.path.join(target_dir, json_filename),shapes,image_path,imageHeight,imageWidth,imageData=data["imageData"])

    count+=1