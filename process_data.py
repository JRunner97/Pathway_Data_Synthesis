import json
import os
import numpy as np
import cv2
import scipy.fftpack as fp
from scipy import stats
from matplotlib import pyplot as plt
import copy
import label_file


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    # sum over pixels the same radius away from center
    tbin = np.bincount(r.ravel(), data.ravel())

    # normalize by distance to center, since as radius grows the # of pixels in radius bin does also
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

def check_slice(template_im,slice_shape,x,y):

    template_slice = template_im[y_target:y_target+slice_shape[0],x_target:x_target+slice_shape[1],:]
    grey_slice = cv2.cvtColor(template_slice, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(grey_slice)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # x,y format
    center = (int(slice_shape[1] / 2),int(slice_shape[0] / 2))

    # get description of fft emitting from center
    radial_prof = radial_profile(magnitude_spectrum, center)

    idx = range(0,radial_prof.shape[0])
    bin_means = stats.binned_statistic(idx, radial_prof, 'mean', bins=4)[0]
    
    if bin_means[-1] < 50 and bin_means[-2] < 50:
        return True
    else:
        return False




directory = "test_subset"
images = []
json_files = []
for filename in os.listdir(directory):
    if filename.endswith(".png"): 
        images.append(os.path.join(directory, filename))
        continue
    elif filename.endswith(".json"): 
        json_files.append(os.path.join(directory, filename))
        continue

images.sort()
json_files.sort()



# TODO:: make this more efficient
# get relation and element json objects with reindexed id's and relative bbox coords and relation image slices 
count = 0
relations = []
relation_slices = []
relation_elements = []
for current_imagepath, json_file in zip(images,json_files):

    # read image and json
    current_image = cv2.imread(current_imagepath)
    with open(json_file) as f:
        data = json.load(f)


    # get relationships
    tmp_relations = []
    tmp_elements = []
    for json_obj in data["shapes"]:
        
        if "activate" in json_obj['label'] or "inhibit" in json_obj['label']:

            # get relation slice
            rel_pts = np.array(json_obj['points'])
            min_x = int(min(rel_pts[:,0]))
            min_y = int(min(rel_pts[:,1]))
            max_x = int(max(rel_pts[:,0]))
            max_y = int(max(rel_pts[:,1]))
            relation_slices.append(current_image[min_y:max_y,min_x:max_x,:])

            # save relation json
            tmp_relations.append(json_obj)
        else:
            tmp_elements.append(json_obj)
        
    # get elements for each relation
    for relation in tmp_relations:
        tmp_str = relation['label'].split("|")

        split1 = tmp_str[0].split(":")
        split2 = tmp_str[1].split(":")

        element1 = split1[1] + ":" + split1[2]
        element2 = split2[0] + ":" + split2[1]
        
        element1_json = None
        element2_json = None
        for element in tmp_elements:

            if element['label'] == element1:
                element1_json = element
            elif element['label'] == element2:
                element2_json = element


        # correct bbox values for slice
        rel_pts = np.array(relation['points'])
        el1_pts = np.array(element1_json['points'])
        el2_pts = np.array(element2_json['points'])
        min_x = min(rel_pts[:,0])
        min_y = min(rel_pts[:,1])

        rel_pts[:,0] -= min_x
        rel_pts[:,1] -= min_y
        el1_pts[:,0] -= min_x
        el1_pts[:,1] -= min_y
        el2_pts[:,0] -= min_x
        el2_pts[:,1] -= min_y
        relation['points'] = rel_pts.tolist()
        element1_json['points'] = el1_pts.tolist()
        element2_json['points'] = el2_pts.tolist()


        relations.append(relation)
        relation_elements.append([element1_json,element2_json])



# remove backgrounds and artifacts from relation slices
processed_slices = []
masks = []
for img in relation_slices:

    # greyscale and filter threshold
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # detect contours based on thresholded pixels
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours on base zero mask
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, contours,-1, 255, -1)

    # get mask to whiten parts that are not detected contours
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    invert_mask = np.invert(mask)
    new_img = cv2.bitwise_or(img, invert_mask)

    masks.append(mask)
    processed_slices.append(new_img)
    
# loop through all templates
directory = "templates"
for template_idx,filename in enumerate(os.listdir(directory)):
    # how many images per template
    num_copies = 1
    for copy_idx in range(num_copies):
        copy_idx = template_idx*num_copies + copy_idx
        
        # loop through templates
        # read template and get query coords
        template_im = cv2.imread(os.path.join(directory, filename))


        # put relations on template and generate annotation
        element_indx = 0
        shapes = []
        for count, current_slice in enumerate(processed_slices):

            current_mask = masks[count]
            relation = relations[count]
            element1_json,element2_json = relation_elements[count]

            # check if queried coords are a valid location
            for idx in range(10):

                # subtracted max bounds to ensure valid coords
                slice_shape = current_slice.shape
                x_target = np.random.randint(0,template_im.shape[1]-slice_shape[1])
                y_target = np.random.randint(0,template_im.shape[0]-slice_shape[0])
                
                # check if selected template area is good
                if check_slice(template_im,slice_shape,x_target,y_target):

                    # add slice
                    # can adjust here to make relation any color
                    template_slice = template_im[y_target:y_target+slice_shape[0],x_target:x_target+slice_shape[1],:]
                    template_slice = template_slice.astype("int16") - current_mask.astype("int16")
                    template_slice = np.clip(template_slice,0,255).astype("uint8")
                    template_im[y_target:y_target+slice_shape[0],x_target:x_target+slice_shape[1],:] = template_slice

                    # reindex labels and adjust bboxes
                    current_relation = copy.deepcopy(relation)
                    current_element1_json = copy.deepcopy(element1_json)
                    current_element2_json = copy.deepcopy(element2_json)

                    # reindex element 1 label and id
                    split1 = current_element1_json['label'].split(":")
                    split1[0] = str(element_indx)
                    current_element1_json['id'] = element_indx
                    current_element1_json['label'] = ":".join(split1)
                    element_indx += 1

                    # reindex element 2 label and id
                    split2 = current_element2_json['label'].split(":")
                    split2[0] = str(element_indx)
                    current_element2_json['id'] = element_indx
                    current_element2_json['label'] = ":".join(split2)
                    element_indx += 1

                    # reindex relation label
                    split3 = current_relation['label'].split("|")
                    activation_type = split3[0].split(":")[0]
                    current_relation['label'] = activation_type + ":" + ":".join(split1) + "|" + ":".join(split2)


                    # correct bbox values for slice
                    rel_pts = np.array(current_relation['points'])
                    el1_pts = np.array(current_element1_json['points'])
                    el2_pts = np.array(current_element2_json['points'])

                    rel_pts[:,0] += x_target
                    rel_pts[:,1] += y_target
                    el1_pts[:,0] += x_target
                    el1_pts[:,1] += y_target
                    el2_pts[:,0] += x_target
                    el2_pts[:,1] += y_target

                    current_relation['points'] = rel_pts.tolist()
                    current_element1_json['points'] = el1_pts.tolist()
                    current_element2_json['points'] = el2_pts.tolist()

                    shapes.append(current_relation)
                    shapes.append(current_element1_json)
                    shapes.append(current_element2_json)

        # save json and new image
        image_path = str(copy_idx) + ".png"
        cv2.imwrite(image_path, template_im)
        imageHeight = template_im.shape[0]
        imageWidth = template_im.shape[1]
        template_label_file = label_file.LabelFile()
        template_label_file.save(str(copy_idx) + ".json",shapes,image_path,imageHeight,imageWidth)

