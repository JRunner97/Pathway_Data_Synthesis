import json
import os
import numpy as np
import cv2
import scipy.fftpack as fp
from scipy import stats
from matplotlib import pyplot as plt

relation_slices = []

with open('image_new/hsa01521.json') as f:
  data = json.load(f)

imgFile = cv2.imread('image_new/hsa01521.png')

# TODO:: make this more efficient
# get relation and element json object and relation image slices
count = 0
relations = []
all_elements = []
for json_obj in data["shapes"]:
    

    if "activate" in json_obj['label'] or "inhibit" in json_obj['label']:
        if "75:HGF" in json_obj['label']:

            # get relation slice
            rel_pts = np.array(json_obj['points'])
            min_x = int(min(rel_pts[:,0]))
            min_y = int(min(rel_pts[:,1]))
            max_x = int(max(rel_pts[:,0]))
            max_y = int(max(rel_pts[:,1]))
            relation_slices.append(imgFile[min_y:max_y,min_x:max_x,:])

            # save relation json
            relations.append(json_obj)
    else:
        all_elements.append(json_obj)



# TODO:: make this more efficient
# get elements for each relation and correct bbox values for slice
# TODO:: this will have to be figure specific when loading in more than one json
print(relations)
relation_elements = []
for relation in relations:


    tmp_str = relation['label'].split("|")
    element1 = tmp_str[0].split(":")[1]
    element2 = tmp_str[1].split(":")[0]
    
    element1_json = None
    element2_json = None
    for element in all_elements:

        if str(element['id']) == element1:
            element1_json = element
        elif str(element['id']) == element2:
            element2_json = element



    # get min vals of relation box
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


# read template and get query coords
template_im = cv2.imread('templates/template3.png')


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


current_slice = processed_slices[0]
current_mask = masks[0]

# check if queried coords are a valid location
for idx in range(50):

    # subtracted max bounds to ensure valid coords
    slice_shape = current_slice.shape
    x_target = np.random.randint(0,template_im.shape[1]-slice_shape[1])
    y_target = np.random.randint(0,template_im.shape[0]-slice_shape[0])
    
    # check if selected template area is good
    if check_slice(template_im,slice_shape,x_target,y_target):

        template_slice = template_im[y_target:y_target+slice_shape[0],x_target:x_target+slice_shape[1],:]

        # add slice
        template_slice = cv2.bitwise_xor(template_slice, current_mask)
        template_im[y_target:y_target+slice_shape[0],x_target:x_target+slice_shape[1],:] = template_slice

   



cv2.imwrite('dst_rt.png', template_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# template_im2 = cv2.imread('templates/template4.png',0)
# # template_im2 = cv2.cvtColor(template_im, cv2.COLOR_BGR2GRAY)
# f = np.fft.fft2(template_im2)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))

# plt.subplot(121),plt.imshow(template_im2, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()
