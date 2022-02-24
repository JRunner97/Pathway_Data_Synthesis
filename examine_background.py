import numpy as np
from scipy.interpolate import make_interp_spline
import os
import label_file
import cv2
import threading
from numpy import inf
import copy
import math
import string
import random
from scipy import stats
import skimage.draw as draw
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from synthetic_shapes import Synthetic_Ellipse, Synthetic_Rectangle, Synthetic_Shape
import numpy as np
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from bokeh.plotting import show
import matplotlib.pyplot as plt

def check_img_backgroun():

    source_dir = "aug_samples"
    files = os.listdir(source_dir)
    freq_pixels = []
    for file in files:
        if file.endswith(".json"):
            continue
        
        img = cv2.imread(os.path.join(source_dir,file)).reshape((-1,3))
        pix_vals, pix_count = np.unique(img,return_counts=True,axis=0)
        # print(pix_count)
        most_freq_pix = pix_vals[np.argmax(pix_count)]
        freq_pixels.append(most_freq_pix)
    
    freq_pixels = np.array(freq_pixels)
    pix_vals, pix_count = np.unique(freq_pixels,return_counts=True,axis=0)
    print(pix_vals)

    pix_vals = [str(val) for val in pix_vals]

    plt.bar(pix_vals, pix_count.tolist(), color ='maroon', width = 0.4)
    plt.xlabel("Pixel Value (255=white, 0=black)")
    plt.ylabel("Image Count")
    plt.title("Most Frequent Background Colors")
    plt.show()

if __name__ == "__main__":

    source_dir = "val_images"
    files = os.listdir(source_dir)
    all_shapes = []
    freq_pixels = []
    for file in files:
        if not file.endswith(".json"):
            continue
        
        current_label_file = label_file.LabelFile(os.path.join(source_dir,file))
        current_img = cv2.imread(os.path.join(source_dir,file[:-4]+"jpg"))

        current_shapes = current_label_file.shapes
        for shape in current_shapes:
            if 'gene:' not in shape['label']:
                continue

            points = np.array(shape['points'])
            maxes = points.max(axis=0).astype(np.int)
            mins = points.min(axis=0).astype(np.int)

            current_slice = current_img[mins[1]:maxes[1],mins[0]:maxes[0],:].reshape((-1,3))

            if current_slice.shape[0] == 0:
                continue

            pix_vals, pix_count = np.unique(current_slice,return_counts=True,axis=0)
            print(current_slice)
            print(pix_count)
            most_freq_pix = pix_vals[np.argmax(pix_count)]
            freq_pixels.append(most_freq_pix)
        

    freq_pixels = np.array(freq_pixels)
    pix_vals, pix_count = np.unique(freq_pixels,return_counts=True,axis=0)
    print(pix_vals)

    pix_vals = [str(val) for val in pix_vals]

    plt.bar(pix_vals, pix_count.tolist(), color ='maroon', width = 0.4)
    plt.xlabel("Pixel Value (255=white, 0=black)")
    plt.ylabel("Image Count")
    plt.title("Most Frequent Background Colors")
    plt.show()