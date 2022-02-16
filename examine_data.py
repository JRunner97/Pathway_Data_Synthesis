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


if __name__ == "__main__":

    source_dir = "output_test"
    files = os.listdir(source_dir)
    all_shapes = []
    for file in files:
        if not file.endswith(".json"):
            continue
        
        current_label_file = label_file.LabelFile(os.path.join(source_dir,file))

        all_shapes += current_label_file.shapes

    gene_dims = []
    inhibit_dims = []
    activate_dims = []
    for shape in all_shapes:

        points = np.array(shape['points'])
        maxes = points.max(axis=0)
        mins = points.min(axis=0)

        width = maxes[0] - mins[0]
        height = maxes[1] - mins[1]

        if 'gene:' in shape['label']:
            gene_dims.append([width,height])
        elif 'activate:' in shape['label']:
            activate_dims.append([width,height])
        elif 'inhibit:' in shape['label']:
            inhibit_dims.append([width,height])

    gene_dims = np.array(gene_dims)
    inhibit_dims = np.array(inhibit_dims)
    activate_dims = np.array(activate_dims)

    gene_avgs = gene_dims.mean(axis=0)
    inhibit_avgs = inhibit_dims.mean(axis=0)
    activate_avgs = activate_dims.mean(axis=0)

    gene_std = gene_dims.std(axis=0)
    inhibit_std = inhibit_dims.std(axis=0)
    activate_std = activate_dims.std(axis=0)

    # points = hv.Points(gene_dims)
    # plot = points.hist(dimension=['x','y'])
    # show(hv.render(plot))

    # points = hv.Points(inhibit_dims)
    # plot = points.hist(dimension=['x','y'])
    # show(hv.render(plot))

    points = hv.Points(activate_dims)
    plot = points.hist(dimension=['x','y'])
    show(hv.render(plot))
 
    # print('cow')

    # np.random.seed(1)
    # data = np.random.randn(10000)
    # frequencies, edges = np.histogram(data, 20)
    # print('Values: %s, Edges: %s' % (frequencies.shape[0], edges.shape[0]))
    # plot = hv.Histogram((edges, frequencies))


    print(gene_avgs)
    print(inhibit_avgs)
    print(activate_avgs)
    print(gene_std)
    print(inhibit_std)
    print(activate_std)
    