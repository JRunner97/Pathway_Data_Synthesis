import cv2
from matplotlib.pyplot import axis
import numpy as np
import random

class Synthetic_Rectangle:
    def __init__(self, center, label, im_path):
        self.center = center
        self.label = label
        self.font_size = random.randint(8, 20)
        self.text_margin = random.randint(5, 15)
        self.text_thickness = random.randint(1, 2)
        self.textbox_background = (0,0,230)
        self.textbox_border_thickness = random.randint(0, 2)

        image = cv2.imread(im_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.cnt = cnts[0][1]



    def draw_shape(self,img,center):

        cnts = self.cnt.reshape((-1,2))
        for cnt in cnts:
            img = cv2.circle(img, (cnt), 2, (0,0,0), -1)
        
        return img

    def get_points(self):

        cnt = self.cnt.reshape((-1,2))
        cols = cnt[:,0]
        rows = cnt[:,1]

        return rows, cols

    def check_point(self, x, y):

        dist = cv2.pointPolygonTest(self.cnt,(x,y),True)

        if dist < 0:
            return 1
        else:
            return 0


    def get_min_max(self):

        cnts = self.cnt.reshape((-1,2))
        
        maxes = cnts.max(axis=0)
        mins = cnts.min(axis=0)

        return mins[0], maxes[0], mins[1], maxes[1]

