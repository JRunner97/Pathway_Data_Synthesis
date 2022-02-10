import cv2
from matplotlib.pyplot import axis
import numpy as np
import random

# image = cv2.imread('2.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('image', gray)
# cv2.waitKey()
# thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow('image', thresh)
# cv2.waitKey()
# cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cnts = cnts[0][1]

# img = np.ones((225,225,3), np.uint8) * 255

# dist = cv2.pointPolygonTest(cnts,(5,5),True)
# img = cv2.circle(img, (5,5), 2, (0,0,0), 1)
# print(dist)

# cnts = cnts.reshape((-1,2))
# for cnt in cnts:
#     img = cv2.circle(img, (cnt), 2, (0,0,0), 1)

#     cv2.imshow('image', img)
#     cv2.waitKey()

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

        
        return img

    def get_points(self):


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

my_shape = Synthetic_Rectangle([0,0],'cats','2.jpg')
tmp_min_x, tmp_max_x, tmp_min_y, tmp_max_y = my_shape.get_min_max()
print(tmp_max_x)