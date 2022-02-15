import cv2
from matplotlib.pyplot import axis
import numpy as np
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# todo just use keypoints from contours
class Synthetic_Shape:
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
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.cnt = cnts[0][1]
        self.source_image_dims = image.shape

    def draw_shape(self,img):

        # TODO:: adjust for center and centering points 0,0
        # look at other draw  shapes and get points for reference
        # i think this should work for zero_centering the points
        # x, y = polygon1.exterior.xy

        # get zero centered points
        rows, cols = self.get_points()

        # send to new center
        rows = rows + self.center[1]
        cols = cols + self.center[0]

        pts = np.stack((cols,rows),axis=-1).astype(np.int32)

        # img = cv2.circle(img, tuple([int(x),int(y)]), 2, (0,0,0), -1)
        img = cv2.polylines(img, [pts], True, (0,0,0), 2)
        return img

    def get_points(self):


        # TODO:: adjust for center
        cnt = self.cnt.reshape((-1,2))
        cols = cnt[:,0]
        rows = cnt[:,1]

        min_x, max_x, min_y, max_y = self.get_min_max()
        len_x = max_x - min_x
        len_y = max_y - min_y

        center = [np.rint((len_x/2)+min_x),np.rint((len_y/2)+min_y)]

        rows = rows - center[1]
        cols = cols - center[0]

        rows = np.rint(rows).astype(np.int32)
        cols = np.rint(cols).astype(np.int32)

        rows, cols = self.transform_points(rows, cols, len_x, len_y)

        return rows, cols

    def transform_points(self,rows,cols,current_width,current_height):

        w_factor = self.width / current_width
        h_factor = self.height / current_height
        
        transformed_cols = []
        transformed_rows = []
        for idx in range(cols.shape[0]):
            x_coord = cols[idx] * w_factor
            y_coord = rows[idx] * h_factor

            transformed_cols.append(x_coord)
            transformed_rows.append(y_coord)

        transformed_cols = np.rint(np.array(transformed_cols)).astype(np.int32)
        transformed_rows = np.rint(np.array(transformed_rows)).astype(np.int32)
        
        return transformed_rows, transformed_cols


    def check_point(self, x, y):

        h = self.center[0]
        k = self.center[1]

        # get zero centered points
        rows, cols = self.get_points()

        # send to new center
        rows = rows + k
        cols = cols + h

        polygon = Polygon(zip(cols,rows))
        point = Point(x, y)
        inside_flag = polygon.contains(point)

        # return zero means its inside
        if inside_flag:
            return 0
        else:
            return 1


    def get_min_max(self):

        cnts = self.cnt.reshape((-1,2))
        
        maxes = cnts.max(axis=0)
        mins = cnts.min(axis=0)

        return mins[0], maxes[0], mins[1], maxes[1]

if __name__ == "__main__":

    my_shape = Synthetic_Shape([0,0], "tree", "2.jpg")

    my_shape.height = 100
    my_shape.width = 100
    rows, cols = my_shape.get_points()
    
    
    img = np.ones((1000,1000,3), np.float32) * 255
    my_shape.center = (500,500)

    img = my_shape.draw_shape(img)

    cv2.imshow('image3',img.astype(np.uint8))
    cv2.waitKey(0)
    
