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

    def draw_shape(self,img,center):

        # TODO:: adjust for center and centering points 0,0
        # look at other draw  shapes and get points for reference
        # i think this should work for zero_centering the points

        min_x, max_x, min_y, max_y = self.get_min_max()
        len_x = max_x - min_x
        len_y = max_y - min_y

        cnts = self.cnt.reshape((-1,2))
        for cnt in cnts:

            zero_centered_point = [cnt[0]-(len_x/2),cnt[1]-(len_y/2)]
            img = cv2.circle(img, (zero_centered_point), 2, (0,0,0), -1)
        
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

        return rows, cols

    def check_point(self, x, y):

        # TODO:: adjust cnt points for locations in point

        h = self.center[0]
        k = self.center[1]
        # a = self.width
        # b = self.height




        # TODO:: adjust for center
        cnt = self.cnt.reshape((-1,2))
        cols = cnt[:,0]
        rows = cnt[:,1]

        min_x, max_x, min_y, max_y = self.get_min_max()
        len_x = max_x - min_x
        len_y = max_y - min_y

        center = [np.rint((len_x/2)+min_x),np.rint((len_y/2)+min_y)]

        rows = rows - center[1] + k
        cols = cols - center[0] + h

        rows = np.rint(rows).astype(np.int32)
        cols = np.rint(cols).astype(np.int32)

        points = np.stack([cols,rows])
        # points = points.reshape((-1,1,2))
        polygon = Polygon(zip(cols,rows))

        img = np.zeros((500,500,3), np.float32)
        # for x_coord, y_coord in zip(cols,rows):
        #     img = cv2.circle(img, (x_coord,y_coord), 1, (0,0,0), -1)

        for y_idx in range(img.shape[0]):
            for x_idx in range(img.shape[1]):
                # dist = cv2.pointPolygonTest(points,(x_idx,y_idx),True)
                point = Point(x_idx, y_idx)
                inside_flag = polygon.contains(point)
                if inside_flag:
                    img[y_idx,x_idx,:] = 255
                else:
                    img[y_idx,x_idx,:] = 0

        # min = img.min()
        # max = img.max()
        # img -= min
        # img /= (max - min) 
        # img *= 255

        # print(dist)
        # img = cv2.circle(img, (x,y), 2, (0,255,0), -1)
        cv2.imshow('image3',img.astype(np.uint8))
        cv2.waitKey(0)


        # return zero means its inside
        if dist < 0:
            return 1
        else:
            return 0


    def get_min_max(self):

        cnts = self.cnt.reshape((-1,2))
        
        maxes = cnts.max(axis=0)
        mins = cnts.min(axis=0)

        return mins[0], maxes[0], mins[1], maxes[1]

if __name__ == "__main__":

    my_shape = Synthetic_Shape([0,0], "tree", "2.jpg")
    rows, cols = my_shape.get_points()
    
    my_shape.center = (200,200)
    # for x_coord, y_coord in zip(cols,rows):
    #     img = cv2.circle(img, (x_coord+my_shape.center[0],y_coord+my_shape.center[1]), 1, (0,0,0), -1)
    
    
    val = my_shape.check_point(120,200)

    # cnt = my_shape.cnt.reshape((-1,2))
    # for point in cnt:
    #     zero_centered_point = [point[0],point[1]]
    #     img = cv2.circle(img, (zero_centered_point), 1, (0,0,0), -1)

    # val = my_shape.check_point(25,150)
    # print(val)
    # img = cv2.circle(img, (25,150), 2, (0,255,0), -1)

    # cv2.imshow('image3',img)
    # cv2.waitKey(0)
    
