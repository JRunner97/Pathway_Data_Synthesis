import numpy as np
import os
import cv2
import math
import random
import skimage.draw as draw
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Synthetic_Ellipse:
    def __init__(self, center, label):
        self.center = center
        self.label = label
        self.font_size = random.randint(8, 20)
        self.text_margin = random.randint(5, 15)
        self.text_thickness = random.randint(1, 2)
        self.textbox_background = (0,0,230)
        self.textbox_border_thickness = random.randint(0, 2)
    
    def draw_shape(self,img,center):

        img = cv2.ellipse(img, tuple(center), (int(round(self.width*.5))+self.text_margin,int(round(self.height*.5))+self.text_margin), 0, 0, 360, self.textbox_background, -1)
        if np.random.randint(2):
            border_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            img = cv2.ellipse(img, tuple(center), (int(round(self.width*.5))+self.text_margin,int(round(self.height*.5))+self.text_margin), 0, 0, 360, border_color, self.textbox_border_thickness)

        return img

    def get_points(self):
        '''
            inputs
                a: text width
                b: text height
        '''
        # a = w
        # b = h
        # a and b are axes lengths
        a = int(round(self.width*.5))+self.text_margin
        b = int(round(self.height*.5))+self.text_margin

        npoints = 1000
        delta_theta=2.0*math.pi/npoints

        theta=[0.0]
        delta_s=[0.0]
        integ_delta_s=[0.0]

        # integrated probability density
        integ_delta_s_val=0.0

        for iTheta in range(1,npoints+1):
            # ds/d(theta):
            delta_s_val=math.sqrt(a**2*math.sin(iTheta*delta_theta)**2+ \
                                b**2*math.cos(iTheta*delta_theta)**2)

            theta.append(iTheta*delta_theta)
            delta_s.append(delta_s_val)
            # do integral
            integ_delta_s_val = integ_delta_s_val+delta_s_val*delta_theta
            integ_delta_s.append(integ_delta_s_val)
            
        # normalize integrated ds/d(theta) to make into a scaled CDF (scaled to 2*pi)
        integ_delta_s_norm = []
        for iEntry in integ_delta_s:
            integ_delta_s_norm.append(iEntry/integ_delta_s[-1]*2.0*math.pi)

        ellip_x_prime=[]
        ellip_y_prime=[]

        npoints_new=40
        delta_theta_new=2*math.pi/npoints_new

        for theta_index in range(npoints_new):
            theta_val = theta_index*delta_theta_new

            # Do lookup:
            for lookup_index in range(len(integ_delta_s_norm)):
                if theta_val >= integ_delta_s_norm[lookup_index] and theta_val < integ_delta_s_norm[lookup_index+1]:
                    theta_prime=theta[lookup_index]
                    break
            
            # ellipse with transformation applied
            ellip_x_prime.append(a*math.cos(theta_prime))
            ellip_y_prime.append(b*math.sin(theta_prime))

        return ellip_x_prime, ellip_y_prime

    # TODO:: pivot this to be a IoU calculation
    def check_point(self, x, y):

        h = self.center[0]
        k = self.center[1]
        a = self.width
        b = self.height

        '''
            inputs
                a: text width
                b: text height
        '''

        # get ellipse axes lengths
        a = int(round(a*.5))+self.text_margin
        b = int(round(b*.5))+self.text_margin

        # a and b are axes widths of ellipse
        # checking the equation of
        # ellipse with the given point
        p = ((math.pow((x - h), 2) / math.pow(a, 2)) +
            (math.pow((y - k), 2) / math.pow(b, 2)))
    
        return p

    def get_min_max(self):

        tmp_min_x = self.center[0] - (self.width/2) - self.text_margin
        tmp_max_x = self.center[0] + (self.width/2) + self.text_margin

        tmp_min_y = self.center[1] - (self.height/2) - self.text_margin
        tmp_max_y = self.center[1] + (self.height/2) + self.text_margin

        return tmp_min_x, tmp_max_x, tmp_min_y, tmp_max_y

class Synthetic_Rectangle:
    def __init__(self, center, label):
        self.center = center
        self.label = label
        self.font_size = random.randint(8, 20)
        self.text_margin = random.randint(5, 15)
        self.text_thickness = random.randint(1, 2)
        self.textbox_background = (0,0,230)
        self.textbox_border_thickness = random.randint(0, 2)

    def draw_shape(self,img,center):

        # location is center
        # x1,y1 is top left corner
        x1 = center[0] - int(round(self.width/2)) - self.text_margin
        y1 = center[1] - int(round(self.height/2)) - self.text_margin

        img = cv2.rectangle(img, (x1, y1), (x1 + self.width + (self.text_margin*2), y1 + self.height + (self.text_margin*2)), self.textbox_background, -1)
        if np.random.randint(2):
            border_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            img = cv2.rectangle(img, (x1, y1), (x1 + self.width + (self.text_margin*2), y1 + self.height + (self.text_margin*2)), border_color, self.textbox_border_thickness)

        return img

    def get_points(self):

        rows,cols = draw.rectangle_perimeter(tuple([0,0]),tuple([self.height+(self.text_margin*2),self.width+(self.text_margin*2)]))
        rows = rows.flatten()
        cols = cols.flatten()

        max_y = np.amax(rows)
        max_x = np.amax(cols)
        center = [max_x/2,max_y/2]

        rows = rows - center[1]
        cols = cols - center[0]

        return cols, rows

    def check_point(self, x, y):

        h = self.center[0]
        k = self.center[1]
        a = self.width
        b = self.height

        a = a + self.text_margin*2
        b = b + self.text_margin*2

        # zero means its inside
        if x < h+(a/2) and x > h-(a/2) and y < k+(b/2) and y > k-(b/2):
            return 0
        else:
            return 1

    def get_min_max(self):

        tmp_min_x = self.center[0] - (self.width/2) - self.text_margin
        tmp_max_x = self.center[0] + (self.width/2) + self.text_margin

        tmp_min_y = self.center[1] - (self.height/2) - self.text_margin
        tmp_max_y = self.center[1] + (self.height/2) + self.text_margin

        return tmp_min_x, tmp_max_x, tmp_min_y, tmp_max_y


class Synthetic_Shape:
    def __init__(self, center, label):
        self.center = center
        self.label = label
        self.font_size = random.randint(8, 20)
        self.text_margin = random.randint(5, 15)
        self.text_thickness = random.randint(1, 2)
        self.textbox_background = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        self.text_color = (255 - self.textbox_background[0], 255 - self.textbox_background[1], 255 - self.textbox_background[2])
        self.textbox_border_thickness = random.randint(0, 2)

        shape_source_dir = "shape_images"
        shape_image = random.choice(os.listdir(shape_source_dir))
        image = cv2.imread(os.path.join(shape_source_dir,shape_image))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.cnt = cnts[0][1]
        self.source_image_dims = image.shape

    

    def draw_shape(self,img,center,textbox_background=None,textbox_border_thickness=None):

        if textbox_background is None:
            textbox_background = self.textbox_background
        if textbox_border_thickness is None:
            textbox_border_thickness = self.textbox_border_thickness

        # get zero centered points
        cols, rows = self.get_points()

        # send to new center
        rows = rows + center[1]
        cols = cols + center[0]

        pts = np.stack((cols,rows),axis=-1).astype(np.int32)

        if np.random.randint(2):
            img = cv2.fillPoly(img, [pts], textbox_background)
        if np.random.randint(2):
            border_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            img = cv2.polylines(img, [pts], True, border_color, self.textbox_border_thickness)

        return img

    def get_points(self):

        # get points of shape
        cnt = self.cnt.reshape((-1,2))
        cols = cnt[:,0]
        rows = cnt[:,1]
        
        # get dims of shape
        maxes = cnt.max(axis=0)
        mins = cnt.min(axis=0)
        min_x, max_x, min_y, max_y = mins[0], maxes[0], mins[1], maxes[1]
        len_x = max_x - min_x
        len_y = max_y - min_y

        # zero center shape
        center = [np.rint((len_x/2)+min_x),np.rint((len_y/2)+min_y)]
        rows = rows - center[1]
        cols = cols - center[0]

        rows = np.rint(rows).astype(np.int32)
        cols = np.rint(cols).astype(np.int32)

        # resize shape
        rows, cols = self.transform_points(rows, cols, len_x, len_y)

        return cols, rows

    def transform_points(self,rows,cols,current_width,current_height):

        # transform base shape to be the desired dimensions

        w_factor = (self.width + (self.text_margin*2)) / current_width
        h_factor = (self.height + (self.text_margin*2)) / current_height
        
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
        cols, rows = self.get_points()

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

        # get zero centered points
        cols, rows = self.get_points()

        # send to new center
        rows = rows + self.center[1]
        cols = cols + self.center[0]

        pts = np.stack((cols,rows),axis=-1).astype(np.int32)
        
        maxes = pts.max(axis=0)
        mins = pts.min(axis=0)

        return mins[0], maxes[0], mins[1], maxes[1]
