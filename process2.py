import numpy as np
from numpy.lib.npyio import save
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import label_file
import cv2
import threading
from numpy import inf
import copy
import math
import random, string
import random
from argparse import Namespace
from scipy import stats

from scipy.sparse import base

X_ORIENTATION = 0
Y_ORIENTATION = 1

START = 0
END = 1

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

INHIBIT = 0
ACTIVATE = 1

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def check_orientation(source_point,comparison_pt):

    '''

        get prevailing cardinal direction of spline at interested end

        Args:
            source_point (list): last point on interested end of spline used as reference as [x,y]
            comparison_pt (list): spline point used as reference for deterimining slope [x,y]
        
        Return:
            orientation (int): prevailing cardinal direction of spline at interested end
            
    '''

    # check orientation
    # look at prevailing dimension of change
    if abs(source_point[0] - comparison_pt[0]) > abs(source_point[1] - comparison_pt[1]):
        # LEFT or RIGHT
        if source_point[0] - comparison_pt[0] > 0:
            orientation = RIGHT
        else:
            orientation = LEFT
    else:
        # UP or DOWN
        if source_point[1] - comparison_pt[1] > 0:
            orientation = DOWN
        else:
            orientation = UP

    return orientation

def get_slope(self,base_points,source_point,x_span,y_span):

    '''

        draws spline between entities

        Args:
            self: contains hyperparameter config
            base_points (np.Array): 2-D numpy array of spline unique pixel coordinates
            source_point (list): last point on interested end of spline used as reference as [x,y]
            x_span (np.Array): x-dim anchor points for spline
            y_span (np.Array): y-dim anchor points for spline
        
        Return:
            f (float): slope of spline at interested end
            comparison_pt (list): spline point used as reference for deterimining slope [x,y]
            
    '''

    # select distinct and nth closest
    max_idx = base_points.shape[0] - 1
    comparison_pt = None
    dist = 50
    # lag used to set which point along spline off of entity do we want to select as comparison point
    lag = 10
    # TODO:: dynamically adjust lag length based on # samples here
    candidate_points = []
    candidate_dists = []
    for idx in range(0,max_idx-1,1):
        if base_points[idx,0] != source_point[0] and base_points[idx,1] != source_point[1]:
            tmp_dist = math.dist(base_points[idx], source_point)
            if tmp_dist < dist:
                candidate_points.append(base_points[idx])
                candidate_dists.append(tmp_dist)

    if len(candidate_dists) >= lag:
        candidate_dists = np.array(candidate_dists)
        candidate_idxs = np.argsort(candidate_dists)
        comparison_pt = candidate_points[candidate_idxs[lag]]

    tmp_len = x_span.size
    if comparison_pt is None:
        if self.arrow_placement == END:
            comparison_pt = [x_span[0],y_span[0]]
        else:
            comparison_pt = [x_span[tmp_len-1],y_span[tmp_len-1]]

    x1 = source_point[0]
    x2 = comparison_pt[0]
    y1 = source_point[1]
    y2 = comparison_pt[1]
                
    f=(y2-y1)/(x2-x1)*-1

    return f, comparison_pt


def draw_spline(self,img,x_span,y_span):

    '''

        draws spline between entities

        Args:
            self: contains hyperparameter config
            img (np.Array): copied template image for stitching
            x_span (np.Array): x-dim anchor points for spline
            y_span (np.Array): y-dim anchor points for spline
        
        Return:
            img (np.Array): updated image with spline
            f (float): slope of spline at interested end
            orientation (int): prevailing cardinal direction of spline at interested end
            spline_bbox (list): 2D-list with bbox corners for spline as [[x,y],[x,y]]
            
    '''

    # get full spline (base_points) from anchor points (x_span,y_span)
    param = np.linspace(0, 1, x_span.size)
    # clever way to break it up to avoid 1st param no duplicate error
    # make linespace to serve as first param and interpolating the target values which is the set of x & y values
    spl = make_interp_spline(param, np.c_[x_span,y_span], k=3) #(1)
    # TODO:: change 500 parameter to be dynamic based on manitude differences in x_span
    X_, Y_ = spl(np.linspace(0, 1, x_span.size * 500)).T #(2)

    X_ = np.round(X_, 0).astype(int)
    Y_ = np.round(Y_, 0).astype(int)

    base_points = np.stack((X_,Y_)).T
    base_points = np.unique(base_points, axis=0)

    base_points_x = base_points[:,0]
    min_x = np.min(base_points_x)
    max_x = np.max(base_points_x)

    base_points_y = base_points[:,1]
    min_y = np.min(base_points_y)
    max_y = np.max(base_points_y)

    spline_bbox = [[min_x,min_y],[max_x,max_y]]

    # draw spline
    for x,y in base_points:
        img = cv2.circle(img, (x,y), self.thickness, self.arrow_color, -1)  

    # get reference point
    tmp_len = x_span.size
    if self.arrow_placement == END:
        source_point = [x_span[tmp_len-1],y_span[tmp_len-1]]
    else:
        source_point = [x_span[0],y_span[0]]

    f, comparison_pt = get_slope(self,base_points,source_point,x_span,y_span)
    orientation = check_orientation(source_point,comparison_pt)

    return img, f, orientation, spline_bbox


def draw_indicator(self,img,x_span,y_span,tip_slope,arrow_orientation):

    '''

        draw indicator

        Args:
            self: contains hyperparameter config
            img (np.Array): copied template image for stitching
            x_span (np.Array): x-dim anchor points for spline
            y_span (np.Array): y-dim anchor points for spline
            tip_slope (float): slope of spline at interested end
            arrow_orientation (int): prevailing cardinal direction of spline at interested end   
        
        Return:
            img (np.Array): updated image with indicator drawn on it
            indicator_bbox (list): 2D-list with bbox corners for indicator as [[x,y],[x,y]]
            
    '''

    # get reference point
    if self.arrow_placement == END:
        tri_source = [x_span[3],y_span[3]]
    else:
        tri_source = [x_span[0],y_span[0]]

    # img = cv2.circle(img, tuple(tri_source), 3, (40,255,0), -1)


    # TODO:: fine-tune these thresholds
    # if slope is extreme, then just place simple inicator in cardinal direction
    if math.isnan(tip_slope) or abs(tip_slope) > 10 or abs(tip_slope) < 0.5:

        if arrow_orientation == UP:
            pt1 = (tri_source[0]-self.base_len, tri_source[1])
            pt2 = (tri_source[0], tri_source[1]-self.tip_len)
            pt3 = (tri_source[0]+self.base_len, tri_source[1])
        elif arrow_orientation == DOWN:
            pt1 = (tri_source[0]+self.base_len, tri_source[1])
            pt2 = (tri_source[0], tri_source[1]+self.tip_len)
            pt3 = (tri_source[0]-self.base_len, tri_source[1])
        elif arrow_orientation == LEFT:
            pt1 = (tri_source[0], tri_source[1]-self.base_len)
            pt2 = (tri_source[0]-self.tip_len, tri_source[1])
            pt3 = (tri_source[0], tri_source[1]+self.base_len)
        else:
            pt1 = (tri_source[0], tri_source[1]+self.base_len)
            pt2 = (tri_source[0]+self.tip_len, tri_source[1])
            pt3 = (tri_source[0], tri_source[1]-self.base_len)

        if self.indicator == INHIBIT:
            triangle_cnt = np.array( [pt1, pt3] )
            cv2.drawContours(img, [triangle_cnt], 0, self.arrow_color, self.inhibit_tickness)
        else:
            triangle_cnt = np.array( [pt1, pt2, pt3] )
            cv2.drawContours(img, [triangle_cnt], 0, self.arrow_color, -1)

    # if slope is 'soft', then calculate appropriate indicator orientation
    else:

        # arrow base slope is just negative reciprocal
        arrowhead_base_slpe = -1/tip_slope

        # returned in radians
        # get angle from slope in right triangle drawn by slopes of tip and base
        tip_deg = math.atan(tip_slope)
        base_deg = math.atan(arrowhead_base_slpe)

        # get location of arrowhead tip point w/ law of sines and similar triangles
        tip_rise = self.tip_len * math.sin(tip_deg)
        tip_run = tip_rise / tip_slope
        tip_rise = math.floor(tip_rise)
        tip_run = math.floor(tip_run)

        # get location of arrowhead base points w/ law of sines and similar triangles
        if arrow_orientation == RIGHT or arrow_orientation == LEFT:
            base_rise = (self.base_len * math.sin(base_deg))
            base_run = base_rise / arrowhead_base_slpe
            base_rise = math.floor(base_rise)
            base_run = math.floor(base_run)
        else:
            # use similar triangles for run instead of pythogorean to avoid sign issues
            base_rise = self.base_len * math.sin(base_deg)
            base_run = base_rise / arrowhead_base_slpe
            base_rise = math.floor(base_rise)
            base_run = math.floor(base_run)


        pt1 = (tri_source[0]-base_run, tri_source[1]+base_rise)
        pt3 = (tri_source[0]+base_run, tri_source[1]-base_rise)

        if arrow_orientation == RIGHT:
            pt2 = (tri_source[0]+tip_run, tri_source[1]-tip_rise)
        elif arrow_orientation == LEFT:
            pt2 = (tri_source[0]-tip_run, tri_source[1]+tip_rise)
        elif arrow_orientation == DOWN:
            # adjust tip_run & tip_rise for positive or negative slope
            if tip_rise < 0:
                tip_rise *= -1
                tip_run *= -1
            pt2 = (tri_source[0]-tip_run, tri_source[1]+tip_rise)
        elif arrow_orientation == UP:
            # adjust tip_run & tip_rise for positive or negative slope on UP arrow
            # don't have to adjust base_rise & base_run for since pt1 and pt3 just flip
            if tip_rise > 0:
                tip_rise *= -1
                tip_run *= -1
            pt2 = (tri_source[0]-tip_run, tri_source[1]+tip_rise)

        if self.indicator == INHIBIT:
            triangle_cnt = np.array( [pt1, pt3] )
            cv2.drawContours(img, [triangle_cnt], 0, self.arrow_color, self.inhibit_tickness)
        else:
            triangle_cnt = np.array( [pt1, pt2, pt3] )
            cv2.drawContours(img, [triangle_cnt], 0, self.arrow_color, -1)

    # get bbox dims for corresponding indicator
    if self.indicator == INHIBIT:
        min_x = min([pt1[0],pt3[0]]) - self.inhibit_tickness
        min_y = min([pt1[1],pt3[1]]) - self.inhibit_tickness

        max_x = max([pt1[0],pt3[0]]) + self.inhibit_tickness
        max_y = max([pt1[1],pt3[1]]) + self.inhibit_tickness
    else:
        min_x = min([pt1[0],pt2[0],pt3[0]])
        min_y = min([pt1[1],pt2[1],pt3[1]])

        max_x = max([pt1[0],pt2[0],pt3[0]])
        max_y = max([pt1[1],pt2[1],pt3[1]])

    indicator_bbox = [[min_x,min_y],[max_x,max_y]]

    return img, indicator_bbox


def draw_textbox(self,img,label,location,w,h):

    '''

        draw text and surronding box

        Args:
            self: contains hyperparameter config
            img (np.Array): copied template image for stitching
            label (str): text to place as entity
            location (list): contains target center of entity bbox as [x,y]
            w (int): contains target x dim of text to place
            h (int): contains target y dim of text to place
        
        Return:
            img (np.Array): updated image with entities
            bbox (list): 2D-list with bbox corners for spline and indicator as [[x,y],[x,y]]
            
    '''

    # location is center
    x1 = location[0] - math.floor(w/2) - self.text_margin
    y1 = location[1] - math.floor(h/2) - self.text_margin

    # places text
    # to add boarder just do same rectangle but don't fill and can just make it to include optionally
    img = cv2.rectangle(img, (x1, y1), (x1 + w + (self.text_margin*2), y1 + h + (self.text_margin*2)), self.textbox_background, -1)
    img = cv2.rectangle(img, (x1, y1), (x1 + w + (self.text_margin*2), y1 + h + (self.text_margin*2)), (0,0,0), self.textbox_border_thickness)
    # putText takes coordinates of the bottom-left corner of the text string
    img = cv2.putText(img, label, (x1 + self.text_margin, y1 + h + self.text_margin), self.font_style, self.font_size, self.text_color, self.text_thickness)

    bbox = [[x1,y1],[x1 + w + (self.text_margin*2), y1 + h + (self.text_margin*2)]]


    return img, bbox

def draw_relationship(self,img,entity1_center,entity2_center,text1_shape,text2_shape,label1,label2):

    '''

        draw entities, draw spline, draw indicator

        Args:
            self: contains hyperparameter config
            img (np.Array): copied template image for stitching
            entity1_center (list): contains target center of entity1 bbox as [entity1_center_x,entity1_center_y]
            entity2_center (list): contains target center of entity2 bbox as [entity2_center_x,entity2_center_y]
            text1_shape (list): contains target dimensions of text to place as [w1,h1]
            text2_shape (list): contains target dimensions of text to place as [w2,h2]
            label1 (str): text to place as entity1 
            label2 (str): text to place as entity2     
        
        Return:
            img (np.Array): updated image with entities, spline, and indicator drawn on it
            relationship_bbox (list): 2D-list with bbox corners for spline and indicator as [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]
            
    '''

    w1,h1 = text1_shape
    w2,h2 = text2_shape

    img, entity1_bbox = draw_textbox(self,img,label1,entity1_center,w1,h1)
    img, entity2_bbox = draw_textbox(self,img,label2,entity2_center,w2,h2)


    # Dataset
    # TODO:: set up to be several different classes of splines
    # TODO:: add arched arrow

    # set num in linspace to be dependent on distance from one center to another
    anchor_points = np.linspace(entity1_center, entity2_center, num=50,dtype=np.int)

    # get start point for spline
    # check for first nth point outside of entity boxes in direct line to other entity
    tar_point = 5
    count = 0
    start_point = None
    for idx in range(anchor_points.shape[0]-1):
        current_point = anchor_points[idx,:]
        if current_point[0] < entity1_bbox[0][0] or current_point[1] < entity1_bbox[0][1] or current_point[0] > entity1_bbox[1][0] or current_point[1] > entity1_bbox[1][1]:
                count += 1
                if count == tar_point:
                    start_point = [math.floor(current_point[0]),math.floor(current_point[1])]

    # get end point for spline
    count = 0
    end_point = None
    for idx in range(anchor_points.shape[0]-1,0,-1):
        current_point = anchor_points[idx,:]
        if current_point[0] < entity2_bbox[0][0] or current_point[1] < entity2_bbox[0][1] or current_point[0] > entity2_bbox[1][0] or current_point[1] > entity2_bbox[1][1]:
                count += 1
                if count == tar_point:
                    end_point = [math.floor(current_point[0]),math.floor(current_point[1])]


    spline_points = np.linspace(start_point, end_point, num=4,dtype=np.int)
    x_span = spline_points[:,0]
    y_span = spline_points[:,1]

    # add noise in better way
    # x_noise = np.random.randint(0,20,(1,))
    # y_noise = np.random.randint(0,20,(1,))
    # x_span[1] = x_span[1] + x_noise
    # y_span[1] = y_span[1] + y_noise
    # x_span[2] = x_span[2] + x_noise
    # y_span[2] = y_span[2] + y_noise

    img, f, orientation, spline_bbox = draw_spline(self,img,x_span,y_span)
    img, indicator_bbox = draw_indicator(self,img,x_span,y_span,f,orientation)

    # get final relationship bbox by taking max dims of spline and indicator
    min_x = int(min([spline_bbox[0][0],indicator_bbox[0][0]]))
    min_y = int(min([spline_bbox[0][1],indicator_bbox[0][1]]))
    max_x = int(max([spline_bbox[1][0],indicator_bbox[1][0]]))
    max_y = int(max([spline_bbox[1][1],indicator_bbox[1][1]]))

    relationship_bbox = [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]

    return img,relationship_bbox

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


# x,y now define a center
def check_slice(template_im,slice_shape,x,y,padding):

    threshold = 50

    template_slice = template_im[y-padding:y+slice_shape[1]+padding,x-padding:x+slice_shape[0]+padding,:]

    # if all pixels are the same, then don't even have to run the rest of check 
    if np.all(template_slice == template_slice[0,0,0]):
        return True

    grey_slice = cv2.cvtColor(template_slice, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(grey_slice)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # x,y format
    center = (int(slice_shape[1] / 2),int(slice_shape[0] / 2))

    # get description of fft emitting from center
    radial_prof = radial_profile(magnitude_spectrum, center)

    radial_prof[radial_prof == -inf] = 0

    idx = range(0,radial_prof.shape[0])
    bin_means = stats.binned_statistic(idx, radial_prof, 'mean', bins=4)[0]
    
    if bin_means[-1] < threshold and bin_means[-2] < threshold:
        return True
    else:
        return False

                    
def get_entity_placement(self,slice_shape,x_target,y_target,label1,label2):

    '''

        find positions to place entities

        Args:
            self: contains hyperparameter config
            slice_shape (list): contains dimensions of target region as [x_dim,y_dim]
            x_target (int): top-left corner x placement location 
            y_target (int): top-left corner y placement location 
            label1 (str): text to place as entity1 
            label2 (str): text to place as entity2     
        
        Return:
            entity1_center (list): contains target center of entity1 bbox as [entity1_center_x,entity1_center_y]
            entity2_center (list): contains target center of entity2 bbox as [entity2_center_x,entity2_center_y]
            text1_shape (list): contains target dimensions of text to place as [w1,h1]
            text2_shape (list): contains target dimensions of text to place as [w2,h2]
            
    '''

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width
    (w1, h1), _ = cv2.getTextSize(label1, self.font_style, self.font_size, self.text_thickness)
    (w2, h2), _ = cv2.getTextSize(label2, self.font_style, self.font_size, self.text_thickness)
    text1_shape = [w1,h1]
    text2_shape = [w2,h2]

    # 4 configurations/positioning: hotdog, hamburger, square1, square2
    dim_ratio = slice_shape[0] / slice_shape[1]
    if dim_ratio > 4:
        # hotdog
        entity1_center_y = math.floor(slice_shape[1] / 2) + y_target
        entity1_center_x = x_target + slice_shape[0] - math.floor(w1/2) - self.text_margin

        entity2_center_y = math.floor(slice_shape[1] / 2) + y_target
        entity2_center_x = x_target + math.floor(w2/2) + self.text_margin
    elif dim_ratio < 0.25:
        # hamburger
        entity1_center_x = math.floor(slice_shape[0] / 2) + x_target
        entity1_center_y = y_target + math.floor(h1/2)+ self.text_margin

        entity2_center_x = math.floor(slice_shape[0] / 2) + x_target
        entity2_center_y = y_target + slice_shape[1] - math.floor(h2/2) - self.text_margin
    else:
        # squares: corners
        if np.random.randint(2):
            entity2_center_x = x_target + math.floor(w2/2) + self.text_margin
            entity2_center_y = y_target + math.floor(h2/2)+ self.text_margin

            entity1_center_x = x_target + slice_shape[0] - math.floor(w1/2) - self.text_margin
            entity1_center_y = y_target + slice_shape[1] - math.floor(h1/2) - self.text_margin
        # squares: flipped corners
        else:
            entity1_center_x = x_target + math.floor(w1/2)+ self.text_margin
            entity1_center_y = y_target + math.floor(h1/2)+ self.text_margin

            entity2_center_x = x_target + slice_shape[0] - math.floor(w2/2) - self.text_margin
            entity2_center_y = y_target + slice_shape[1] - math.floor(h2/2) - self.text_margin

    entity1_center = [entity1_center_x,entity1_center_y]
    entity2_center = [entity2_center_x,entity2_center_y]

    return entity1_center, entity2_center, text1_shape, text2_shape



class template_thread(threading.Thread):
    def __init__(self, threadID,name,template_list,directory):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.template_list = template_list
        self.directory = directory

    def run(self):

        '''

        Start 4 threads for generating x# samples from same templates at once 

        '''

        filename = self.template_list[self.threadID]
        
        # how many images per template
        stop_child_flag = False
        num_copies = 8
        for copy_idx in range(0,num_copies):

            # children in here
            child_thread_idx = copy_idx * 4
            # other parent threads
            child_thread_idx = self.threadID*num_copies + child_thread_idx

            if stop_child_flag:
                break

            child_thread0 = copy_thread(child_thread_idx,"child0",self.directory,filename)
            child_thread1 = copy_thread(child_thread_idx+1,"child1",self.directory,filename)
            child_thread2 = copy_thread(child_thread_idx+2,"child2",self.directory,filename)
            child_thread3 = copy_thread(child_thread_idx+3,"child3",self.directory,filename)

            child_thread0.start()
            if (copy_idx*4) + 1 > num_copies:
                stop_child_flag = True
                continue
            else:
                child_thread1.start()
            if (copy_idx*4) + 2 > num_copies:
                stop_child_flag = True
                continue
            else:
                child_thread2.start()
            if (copy_idx*4) + 3 > num_copies:
                stop_child_flag = True
                continue
            else:
                child_thread3.start()

            child_thread0.join()
            child_thread1.join()
            child_thread2.join()
            child_thread3.join()

class copy_thread(threading.Thread):
    def __init__(self,copyID,name,directory,filename):
        threading.Thread.__init__(self)
        self.copyID = copyID
        self.name = name
        self.directory = directory
        self.filename = filename

    def run(self):

        '''

        Attempt to place x# of samples on a template, save new image, and save annotation

        '''

        self.font_style = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size = 0.6
        self.padding = 0
        self.thickness = 1
        self.tip_len = 10
        self.base_len = 10
        self.text_margin = 10
        self.arrow_placement = END
        self.arrow_color = (120,120,0)
        self.textbox_background = (255,30,0)
        self.textbox_border_thickness = 1
        self.text_color = (0,0,0)
        self.text_thickness = 1
        self.indicator = INHIBIT
        self.inhibit_tickness = 2

        # loop through templates
        # read template and get query coords
        template_im = cv2.imread(os.path.join(self.directory, self.filename))

        # put relations on template and generate annotation
        element_indx = 0
        shapes = []
        base_shape = {
            "line_color": None,
            "fill_color": None,
            "component": [],
            "rotated_box": [],
            "ID": None,
            "label": None,
            "points": [],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        for relation_idx in range(30):

            # TODO:: make set of names to pull from or characters

            tmp_str_len = random.randint(3,7)
            label1 = randomword(tmp_str_len).upper()
            tmp_str_len = random.randint(3,7)
            label2 = randomword(tmp_str_len).upper()

            # TODO:: set y_dim params based on x_dim value
            x_dim = np.random.randint(100,500)
            y_dim = np.random.randint(100,500)
            slice_shape = [x_dim,y_dim]

            # randomly select indicator head
            if np.random.randint(2):
                self.indicator = INHIBIT
            else:
                self.indicator = ACTIVATE

            # check if queried coords are a valid location
            for idx in range(50):

                # subtracted max bounds to ensure valid coords
                x_target = np.random.randint(0+self.padding,template_im.shape[1]-slice_shape[0]-self.padding)
                y_target = np.random.randint(0+self.padding,template_im.shape[0]-slice_shape[1]-self.padding)
                
                # check if selected template area is good
                if check_slice(template_im,slice_shape,x_target,y_target,self.padding):

                    entity1_center,entity2_center,text1_shape,text2_shape = get_entity_placement(self,slice_shape,x_target,y_target,label1,label2)
                    template_im,relationship_bbox = draw_relationship(self,template_im,entity1_center,entity2_center,text1_shape,text2_shape,label1,label2)

                    # generate annotation
                    label1_x1 = math.floor(entity1_center[0] - (text1_shape[0]/2))
                    label1_y1 = math.floor(entity1_center[1] - (text1_shape[1]/2))
                    label1_x2 = math.ceil(entity1_center[0] + (text1_shape[0]/2))
                    label1_y2 = math.ceil(entity1_center[1] + (text1_shape[1]/2))
                    label1_bbox = [[label1_x1,label1_y1],[label1_x2,label1_y1],[label1_x2,label1_y2],[label1_x1,label1_y2]]

                    label2_x1 = math.floor(entity2_center[0] - (text2_shape[0]/2))
                    label2_y1 = math.floor(entity2_center[1] - (text2_shape[1]/2))
                    label2_x2 = math.ceil(entity2_center[0] + (text2_shape[0]/2))
                    label2_y2 = math.ceil(entity2_center[1] + (text2_shape[1]/2))
                    label2_bbox = [[label2_x1,label2_y1],[label2_x2,label2_y1],[label2_x2,label2_y2],[label2_x1,label2_y2]]

                    label1_shape = copy.deepcopy(base_shape)
                    label1_shape['points'] = label1_bbox
                    label1_shape['ID'] = element_indx
                    label1_shape['label'] = str(element_indx) + ":gene:" + label1
                    element_indx += 1

                    label2_shape = copy.deepcopy(base_shape)
                    label2_shape['points'] = label2_bbox
                    label2_shape['ID'] = element_indx
                    label2_shape['label'] = str(element_indx) + ":gene:" + label2
                    element_indx += 1

                    # TODO:: change order label1 and label2 based on arrow pos
                    indicator_shape = copy.deepcopy(base_shape)
                    indicator_shape['points'] = relationship_bbox
                    indicator_shape['ID'] = element_indx
                    if self.indicator == INHIBIT:
                        indicator_shape['label'] = str(element_indx) + ":inhibit:" + str(label1_shape['ID']) + "|" + str(label2_shape['ID'])
                    else:
                        indicator_shape['label'] = str(element_indx) + ":activate:" + str(label1_shape['ID']) + "|" + str(label2_shape['ID'])
                    element_indx += 1

                    shapes.append(label1_shape)
                    shapes.append(label2_shape)
                    shapes.append(indicator_shape)

                    break

        # save json and new image
        im_dir = "output_test"
        image_path = str(self.copyID) + ".png"
        cv2.imwrite(os.path.join(im_dir, image_path), template_im)
        imageHeight = template_im.shape[0]
        imageWidth = template_im.shape[1]
        template_label_file = label_file.LabelFile()
        template_label_file.save(os.path.join(im_dir,str(self.copyID) + ".json"),shapes,image_path,imageHeight,imageWidth)

def populate_figures():

    '''

    Start multiple threads for generating samples from 4 different templates at once 

    '''

    # loop through all templates
    stop_flag = False
    directory = "templates"
    template_list = os.listdir(directory)
    for template_idx in range(0,len(template_list)-1,4):

        if stop_flag:
            break

        thread0 = template_thread(template_idx,"thread-0",template_list,directory)
        thread1 = template_thread(template_idx+1,"thread-1",template_list,directory)
        thread2 = template_thread(template_idx+2,"thread-2",template_list,directory)
        thread3 = template_thread(template_idx+3,"thread-3",template_list,directory)

        thread0.start()
        if template_idx + 1 > len(template_list):
            stop_flag = True
            continue
        else:
            thread1.start()
        if template_idx + 2 > len(template_list):
            stop_flag = True
            continue
        else:
            thread2.start()
        if template_idx + 3 > len(template_list):
            stop_flag = True
            continue
        else:
            thread3.start()

        thread0.join()
        thread1.join()
        thread2.join()
        thread3.join()

if __name__ == "__main__":

    # another interesting idea would be to include targeted noise (i.e. lines with no indicator connecting no entities)
    populate_figures()
    