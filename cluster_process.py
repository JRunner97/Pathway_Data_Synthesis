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
INDIRECT_INHIBIT = 2
INDIRECT_ACTIVATE = 3

LINE = 0
ARCH = 1
CORNER = 2

LONG = 0
TALL = 1
DOWN_SLASH = 2
UP_SLASH = 3

ELLIPSE = 0
RECT = 1
NO_SHAPE = 2
POLYGON = 3


def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def check_orientation(source_point,comparison_pt):

    """

        get prevailing cardinal direction of spline at interested end

        Args:
            source_point (list): last point on interested end of spline used as reference as [x,y]
            comparison_pt (list): spline point used as reference for deterimining slope [x,y]
        
        Return:
            orientation (int): prevailing cardinal direction of spline at interested end
            
    """

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

    """

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
            
    """



    # select distinct and nth closest
    max_idx = base_points.shape[0] - 1
    comparison_pt = None
    dist = 50
    # lag used to set which point along spline off of entity do we want to select as comparison point

    # lag = 10
    lag = int(0.1 * max_idx)
    
    candidate_points = []
    candidate_dists = []
    for idx in range(0,max_idx-1,1):
        if base_points[idx,0] != source_point[0] and base_points[idx,1] != source_point[1]:
            tmp_dist = math.dist(base_points[idx], source_point)
            if tmp_dist < dist:
                candidate_points.append(base_points[idx])
                candidate_dists.append(tmp_dist)


    try:
        if len(candidate_dists) >= lag:
            candidate_dists = np.array(candidate_dists)
            candidate_idxs = np.argsort(candidate_dists)
            comparison_pt = candidate_points[candidate_idxs[lag]]
    
    except Exception as e:
        print(e)
        print(candidate_dists)
        print(comparison_pt)
        print("LAHHHHHH")
    


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

    """

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
            
    """


    
    if self.spline_type == CORNER:
        spline_coef = x_span.size-2
    else:
        spline_coef = x_span.size-1

    # get full spline (base_points) from anchor points (x_span,y_span)
    param = np.linspace(0, 1, x_span.size)
    # clever way to break it up to avoid 1st param no duplicate error
    # make linespace to serve as first param and interpolating the target values which is the set of x & y values
    spl = make_interp_spline(param, np.c_[x_span,y_span], k=spline_coef) #(1)


    # TODO:: not sure what multiplier to use here
    X_, Y_ = spl(np.linspace(0, 1, x_span.size * 200)).T

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

    # dashed interval dependent on spline width
    if self.indicator == INDIRECT_ACTIVATE or self.indicator == INDIRECT_INHIBIT:
        arr_len = base_points.shape[0]
        base_interval = random.randint(5, 15)
        intervals = base_interval + 3 * self.thickness
        odd_out = arr_len % intervals
        base_points = base_points[:arr_len-odd_out,:].reshape(-1,intervals,2)[::2].reshape(-1,2)

    spline_bbox = [[min_x,min_y],[max_x,max_y]]



    # draw spline
    for x,y in base_points:
        img = cv2.circle(img, (x,y), self.thickness, self.arrow_color, -1)  

    # get reference point
    tmp_len = x_span.size
    if self.arrow_placement == END:
        source_point = [x_span[tmp_len-1],y_span[tmp_len-1]]
        corner_comparison_pt = [x_span[tmp_len-2],y_span[tmp_len-2]]
    else:
        source_point = [x_span[0],y_span[0]]
        corner_comparison_pt = [x_span[1],y_span[1]]


    if self.spline_type == CORNER:
        orientation = check_orientation(source_point,corner_comparison_pt)
        f = np.nan
    else:
        f, comparison_pt = get_slope(self,base_points,source_point,x_span,y_span)
        orientation = check_orientation(source_point,comparison_pt)
    

    return img, f, orientation, spline_bbox


def draw_indicator(self,img,x_span,y_span,tip_slope,arrow_orientation):

    """

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
            
    """

    num_points = x_span.size

    # get reference point
    if self.arrow_placement == END:
        tri_source = [x_span[num_points-1],y_span[num_points-1]]
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
            cv2.drawContours(img, [triangle_cnt], 0, self.arrow_color, self.thickness+2)
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
            cv2.drawContours(img, [triangle_cnt], 0, self.arrow_color, self.thickness+2)
        else:
            triangle_cnt = np.array( [pt1, pt2, pt3] )
            cv2.drawContours(img, [triangle_cnt], 0, self.arrow_color, -1)

    # get bbox dims for corresponding indicator
    if self.indicator == INHIBIT:
        min_x = min([pt1[0],pt3[0]]) - self.thickness + 2
        min_y = min([pt1[1],pt3[1]]) - self.thickness + 2

        max_x = max([pt1[0],pt3[0]]) + self.thickness + 2
        max_y = max([pt1[1],pt3[1]]) + self.thickness + 2
    else:
        min_x = min([pt1[0],pt2[0],pt3[0]])
        min_y = min([pt1[1],pt2[1],pt3[1]])

        max_x = max([pt1[0],pt2[0],pt3[0]])
        max_y = max([pt1[1],pt2[1],pt3[1]])

    indicator_bbox = [[min_x,min_y],[max_x,max_y]]

    return img, indicator_bbox


def draw_textbox(self,img,current_entitiy,location):

    '''
    
    location is shape center
    
    '''

    # self,img,current_entitiy['label'],current_center,current_entitiy['width'],current_entitiy['height']

    label = current_entitiy.label
    w = current_entitiy.width
    h = current_entitiy.height

    """

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
            
    """

    img = current_entitiy.draw_shape(img,location)

    # location is center
    # x1,y1 is top left corner
    x1 = location[0] - int(round(w/2)) - current_entitiy.text_margin
    y1 = location[1] - int(round(h/2)) - current_entitiy.text_margin

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x1 + current_entitiy.text_margin, y1 + h + current_entitiy.text_margin), label, font=current_entitiy.font, fill=current_entitiy.text_color, anchor='lb')
    img = np.array(img_pil)

    bbox = [[x1,y1],[x1 + w + (current_entitiy.text_margin*2), y1 + h + (current_entitiy.text_margin*2)]]

    # cv2.imshow('image3',img)
    # cv2.waitKey(0)


    return img, bbox

def get_arch_anchors(self,start_point,end_point):

    point_dist = math.dist(start_point,end_point)
    arch_radius = self.arch_ratio * point_dist

    # randomly select indicator head
    if np.random.randint(2):
        arch_radius *= -1

    x1 = start_point[0]
    y1 = start_point[1]
    x2 = end_point[0]
    y2 = end_point[1]
    mid_point = [(x1+x2)/2,(y1+y2)/2]

    # TODO:: make sign of arch_radius dynamic
    # account for different orientations of end_point and start_point
    if x1 == x2:
        arch_anchor = [math.floor(mid_point[0]+arch_radius),math.floor(mid_point[1])]
    elif y1 == y2:
        arch_anchor = [math.floor(mid_point[0]),math.floor(mid_point[1]+arch_radius)]
    else:
        f=(y2-y1)/(x2-x1)*-1
        perp_slope = -1/f

        radians = math.atan(perp_slope)
        base_rise = arch_radius * math.sin(radians)
        base_run = base_rise / perp_slope
        base_rise = math.floor(base_rise)
        base_run = math.floor(base_run)

        mid_point = [(x1+x2)/2,(y1+y2)/2]
        arch_anchor = [math.floor(mid_point[0]-base_run),math.floor(mid_point[1]+base_rise)]

    spline_points = np.array([start_point,arch_anchor,end_point])

    return spline_points

def get_corner_anchors(entity_configuration,entity1_center,entity2_center,entity1_bbox,entity2_bbox):

    pad = 15

    if entity_configuration == DOWN_SLASH:
        if np.random.randint(2):
            start_point = [entity1_bbox[1][0]+pad,entity1_center[1]]
            end_point = [entity2_center[0],entity2_bbox[0][1]-pad]
            corner_point = [entity2_center[0],entity1_center[1]]
        else:
            start_point = [entity1_center[0],entity1_bbox[1][1]+pad]
            end_point = [entity2_bbox[0][0]-pad,entity2_center[1]]
            corner_point = [entity1_center[0],entity2_center[1]]

    else:
        if np.random.randint(2):
            start_point = [entity1_bbox[0][0]-pad,entity1_center[1]]
            end_point = [entity2_center[0],entity2_bbox[0][1]-pad]
            corner_point = [entity2_center[0],entity1_center[1]]
        else:
            start_point = [entity1_center[0],entity1_bbox[1][1]+pad]
            end_point = [entity2_bbox[1][0]+pad,entity2_center[1]]
            corner_point = [entity1_center[0],entity2_center[1]]

    spline_points = np.array([start_point,corner_point,end_point])

    return spline_points

def get_spline_anchors(self,entity1_center,entity2_center,entity1_bbox,entity2_bbox,entity_configuration):

    """

        draw entities, draw spline, draw indicator

        Args:
            self: contains hyperparameter config
            entity1_center (list): contains target center of entity1 bbox as [entity1_center_x,entity1_center_y]
            entity2_center (list): contains target center of entity2 bbox as [entity2_center_x,entity2_center_y]  
            entity1_bbox (list): 2D-list with entity1 bbox corners as [[x,y],[x,y]]
            entity2_bbox (list): 2D-list with entity2 bbox corners as [[x,y],[x,y]]
        
        Return:
            x_span (np.Array): x-dim anchor points for spline
            y_span (np.Array): y-dim anchor points for spline
            
    """

    # Dataset
    # TODO:: set up to be several different classes of splines
    # TODO:: add arched arrow
    # TODO:: corner spline on squares

    # TODO:: don't need top block for corners

    # set num in linspace to be dependent on distance from one center to another
    canidate_points = np.linspace(entity1_center, entity2_center, num=50, dtype=np.int)

    # get start point for spline
    # check for first nth point outside of entity boxes in direct line to other entity
    tar_point = 5
    count = 0
    start_point = None
    for idx in range(canidate_points.shape[0]-1):
        current_point = canidate_points[idx,:]
        if current_point[0] < entity1_bbox[0][0] or current_point[1] < entity1_bbox[0][1] or current_point[0] > entity1_bbox[1][0] or current_point[1] > entity1_bbox[1][1]:
                count += 1
                if count == tar_point:
                    start_point = [math.floor(current_point[0]),math.floor(current_point[1])]

    # get end point for spline
    count = 0
    # TODO:: end_point not getting set **************** bug
    end_point = None
    for idx in range(canidate_points.shape[0]-1,0,-1):
        current_point = canidate_points[idx,:]
        if current_point[0] < entity2_bbox[0][0] or current_point[1] < entity2_bbox[0][1] or current_point[0] > entity2_bbox[1][0] or current_point[1] > entity2_bbox[1][1]:
                count += 1
                if count == tar_point:
                    end_point = [math.floor(current_point[0]),math.floor(current_point[1])]


    if entity_configuration == LONG or entity_configuration == TALL:

        if np.random.randint(2):
            # LINE
            spline_points = np.linspace(start_point, end_point, num=4,dtype=np.int)
        else:
            # ARCH
            spline_points = get_arch_anchors(self,start_point,end_point)

    else:

        # TODO:: do this better, maybe set multimodal distribution to pull from
        rand_int = np.random.randint(3)
        if rand_int == 0 and abs(entity1_center[0]-entity2_center[0]) > 50 and abs(entity1_center[1]-entity2_center[1]) > 50:
            self.spline_type = CORNER
            spline_points = get_corner_anchors(entity_configuration,entity1_center,entity2_center,entity1_bbox,entity2_bbox)
        elif rand_int == 1:
            # ARCH
            spline_points = get_arch_anchors(self,start_point,end_point)
        else:
            # LINE
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


    return x_span,y_span

def check_ellipse_point( h, k, x, y, a, b):

    # a and b are axes widths of ellipse
 
    # checking the equation of
    # ellipse with the given point
    p = ((math.pow((x - h), 2) / math.pow(a, 2)) +
         (math.pow((y - k), 2) / math.pow(b, 2)))
 
    return p

def check_rect_point(h, k, x, y, a, b):

    # ref_center[0],ref_center[1],updated_x,updated_y,ref_width,entity['height']

    if x < h+(a/2) and x > h-(a/2) and y < k+(b/2) and y > k-(b/2):
        return 0
    else:
        return 1

def get_ellipse(self,a,b):
    # a = w
    # b = h

    # a and b are axis widths
    a = int(round(a*.5))+self.text_margin
    b = int(round(b*.5))+self.text_margin

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

def get_square(self,a,b):

    # + (self.text_margin*2)

    rows,cols = draw.rectangle_perimeter(tuple([0,0]),tuple([a+(self.text_margin*2),b+(self.text_margin*2)]))
    rows = rows.flatten()
    cols = cols.flatten()

    max_y = np.amax(rows)
    max_x = np.amax(cols)
    center = [max_x/2,max_y/2]

    rows = rows - center[1]
    cols = cols - center[0]

    return rows, cols

def draw_cluster(entity1,current_entitiy,placed_entities,img):

    # get placed cluster mask
    cluster_mask = np.zeros((1000,1000,1), np.float32)
    for entity in placed_entities:
        cluster_mask = entity.draw_shape(cluster_mask.astype(np.uint8),[entity.center[0]+500,entity.center[1]+500],(255,255,255),(255,255,255))

    entity1_center = entity1.center

    # get shape 1 and 2
    shape1_x,shape1_y = entity1.get_points()

    # TODO:: problem of unbalance direction sampling from sparse set of points from get_points()
    # select point from first entity in cluster and keep going in that dir until out of cluster
    tmp_idx = random.randint(0,len(shape1_x)-1)
    factor = 1.0
    while True:
        
        # get new entity mask
        new_entity_mask = np.zeros((1000,1000,1), np.float32)
        test_center = [int(shape1_x[tmp_idx]*factor)+entity1_center[0],int(shape1_y[tmp_idx]*factor)+entity1_center[1]]
        new_entity_mask = current_entitiy.draw_shape(new_entity_mask.astype(np.uint8),[test_center[0]+500,test_center[1]+500],(255,255,255),(255,255,255))
        
        # check iou
        mul_mask = np.multiply(cluster_mask.astype(np.int32),new_entity_mask.astype(np.int32))
        where_0 = np.where(mul_mask > 255)
        mul_mask[where_0] = 255

        # if no overlap break out
        if np.all(mul_mask == 0):
            break

        factor += 0.1

    new_center = [int(shape1_x[tmp_idx]*factor)+entity1_center[0],int(shape1_y[tmp_idx]*factor)+entity1_center[1]]

    return new_center, img

def draw_relationship(self,img,entity1_center,entity2_center,entity_configuration,placed_entities1,placed_entities2,text1_shape,text2_shape):

    """

        draw entities, draw spline, draw indicator

        Args:
            self: contains hyperparameter config
            img (np.Array): copied template image for stitching
            entity1_center (list): contains target center of cluster1 bbox as [entity1_center_x,entity1_center_y]
            entity2_center (list): contains target center of cluster2 bbox as [entity2_center_x,entity2_center_y]
            text1_shape (list): contains target dimensions of text to place as [w1,h1]
            text2_shape (list): contains target dimensions of text to place as [w2,h2]
            label1 (str): text to place as entity1 
            label2 (str): text to place as entity2     
        
        Return:
            img (np.Array): updated image with entities, spline, and indicator drawn on it
            relationship_bbox (list): 2D-list with bbox corners for spline and indicator as [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]
            
    """

    
    w1 = text1_shape[0]
    h1 = text1_shape[1]
    # iteratively place cluster entities
    for idx in range(len(placed_entities1)):

        current_entitiy = placed_entities1[idx]

        current_entitiy.textbox_background = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        current_entitiy.text_color = (255 - current_entitiy.textbox_background[0], 255 - current_entitiy.textbox_background[1], 255 - current_entitiy.textbox_background[2])

        current_center = [current_entitiy.center[0] + entity1_center[0] - int(w1/2),current_entitiy.center[1] + entity1_center[1] - int(h1/2)]
        img, tmp_bbox = draw_textbox(self,img,current_entitiy,current_center)

        placed_entities1[idx].bbox = tmp_bbox

        # img = cv2.rectangle(img, tuple([int(current_center[0]-current_entitiy['width']*.8),current_center[1]-current_entitiy['height']]),tuple([int(current_center[0]+current_entitiy['width']*.8),current_center[1]+current_entitiy['height']]),(255,0,0),1)
        # img = cv2.rectangle(img, tuple([tmp_bbox[0][0],tmp_bbox[0][1]]),tuple([tmp_bbox[1][0],tmp_bbox[1][1]]),(255,0,0),1)


    # img = cv2.circle(img,tuple(entity1_center),2,(255,0,0),1)

    w1 = text1_shape[0]
    h1 = text1_shape[1]
    entity1_bbox = [[int(entity1_center[0]-(w1/2)),int(entity1_center[1]-(h1/2))],[int(entity1_center[0]+(w1/2)),int(entity1_center[1]+(h1/2))]]

    # img = cv2.rectangle(img, tuple([entity1_center[0] - int(w1/2),entity1_center[1] - int(h1/2)]),tuple([entity1_center[0] + int(w1/2),entity1_center[1] + int(h1/2)]),(0,255,0),1)

    
    w2 = text2_shape[0]
    h2 = text2_shape[1]
    # iteratively place cluster entities
    for idx in range(len(placed_entities2)):

        current_entitiy = placed_entities2[idx]

        current_entitiy.textbox_background = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        current_entitiy.text_color = (255 - current_entitiy.textbox_background[0], 255 - current_entitiy.textbox_background[1], 255 - current_entitiy.textbox_background[2])

        current_center = [current_entitiy.center[0] + entity2_center[0] - int(w2/2),current_entitiy.center[1] + entity2_center[1] - int(h2/2)]
        img, tmp_bbox = draw_textbox(self,img,current_entitiy,current_center)

        placed_entities2[idx].bbox = tmp_bbox

        # img = cv2.rectangle(img, tuple([int(current_center[0]-current_entitiy['width']*.8),current_center[1]-current_entitiy['height']]),tuple([int(current_center[0]+current_entitiy['width']*.8),current_center[1]+current_entitiy['height']]),(255,0,0),1)

    # img = cv2.circle(img,tuple(entity2_center),2,(255,0,0),1)
 
    w2 = text2_shape[0]
    h2 = text2_shape[1]
    entity2_bbox = [[int(entity2_center[0]-(w2/2)),int(entity2_center[1]-(h2/2))],[int(entity2_center[0]+(w2/2)),int(entity2_center[1]+(h2/2))]]

    # img = cv2.rectangle(img, tuple([entity2_center[0] - int(w2/2),entity2_center[1] - int(h2/2)]),tuple([entity2_center[0] + int(w2/2),entity2_center[1] + int(h2/2)]),(0,255,0),1)

    # force anchors to be outside of cluster bboxes

    try:
        x_span,y_span = get_spline_anchors(self,entity1_center,entity2_center,entity1_bbox,entity2_bbox,entity_configuration)
    except Exception as e:
        print("BAHAHAH")
        print(e)
        raise ValueError


    img, f, orientation, spline_bbox = draw_spline(self,img,x_span,y_span)


    img, indicator_bbox = draw_indicator(self,img,x_span,y_span,f,orientation)


    min_x = int(indicator_bbox[0][0]) - 5
    min_y = int(indicator_bbox[0][1]) - 5
    max_x = int(indicator_bbox[1][0]) + 5
    max_y = int(indicator_bbox[1][1]) + 5
    indicator_bbox = [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]

    # get final relationship bbox by taking max dims of spline and indicator
    min_x = int(min([spline_bbox[0][0],indicator_bbox[0][0]]))
    min_y = int(min([spline_bbox[0][1],indicator_bbox[0][1]]))
    max_x = int(max([spline_bbox[1][0],indicator_bbox[1][0]]))
    max_y = int(max([spline_bbox[1][1],indicator_bbox[1][1]]))


    # if want relationship_bbox instead of indicator just return this in its place
    relationship_bbox = [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]


    
    
    

    return img,indicator_bbox,placed_entities1,placed_entities2

def radial_profile(data, center):

    """

        generates radial profile on given image

        Args:
            data (np.Array): input 'image'
            center (list): center coordinates [y,x] of 'image'
        
        Return:
            radialprofile (np.Array): contains normalized sum of binned pixel values from center by radius 
            
    """

    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    # sum over pixels the same radius away from center
    tbin = np.bincount(r.ravel(), data.ravel())

    # normalize by distance to center, since as radius grows the # of pixels in radius bin does also
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    radialprofile = radialprofile * 255

    return radialprofile 


# x,y now define a center
def check_slice(template_im,slice_shape,x,y,padding=0):

    """

        check if region on image is a location with few high frequency elements

        Args:
            template_im (np.Array): template image
            slice_shape (list): contains dimensions of target region as [x_dim,y_dim]
            x (int): interested location top-left corner x
            y (int): interested location top-left corner y
            padding (int): optional extra spacing arround region  
        
        Return:
            (bool): indicates if region is good or not
            
    """

    threshold = 10

    template_slice = template_im[y-padding:y+slice_shape[1]+padding,x-padding:x+slice_shape[0]+padding,:]

    # if all pixels are the same, then don't even have to run the rest of check
    if np.all(template_slice == template_slice[0,0,0]):
        return True

    grey_slice = cv2.cvtColor(template_slice, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(grey_slice)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum *= 1.0 / magnitude_spectrum.max() 

    # x,y format
    center = (int(slice_shape[1] / 2),int(slice_shape[0] / 2))

    # get description of fft emitting from center
    radial_prof = radial_profile(magnitude_spectrum, center)

    radial_prof[radial_prof == -inf] = 0

    idx = range(0,radial_prof.shape[0])
    bin_means = stats.binned_statistic(idx, radial_prof, 'mean', bins=4)[0]

    # cv2.imshow('image',template_slice)
    # cv2.waitKey(0)

    # cv2.imshow('grey image',grey_slice)
    # cv2.waitKey(0)

    # cv2.imshow('image2',magnitude_spectrum)
    # cv2.waitKey(0)

    # xs = np.array(range(radial_prof.shape[0]))
    # plt.plot(xs, radial_prof)
    # plt.show()

    if bin_means[-1] < threshold and bin_means[-2] < threshold:
        return True
    else:
        return False

                    
def get_entity_placement(slice_shape,x_target,y_target,text1_shape,text2_shape,template_im):

    """

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
    """

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width
    (w1, h1) = text1_shape
    (w2, h2) = text2_shape

    # 4 configurations/positioning: hotdog, hamburger, square1, square2
    dim_ratio = slice_shape[0] / slice_shape[1]
    if dim_ratio > 1.67:

        # print('cluster width')
        # print(w1)

        # LONG
        entity1_center_y = math.floor(slice_shape[1] / 2) + y_target
        entity1_center_x = x_target + slice_shape[0] - math.floor(w1/2)

        # template_im = cv2.circle(template_im, (entity1_center_x,entity1_center_y), math.floor(w1/2), self.arrow_color, -1)  

        # template_im = cv2.circle(template_im, (x_target,y_target), 20, self.arrow_color, -1)  

        entity2_center_y = entity1_center_y
        entity2_center_x = x_target + math.floor(w2/2)



        entity_configuration = LONG


    elif dim_ratio < .6:
        # TALL
        entity1_center_x = math.floor(slice_shape[0] / 2) + x_target
        entity1_center_y = y_target + math.floor(h1/2)

        entity2_center_x = entity1_center_x
        entity2_center_y = y_target + slice_shape[1] - math.floor(h2/2)

        entity_configuration = TALL


    else:
        # DOWN_SLASH
        if np.random.randint(2):
            entity1_center_x = x_target + math.floor(w1/2)
            entity1_center_y = y_target + math.floor(h1/2)

            entity2_center_x = x_target + slice_shape[0] - math.floor(w2/2)
            entity2_center_y = y_target + slice_shape[1] - math.floor(h2/2)

            entity_configuration = DOWN_SLASH
        # UP_SLASH
        else:
            entity1_center_x = x_target + slice_shape[0] - math.floor(w1/2)
            entity1_center_y = y_target + math.floor(h1/2)

            entity2_center_x = x_target + math.floor(w2/2)
            entity2_center_y = y_target + slice_shape[1] - math.floor(h2/2)

            entity_configuration = UP_SLASH


    entity1_center = [entity1_center_x,entity1_center_y]
    entity2_center = [entity2_center_x,entity2_center_y]

    return entity1_center, entity2_center, entity_configuration,template_im



class template_thread(threading.Thread):
    def __init__(self, threadID,name,template_list,directory):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.template_list = template_list
        self.directory = directory

    def run(self):

        """

        Start 4 threads for generating x# samples from same templates at once 

        """

        filename = self.template_list[self.threadID]
        
        # how many images per template
        stop_child_flag = False
        num_copies = 1024
        for copy_idx in range(0,num_copies):

            # children in here
            child_thread_idx = copy_idx * 4
            # other parent threads
            child_thread_idx = self.threadID*num_copies + child_thread_idx

            if stop_child_flag:
                break

            child_thread0 = copy_thread(child_thread_idx,"child0",self.directory,filename)
            # child_thread1 = copy_thread(child_thread_idx+1,"child1",self.directory,filename)
            # child_thread2 = copy_thread(child_thread_idx+2,"child2",self.directory,filename)
            # child_thread3 = copy_thread(child_thread_idx+3,"child3",self.directory,filename)

            child_thread0.start()
            # if (copy_idx*4) + 1 > num_copies:
            #     stop_child_flag = True
            #     continue
            # else:
            #     child_thread1.start()
            # if (copy_idx*4) + 2 > num_copies:
            #     stop_child_flag = True
            #     continue
            # else:
            #     child_thread2.start()
            # if (copy_idx*4) + 3 > num_copies:
            #     stop_child_flag = True
            #     continue
            # else:
            #     child_thread3.start()

            child_thread0.join()
            # child_thread1.join()
            # child_thread2.join()
            # child_thread3.join()
            break

def get_entities(self,num_entities,img):

    # get cluster entities to place
    c_entities = []
    for c_idx in range(num_entities):
        
        # get text
        cluster_str_len = np.random.randint(3,7)
        cluster_label = randomword(cluster_str_len).upper()

        # instantiate shape
        # entity_type = random.choice([RECT, ELLIPSE, POLYGON])
        entity_type = POLYGON
        if entity_type == RECT:
            c_entity = Synthetic_Rectangle([0,0],cluster_label)
        elif entity_type == ELLIPSE:
            c_entity = Synthetic_Ellipse([0,0],cluster_label)
        elif entity_type == POLYGON:
            c_entity = Synthetic_Shape([0,0],cluster_label)

        # load font and get text size
        fontpath = os.path.join(self.font_folder, self.font_style)    
        font = ImageFont.truetype(fontpath, c_entity.font_size)
        (c_w1, c_h1) = font.getsize(cluster_label)

        c_entity.width = c_w1
        c_entity.height = c_h1
        c_entity.font = font

        c_entities.append(copy.deepcopy(c_entity))


    placed_entities = [c_entities[0]]
    entity1 = c_entities[0]


    for current_entitiy in c_entities[1:]:

        new_center,img = draw_cluster(entity1,current_entitiy,placed_entities,img)
        current_entitiy.center = new_center
        placed_entities.append(current_entitiy)


    # get shape of cluster
    min_x = 10000
    max_x = -10000
    min_y = 100000
    max_y = -100000
    sum_x = 0
    sum_y = 0
    for entity in c_entities:

        # TODO:: may be different for other shapes
        # **********************************************************************************************************
        tmp_min_x, tmp_max_x, tmp_min_y, tmp_max_y = entity.get_min_max()


        if tmp_min_x < min_x:
            min_x = tmp_min_x
        if tmp_min_y < min_y:
            min_y = tmp_min_y
        if tmp_max_x > max_x:
            max_x = tmp_max_x
        if tmp_max_y > max_y:
            max_y = tmp_max_y

        sum_x += entity.center[0]
        sum_y += entity.center[1]

    # adjust reference from first entity to top-left corner
    avg_x = sum_x / len(c_entities)
    avg_y = sum_y / len(c_entities)
    for idx in range(len(c_entities)):
        c_entities[idx].center[0] = int(c_entities[idx].center[0] + abs(min_x))
        c_entities[idx].center[1] = int(c_entities[idx].center[1] + abs(min_y))
        
    
    w1 = int(abs(min_x) + abs(max_x))
    h1 = int(abs(min_y) + abs(max_y))
    text_shape = [w1,h1]

    return placed_entities, text_shape, img

def set_relationship_config(self):
    
    self.thickness = random.randint(1, 3)
    self.tip_len = random.randint(5, 15)
    self.base_len = random.randint(10, 20)
    
    self.arrow_placement = random.choice([START, END])
    # self.arrow_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    self.arrow_color = (0, 0, 0)
    # self.text_color = (0,0,0)
    # self.indicator = random.choice([INHIBIT, ACTIVATE, INDIRECT_INHIBIT, INDIRECT_ACTIVATE])
    self.indicator = random.choice([INHIBIT, ACTIVATE, INDIRECT_INHIBIT, INDIRECT_ACTIVATE])
    self.arch_ratio = 0.1
    self.spline_type = LINE

    return self

def set_text_config(self):

    self.font_folder = "font_folder"
    font_style_list = os.listdir(self.font_folder)
    self.font_style = random.choice(font_style_list)

    return self

# TODO:: don't have text_margin effect bbox annotation for text
class copy_thread(threading.Thread):
    def __init__(self,copyID,name,directory,filename):
        threading.Thread.__init__(self)
        self.copyID = copyID
        self.name = name
        self.directory = directory
        self.filename = filename

    def run(self):

        """

        Attempt to place x# of samples on a template, save new image, and save annotation

        """

         # TODO:: to get boarder on spline, just do thickness + 1 and don't fill, then run back over with different color at thickness and fill

        # TODO:: do this random selection every placement
        self.padding = 30

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
        for relation_idx in range(50):

            # change to remove setting CORNER
            # make each entity config has set of possible arrows (eg. hotog has line and arch/ square has line, arch, and corner)

            # TODO:: make set of names to pull from or characters
            # TODO:: make background textbox color change, focus on light colors
            # TODO:: include more dynamic variations of textbox (oval, no textbox)
            # TODO:: dynamically change indicator length and width

            self = set_text_config(self)

            num_entities = np.random.randint(1,5)
            # num_entities = 1
            # text shape is actually cluster shape dimensions
            placed_entities1, text1_shape,template_im = get_entities(self,num_entities,template_im)
            w1 = text1_shape[0]
            h1 = text1_shape[1]

            num_entities = np.random.randint(1,5)
            # num_entities = 1
            placed_entities2, text2_shape,template_im = get_entities(self,num_entities,template_im)
            w2 = text2_shape[0]
            h2 = text2_shape[1]


            # y_dim dependent on x_dim value
            # TODO:: probably want to change this to make short tall and longs more likely
            dist1 = np.random.randint(0,150)
            x_dim = dist1 + w1 + w2
            y_dim = np.random.randint(150-dist1,151) + h1 + h2
            slice_shape = [x_dim,y_dim]

            # check if queried coords are a valid location
            for idx in range(50):

                # subtracted max bounds to ensure valid coords

                #// low>= high value error
                # top left corner target
                # print(template_im.shape)
                # print(slice_shape)
                x_target = np.random.randint(0+self.padding,template_im.shape[1]-slice_shape[0]-self.padding)
                y_target = np.random.randint(0+self.padding,template_im.shape[0]-slice_shape[1]-self.padding)
                
                # check if selected template area is good
                if check_slice(template_im,slice_shape,x_target,y_target,self.padding):

                    # template_slice = template_im[y-padding:y+slice_shape[1]+padding,x-padding:x+slice_shape[0]+padding,:]

                    # template_im = cv2.rectangle(template_im, (x_target-self.padding, y_target-self.padding), (x_target+x_dim+self.padding, y_target+y_dim+self.padding), (0,0,0), 1)
                    # rows,cols = draw.rectangle_perimeter((x_target-self.padding, y_target-self.padding),(x_target+x_dim+self.padding, y_target+y_dim+self.padding))

                    self.spline_type = LINE

                    self = set_relationship_config(self)
                    entity1_center,entity2_center,entity_configuration,template_im = get_entity_placement(slice_shape,x_target,y_target,(w1,h1),(w2,h2),template_im)
                    
                    try:
                        template_im,relationship_bbox,placed_entities1,placed_entities2 = draw_relationship(self,template_im,entity1_center,entity2_center,entity_configuration,placed_entities1,placed_entities2,text1_shape,text2_shape)
                    except Exception as e: 
                        print("CATS")
                        print(e)
                        continue

                    # cv2.imshow('image3',template_im)
                    # cv2.waitKey(0)

                    shapes_1 = []
                    for entity in placed_entities1:

                        min_x = entity.bbox[0][0]
                        min_y = entity.bbox[0][1]
                        max_x = entity.bbox[1][0]
                        max_y = entity.bbox[1][1]
                        label1_bbox = [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]

                        label1_shape = copy.deepcopy(base_shape)
                        label1_shape['points'] = label1_bbox
                        label1_shape['ID'] = element_indx
                        label1_shape['label'] = str(element_indx) + ":gene:" + entity.label
                        element_indx += 1
                        shapes_1.append(label1_shape)

                    shapes_2 = []
                    for entity in placed_entities2:

                        min_x = entity.bbox[0][0]
                        min_y = entity.bbox[0][1]
                        max_x = entity.bbox[1][0]
                        max_y = entity.bbox[1][1]
                        label2_bbox = [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]


                        label2_shape = copy.deepcopy(base_shape)
                        label2_shape['points'] = label2_bbox
                        label2_shape['ID'] = element_indx
                        label2_shape['label'] = str(element_indx) + ":gene:" + entity.label
                        element_indx += 1
                        shapes_2.append(label2_shape)



                    shapes_1_str = '['
                    if len(shapes_1) == 1:
                        shapes_1_str = str(shapes_1[0]['ID'])
                    else:
                        for shape in shapes_1:
                            shapes_1_str = shapes_1_str + str(shape['ID']) + ','
                        shapes_1_str = shapes_1_str[:-1] + ']'


                    shapes_2_str = '['
                    if len(shapes_2) == 1:
                        shapes_2_str = str(shapes_2[0]['ID'])
                    else:
                        for shape in shapes_2:
                            shapes_2_str = shapes_2_str + str(shape['ID']) + ','
                        shapes_2_str = shapes_2_str[:-1] + ']'


                    if self.arrow_placement == START:
                        id2id_str = shapes_2_str + "|" + shapes_1_str
                    else:
                        id2id_str = shapes_1_str + "|" + shapes_2_str




                    indicator_shape = copy.deepcopy(base_shape)
                    indicator_shape['points'] = relationship_bbox
                    indicator_shape['ID'] = element_indx
                    if self.indicator == INHIBIT:
                        indicator_shape['label'] = str(element_indx) + ":inhibit:" + id2id_str
                    elif self.indicator == ACTIVATE:
                        indicator_shape['label'] = str(element_indx) + ":activate:" + id2id_str
                    elif self.indicator == INDIRECT_ACTIVATE:
                        indicator_shape['label'] = str(element_indx) + ":indirect_activate:" + id2id_str
                    else:
                        indicator_shape['label'] = str(element_indx) + ":indirect_inhibit:" + id2id_str
                    element_indx += 1


                    # TODO:: loop through each shape
                    for shape in shapes_1:
                        shapes.append(shape)
                    for shape in shapes_2:
                        shapes.append(shape)
                    shapes.append(indicator_shape)

                    break
            # break

            # # TODO:: these boxes do not match with fig
            # template_im = cv2.rectangle(template_im, (x_target, y_target), (x_target+x_dim, y_target+y_dim), (0,0,0), 1)
            # if relation_idx == 2:
            #     break
            # break

        # save json and new image
        im_dir = "output_test"
        image_path = str(self.copyID) + ".png"
        cv2.imwrite(os.path.join(im_dir, image_path), template_im)
        imageHeight = template_im.shape[0]
        imageWidth = template_im.shape[1]
        template_label_file = label_file.LabelFile()
        template_label_file.save(os.path.join(im_dir,str(self.copyID) + ".json"),shapes,image_path,imageHeight,imageWidth)

def populate_figures():

    """

    Start multiple threads for generating samples from 4 different templates at once

    """

    # loop through all templates
    stop_flag = False
    directory = "templates"
    filename_list = os.listdir(directory)
    template_list = []
    for file in filename_list:
        if '.DS_' not in file:
            template_list.append(file)

    for template_idx in range(0,len(template_list)-1,4):

        if stop_flag:
            break

        thread0 = template_thread(template_idx,"thread-0",template_list,directory)
        # thread1 = template_thread(template_idx+1,"thread-1",template_list,directory)
        # thread2 = template_thread(template_idx+2,"thread-2",template_list,directory)
        # thread3 = template_thread(template_idx+3,"thread-3",template_list,directory)

        thread0.start()
        # if template_idx + 1 > len(template_list):
        #     stop_flag = True
        #     continue
        # else:
        #     thread1.start()
        # if template_idx + 2 > len(template_list):
        #     stop_flag = True
        #     continue
        # else:
        #     thread2.start()
        # if template_idx + 3 > len(template_list):
        #     stop_flag = True
        #     continue
        # else:
        #     thread3.start()

        thread0.join()
        # thread1.join()
        # thread2.join()
        # thread3.join()
        break



if __name__ == "__main__":

    # TODO:: lower high-freq threshold based on colors being used for entities and arrows
    # another interesting idea would be to include targeted noise (i.e. lines with no indicator connecting no entities)
    populate_figures()
