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
from detectron2.structures import Instances, RotatedBoxes,BoxMode, Boxes
import torch
import skimage.draw as draw
from PIL import ImageFont, ImageDraw, Image
from synthetic_shapesv2 import Synthetic_Shape, Synthetic_Arrow
import warnings



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

DOWN_SLASH = 2
UP_SLASH = 3
UP_DOWN = 4
LEFT_RIGHT = 5


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
    # lag = int(0.1 * max_idx)
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
    except:
        comparison_pt = None

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
    spl = make_interp_spline(param, np.c_[x_span,y_span], k=spline_coef)

    # TODO:: not sure what multiplier to use here
    X_, Y_ = spl(np.linspace(0, 1, x_span.size * 400)).T

    X_ = np.around(X_, 0).astype(int)
    Y_ = np.around(Y_, 0).astype(int)

    base_points = np.stack((X_,Y_)).T
    _,base_indeces = np.unique(base_points, return_index=True, axis=0)
    base_points = base_points[np.sort(base_indeces)]

    base_points_x = base_points[:,0]
    min_x = np.min(base_points_x)
    max_x = np.max(base_points_x)

    base_points_y = base_points[:,1]
    min_y = np.min(base_points_y)
    max_y = np.max(base_points_y)

    if math.dist([min_x,min_y],[max_x,max_y]) < 50:
        if self.indicator == INDIRECT_ACTIVATE:
            self.indicator = ACTIVATE
        elif self.indicator == INDIRECT_INHIBIT:
            self.indicator = INHIBIT

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
        if self.thickness > 0:
            img = cv2.circle(img, (x,y), self.thickness, self.arrow_color, -1)
        else:
            img[y,x,:] = np.array(self.arrow_color).astype(np.uint8)

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



    # get shape
    tmp_img = np.ones((500,500,3)) * 255
    if self.indicator == INHIBIT or self.indicator == INDIRECT_INHIBIT:

        pt1 = [250-self.base_len, 250]
        pt3 = [250+self.base_len, 250]

        triangle_cnt = np.array( [tuple(pt1), tuple(pt3)] )
        cv2.drawContours(tmp_img, [triangle_cnt], 0, self.arrow_color, self.thickness+1)

    else:

        my_arrow = Synthetic_Arrow(self.thickness)
        tmp_img = my_arrow.draw_shape(tmp_img,[250,250])
        my_arrow.center = tri_source


    # if slope is extreme, then just place simple inicator in cardinal direction
    if math.isnan(tip_slope) or abs(tip_slope) > 15 or abs(tip_slope) < 0.05:

        angle = 0
        if arrow_orientation == DOWN:
            angle = 180
        elif arrow_orientation == LEFT:
            angle = 90
        elif arrow_orientation == RIGHT:
            angle = 270
                
        image_center = ((tmp_img.shape[1] - 1) / 2, (tmp_img.shape[0] - 1) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        tmp_img = cv2.warpAffine(tmp_img, rot_mat, tmp_img.shape[1::-1], flags=cv2.INTER_NEAREST)

        tmp_img = tmp_img[200:300,200:300,:]

        blk_idx = np.array(np.where((tmp_img!=[255,255,255]).all(axis=2))) - 50
        blk_idx = (blk_idx[0]+tri_source[1],blk_idx[1]+tri_source[0])

        img[blk_idx] = list(self.arrow_color)
    

    # if slope is 'soft', then calculate appropriate indicator orientation
    else:

        # TODO:: debug problem w/ tip slope 
        angle = 90
        if arrow_orientation == UP:
            if tip_slope > 0:
                angle = 270
        elif arrow_orientation == DOWN:
            if tip_slope < 0:
                angle = 270
        elif arrow_orientation == RIGHT:
            angle = 270

        tip_angle = angle + math.degrees(math.atan(tip_slope))

        image_center = (250, 250)

        rot_mat = cv2.getRotationMatrix2D(image_center, tip_angle, 1.0)
        tmp_img = cv2.warpAffine(tmp_img, rot_mat, tmp_img.shape[1::-1], flags=cv2.INTER_LINEAR)


        tmp_img = tmp_img[200:300,200:300,:]

        blk_idx = np.array(np.where((tmp_img!=[255,255,255]).all(axis=2))) - 50
        blk_idx = (blk_idx[0]+tri_source[1],blk_idx[1]+tri_source[0])

        img[blk_idx] = list(self.arrow_color)

        

    # get bbox dims for corresponding indicator
    if self.indicator == INHIBIT or self.indicator == INDIRECT_INHIBIT:

        blk_idx = np.array(blk_idx)
        mins = np.min(blk_idx,axis=1)
        maxes = np.max(blk_idx,axis=1)

        min_x = mins[1]
        min_y = mins[0]

        max_x = maxes[1]
        max_y = maxes[0]
    else:
        min_x, max_x, min_y, max_y = my_arrow.get_min_max()

    indicator_bbox = [[min_x,min_y],[max_x,max_y]]




    

    return img, indicator_bbox


def draw_textbox(img,current_entitiy):

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
    # get text label and textbox dims
    label = current_entitiy.label
    w = current_entitiy.width
    h = current_entitiy.height

    # draw entity shape
    img = current_entitiy.draw_shape(img,current_entitiy.center)

    # x1,y1 is top left corner of textbox w/ textmargin
    x1 = current_entitiy.center[0] - int(round(w/2)) - current_entitiy.text_margin
    y1 = current_entitiy.center[1] - int(round(h/2)) - current_entitiy.text_margin

    # draw text
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x1 + current_entitiy.text_margin, y1 + h + current_entitiy.text_margin), label, font=current_entitiy.font, fill=current_entitiy.text_color, anchor='lb')
    img = np.array(img_pil)

    # calculate textbox label
    bbox = [[x1+current_entitiy.text_margin,y1+current_entitiy.text_margin],[x1 + current_entitiy.text_margin + w, y1 + current_entitiy.text_margin + h]]

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

def get_start_end_points(cluster1_center,cluster2_center,cluster1_bbox,cluster2_bbox):

    # TODO:: set num in linspace to be dependent on distance from one center to another
    canidate_points = np.linspace(cluster1_center, cluster2_center, num=50, dtype=np.int)

    # get start point for spline
    # check for first nth point outside of entity boxes in direct line to other entity
    tar_point = 5
    count = 0
    start_point = None
    for idx in range(canidate_points.shape[0]-1):
        current_point = canidate_points[idx,:]
        # check if point is in box
        if current_point[0] < cluster1_bbox[0][0] or current_point[1] < cluster1_bbox[0][1] or current_point[0] > cluster1_bbox[1][0] or current_point[1] > cluster1_bbox[1][1]:
            count += 1
            if count == tar_point:
                start_point = [math.floor(current_point[0]),math.floor(current_point[1])]
    
    if not start_point:
        start_point = [math.floor(canidate_points[15][0]),math.floor(canidate_points[15][1])]

    # get end point for spline
    count = 0
    end_point = None
    for idx in range(canidate_points.shape[0]-1,0,-1):
        current_point = canidate_points[idx,:]
        if current_point[0] < cluster2_bbox[0][0] or current_point[1] < cluster2_bbox[0][1] or current_point[0] > cluster2_bbox[1][0] or current_point[1] > cluster2_bbox[1][1]:
            count += 1
            if count == tar_point:
                end_point = [math.floor(current_point[0]),math.floor(current_point[1])]

    if not end_point:
        end_point = [math.floor(canidate_points[-15][0]),math.floor(canidate_points[-15][1])]


    return start_point, end_point
    

def get_spline_anchors(self,cluster1_center,cluster2_center,cluster1_bbox,cluster2_bbox,entity_configuration):

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

    # TODO:: make classes for structures
    # TODO:: do this better, maybe set multimodal distribution to pull from
    rand_int = np.random.randint(3)
    if rand_int == 0 and abs(cluster1_center[0]-cluster2_center[0]) > 70 and abs(cluster1_center[1]-cluster2_center[1]) > 70 and entity_configuration != UP_DOWN and entity_configuration != LEFT_RIGHT:
        # CORNER
        self.spline_type = CORNER
        spline_points = get_corner_anchors(entity_configuration,cluster1_center,cluster2_center,cluster1_bbox,cluster2_bbox)
    elif rand_int == 1:
        # ARCH
        start_point, end_point = get_start_end_points(cluster1_center,cluster2_center,cluster1_bbox,cluster2_bbox)
        self.spline_type = ARCH
        spline_points = get_arch_anchors(self,start_point,end_point)

    else:
        # LINE
        start_point, end_point = get_start_end_points(cluster1_center,cluster2_center,cluster1_bbox,cluster2_bbox)
        self.spline_type = LINE
        spline_points = np.linspace(start_point, end_point, num=4,dtype=np.int)

    

    x_span = spline_points[:,0]
    y_span = spline_points[:,1]

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

def get_cluster_arrangement(current_entitiy,placed_entities):

    '''
    get placement of new entity in cluster
    '''



    entity1 = placed_entities[0]
    entity1_center = entity1.center

    # get shape 1 and 2
    shape1_x,shape1_y = entity1.get_points()

    # get placed entities cluster mask
    mask_dim = 1000
    cluster_mask = np.zeros((mask_dim,mask_dim,1), np.float32)
    for entity in placed_entities:
        cluster_mask = entity.draw_shape(cluster_mask.astype(np.uint8),[entity.center[0]+(mask_dim/2),entity.center[1]+(mask_dim/2)],(255,255,255),(255,255,255),fill=True)

    cv2.imwrite('mask.jpg',cluster_mask)
    # cv2.waitKey(0)

    # TODO:: problem of unbalance direction sampling from sparse set of points from get_points()
    # select point from first entity in cluster and keep going in that dir until out of cluster
    tmp_idx = random.randint(0,len(shape1_x)-1)
    factor = 1.0
    while True:
        
        # get new entity mask
        new_entity_mask = np.zeros((mask_dim,mask_dim,1), np.float32)
        test_center = [int(shape1_x[tmp_idx]*factor)+entity1_center[0],int(shape1_y[tmp_idx]*factor)+entity1_center[1]]
        new_entity_mask = current_entitiy.draw_shape(new_entity_mask.astype(np.uint8),[test_center[0]+(mask_dim/2),test_center[1]+(mask_dim/2)],(255,255,255),(255,255,255),fill=True)
        
        # check iou
        mul_mask = np.multiply(cluster_mask.astype(np.int32),new_entity_mask.astype(np.int32))
        where_0 = np.where(mul_mask > 255)
        mul_mask[where_0] = 255

        # if no overlap break out
        if np.all(mul_mask == 0):
            break

        cv2.imwrite(str(factor)+'mask.jpg',mul_mask.astype(np.uint8))
        # cv2.waitKey(0)

        # cv2.imshow('mask'+str(factor)+'.jpg',mul_mask.astype(np.float16))
        # cv2.waitKey(0)
        

        factor += 0.1

    new_center = [int(shape1_x[tmp_idx]*factor)+entity1_center[0],int(shape1_y[tmp_idx]*factor)+entity1_center[1]]

    return new_center

def draw_relationship(self,img,cluster1_center,cluster2_center,entity_configuration,placed_entities1,placed_entities2,cluster1_shape,cluster2_shape):

    """

        draw entities, draw spline, draw indicator

        Args:
            self: contains hyperparameter config
            img (np.Array): copied template image for stitching
            cluster1_center (list): contains target center of cluster1 bbox as [entity1_center_x,entity1_center_y]
            cluster2_center (list): contains target center of cluster2 bbox as [entity2_center_x,entity2_center_y]
            cluster1_shape (list): contains target dimensions of text to place as [w1,h1]
            cluster2_shape (list): contains target dimensions of text to place as [w2,h2]  
        
        Return:
            img (np.Array): updated image with entities, spline, and indicator drawn on it
            relationship_bbox (list): 2D-list with bbox corners for spline and indicator as [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]
            
    """

    
    w1 = cluster1_shape[0]
    h1 = cluster1_shape[1]
    # iteratively place cluster entities
    for idx in range(len(placed_entities1)):
        img, tmp_bbox = draw_textbox(img,placed_entities1[idx])
        placed_entities1[idx].bbox = tmp_bbox

    cluster1_bbox = [[int(cluster1_center[0]-(w1/2)),int(cluster1_center[1]-(h1/2))],[int(cluster1_center[0]+(w1/2)),int(cluster1_center[1]+(h1/2))]]


    w2 = cluster2_shape[0]
    h2 = cluster2_shape[1]
    # iteratively place cluster entities
    for idx in range(len(placed_entities2)):
        img, tmp_bbox = draw_textbox(img,placed_entities2[idx])
        placed_entities2[idx].bbox = tmp_bbox

    cluster2_bbox = [[int(cluster2_center[0]-(w2/2)),int(cluster2_center[1]-(h2/2))],[int(cluster2_center[0]+(w2/2)),int(cluster2_center[1]+(h2/2))]]

    # force anchors to be outside of cluster bboxes
    x_span,y_span = get_spline_anchors(self,cluster1_center,cluster2_center,cluster1_bbox,cluster2_bbox,entity_configuration)

    img, f, orientation, spline_bbox = draw_spline(self,img,x_span,y_span)

    img, indicator_bbox = draw_indicator(self,img,x_span,y_span,f,orientation)

    # get indiactor bbox
    min_x = int(indicator_bbox[0][0]) - 3
    min_y = int(indicator_bbox[0][1]) - 3
    max_x = int(indicator_bbox[1][0]) + 3
    max_y = int(indicator_bbox[1][1]) + 3
    indicator_bbox = [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]

    # get final relationship bbox by taking max dims of spline and indicator
    min_x = int(min([spline_bbox[0][0]-5,indicator_bbox[0][0]]))
    min_y = int(min([spline_bbox[0][1]-5,indicator_bbox[0][1]]))
    max_x = int(max([spline_bbox[1][0]+5,indicator_bbox[2][0]]))
    max_y = int(max([spline_bbox[1][1]+5,indicator_bbox[2][1]]))

    # if want relationship_bbox instead of indicator just return this in its place
    relationship_bbox = [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]

    # img = cv2.rectangle(img, (indicator_bbox[0][0],indicator_bbox[0][1]), (indicator_bbox[2][0],indicator_bbox[2][1]), (255,0,0), 2)
    # img = cv2.rectangle(img, (relationship_bbox[0][0],relationship_bbox[0][1]), (relationship_bbox[2][0],relationship_bbox[2][1]), (0,255,0), 2)

    return img,relationship_bbox,placed_entities1,placed_entities2
 
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

    threshold = 3

    template_slice = template_im[y-padding:y+slice_shape[1]+padding,x-padding:x+slice_shape[0]+padding,:]

    # if all pixels are the same, then don't even have to run the rest of check
    if np.all(template_slice == template_slice[0,0,0]):
        return True

    # convert to frequency domain
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

    if bin_means[-1] < threshold and bin_means[-2] < threshold:
        return True
    else:
        return False

                    
def get_entity_placement(slice_shape,x_target,y_target,cluster1_shape,cluster2_shape,placed_entities1,placed_entities2):

    """

        find positions to place entities

        Args:
            slice_shape (list): contains dimensions of target region as [x_dim,y_dim]
            x_target (int): top-left corner x placement location 
            y_target (int): top-left corner y placement location 
            cluster1_shape (tuple): (w,h) of cluster 1
            cluster2_shape (tuple): (w,h) of cluster 2     
        
        Return:
            entity1_center (list): contains target center of entity1 bbox as [entity1_center_x,entity1_center_y]
            entity2_center (list): contains target center of entity2 bbox as [entity2_center_x,entity2_center_y]
    """

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width
    (w1, h1) = cluster1_shape
    (w2, h2) = cluster2_shape

    # 2 configurations/positioning: square1, square2
    # DOWN_SLASH
    slice_dim_ratio = slice_shape[0] / slice_shape[1]

    
    
    # top mid & bottom mid
    pos_rand = np.random.randint(2)
    if slice_dim_ratio < 0.9 and pos_rand:
        cluster1_center_x = x_target + (slice_shape[0] / 2)
        cluster1_center_y = y_target + math.floor(h1/2)

        cluster2_center_x = x_target + (slice_shape[0] / 2)
        cluster2_center_y = y_target + slice_shape[1] - math.floor(h2/2)

        entity_configuration = UP_DOWN

    elif slice_dim_ratio > 1.6 and pos_rand:

        cluster1_center_x = x_target + math.floor(w1/2)
        cluster1_center_y = y_target + (slice_shape[1] / 2)

        cluster2_center_x = x_target + slice_shape[0] - math.floor(w2/2)
        cluster2_center_y = y_target + (slice_shape[1] / 2)

        entity_configuration = LEFT_RIGHT
        
    elif pos_rand:
        cluster1_center_x = x_target + math.floor(w1/2)
        cluster1_center_y = y_target + math.floor(h1/2)

        cluster2_center_x = x_target + slice_shape[0] - math.floor(w2/2)
        cluster2_center_y = y_target + slice_shape[1] - math.floor(h2/2)

        entity_configuration = DOWN_SLASH
    # UP_SLASH
    else:
        cluster1_center_x = x_target + slice_shape[0] - math.floor(w1/2)
        cluster1_center_y = y_target + math.floor(h1/2)

        cluster2_center_x = x_target + math.floor(w2/2)
        cluster2_center_y = y_target + slice_shape[1] - math.floor(h2/2)

        entity_configuration = UP_SLASH



    cluster1_center = [cluster1_center_x,cluster1_center_y]
    cluster2_center = [cluster2_center_x,cluster2_center_y]

    # update entity centers to be relative to image
    for entity in placed_entities1:
        entity.center = [entity.center[0] + cluster1_center[0] - int(w1/2),entity.center[1] + cluster1_center[1] - int(h1/2)]
    for entity in placed_entities2:
        entity.center = [entity.center[0] + cluster2_center[0] - int(w2/2),entity.center[1] + cluster2_center[1] - int(h2/2)]

    return cluster1_center, cluster2_center, entity_configuration

def get_snr1(copy_template,template_im):
    # average pixel difference

    # SNR 1
    im_dif = np.abs(copy_template - template_im)
    dif_sum = np.sum(im_dif.flatten()) 
    snr = dif_sum / (template_im.shape[0]*template_im.shape[1]*3)

    return snr

    

def get_snr2(copy_template,template_im):

    # SNR 2
    im_dif = copy_template - template_im
    dif_sq = np.square(im_dif)
    dif_sum = np.sum(dif_sq.flatten().astype(np.float64)) 

    numerator = np.sum(np.square(copy_template.flatten().astype(np.float64))) 

    snr = numerator / dif_sum
    return snr

def get_entities(self,num_entities):

    # get cluster entities to place
    c_entities = []
    for c_idx in range(num_entities):
        
        # get text
        cluster_str_len = np.random.randint(3,7)
        cluster_label = randomword(cluster_str_len).upper()

        # instantiate shape
        c_entity = Synthetic_Shape([0,0],cluster_label)

        # load font and get text size
        fontpath = os.path.join(self.font_folder, self.font_style)    
        font = ImageFont.truetype(fontpath, c_entity.font_size)
        (c_w1, c_h1) = font.getsize(cluster_label)

        c_entity.width = c_w1
        c_entity.height = c_h1
        c_entity.font = font

        c_entities.append(copy.deepcopy(c_entity))


    # get cluster entities arangment
    placed_entities = [c_entities[0]]
    min_x, max_x, min_y, max_y = c_entities[0].get_min_max()
    for current_entitiy in c_entities[1:]:
        new_center = get_cluster_arrangement(current_entitiy,placed_entities)
        current_entitiy.center = new_center
        placed_entities.append(current_entitiy)

        # get shape of cluster
        tmp_min_x, tmp_max_x, tmp_min_y, tmp_max_y = current_entitiy.get_min_max()

        if tmp_min_x < min_x:
            min_x = tmp_min_x
        if tmp_min_y < min_y:
            min_y = tmp_min_y
        if tmp_max_x > max_x:
            max_x = tmp_max_x
        if tmp_max_y > max_y:
            max_y = tmp_max_y

    # get dims of cluster
    w1 = int(abs(min_x) + abs(max_x))
    h1 = int(abs(min_y) + abs(max_y))
    cluster_shape = [w1,h1]

    # adjust center reference from first entity to top-left corner of cluster
    for idx in range(len(c_entities)):
        c_entities[idx].center[0] = int(c_entities[idx].center[0] + abs(min_x))
        c_entities[idx].center[1] = int(c_entities[idx].center[1] + abs(min_y))
    
    
    return placed_entities, cluster_shape

def set_relationship_config(self):
    
    # set tip_len based on base_len 
    # thick_rand = random.randint(1, 10)
    # if thick_rand < 5:
    #     self.thickness = 0
    # elif thick_rand < 8:
    #     self.thickness = 1
    # else:
    #     self.thickness = 2
    self.thickness = 0
    # self.thickness = random.randint(1, 3)
    self.base_len = random.randint(self.thickness+8, 12)
    self.tip_len = random.randint(self.base_len, 22)
    
    self.arrow_placement = random.choice([START, END])
    self.arrow_color = tuple([int(np.random.randint(0,200)),int(np.random.randint(0,200)),int(np.random.randint(0,200))])
    # self.indicator = random.choice([ACTIVATE, INDIRECT_ACTIVATE])
    self.indicator = random.choice([INHIBIT, ACTIVATE, INDIRECT_INHIBIT, INDIRECT_ACTIVATE])
    self.arch_ratio = 0.1
    self.spline_type = LINE

    return self

def set_text_config(self):

    self.font_folder = "font_folder"
    font_style_list = os.listdir(self.font_folder)
    self.font_style = random.choice(font_style_list)

    return self


def get_batch(self):

    self.template_dir = "templates"

    training_images = []
    training_labels = []

    for sample_idx in range(self.num_samples):

        # TODO:: do this random selection every placement
        self.padding = 20

        # loop through templates
        # read template and get query coords
        temp_x = random.randint(580, 810)
        temp_y = random.randint(400, 1000)
        template_im = np.ones((temp_y,temp_x,3)) * 255
        template_im = template_im.astype(np.uint8)
        # template_im = cv2.imread(os.path.join(self.directory, self.filename))


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

        num_to_place = random.randint(5, 20)
        for relation_idx in range(num_to_place):

            self = set_text_config(self)

            # num_entities = np.random.randint(1,4)
            num_entities = 1
            # text shape is actually cluster shape dimensions
            placed_entities1, cluster1_shape = get_entities(self,num_entities)
            w1 = cluster1_shape[0]
            h1 = cluster1_shape[1]

            # num_entities = np.random.randint(1,4)
            num_entities = 1
            placed_entities2, cluster2_shape = get_entities(self,num_entities)
            w2 = cluster2_shape[0]
            h2 = cluster2_shape[1]


            # get shape of target region
            # try to match g.t. relationship shape dist, but can't exactly since our generation needs z distance bewtween entities
            # scale = random.choice([40.0,80.0])
            scale = random.choice([40.0])
            dim_rand = np.random.randint(6)
            if dim_rand < 2:
                x_dim = int(np.abs(np.random.normal(loc=0.0, scale=scale))) + w1 + w2
                y_dim = int(np.abs(np.random.normal(loc=0.0, scale=scale))) + h1 + h2 + 50
            elif 2 <= dim_rand < 4:
                y_dim = int(np.abs(np.random.normal(loc=0.0, scale=scale))) + h1 + h2
                x_dim = int(np.abs(np.random.normal(loc=0.0, scale=scale))) + w1 + w2 + 50
            else:
                # more tall or long
                if np.random.randint(2):
                    x_dim = int(np.abs(np.random.normal(loc=0.0, scale=scale))) + max([w1,w2])
                    y_dim = int(np.abs(np.random.normal(loc=0.0, scale=scale))) + h1 + h2 + 50
                else:
                    y_dim = int(np.abs(np.random.normal(loc=0.0, scale=scale))) + max([h1,h2])
                    x_dim = int(np.abs(np.random.normal(loc=0.0, scale=scale))) + w1 + w2 + 50

            slice_shape = [x_dim,y_dim]

            # repeately check if queried coords are a valid location
            for idx in range(50):

                # subtracted max bounds of shape to ensure valid coords
                x_target = np.random.randint(0+self.padding,template_im.shape[1]-slice_shape[0]-self.padding)
                y_target = np.random.randint(0+self.padding,template_im.shape[0]-slice_shape[1]-self.padding)
                
                # check if selected template area is good
                if check_slice(template_im,slice_shape,x_target,y_target,self.padding):

                    # optionally show outline of queried region
                    # template_im = cv2.rectangle(template_im, (x_target-self.padding, y_target-self.padding), (x_target+x_dim+self.padding, y_target+y_dim+self.padding), (0,0,0), 1)

                    self = set_relationship_config(self)

                    # select indicator guided by training arguement, but with element of variability
                    indicator_prob = random.uniform(0, 1)

                    # indicator_prob = np.random.randint(0, 4)
                    if indicator_prob < self.classes[0]:
                        self.indicator = ACTIVATE
                    elif indicator_prob < (self.classes[0] + self.classes[1]):
                        self.indicator = INHIBIT
                    elif indicator_prob < (self.classes[0] + self.classes[1] + self.classes[2]):
                        self.indicator = INDIRECT_ACTIVATE
                    else:
                        self.indicator = INDIRECT_INHIBIT

                    cluster1_center,cluster2_center,entity_configuration = get_entity_placement(slice_shape,x_target,y_target,cluster1_shape,cluster2_shape,placed_entities1,placed_entities2)
                    
                    
                    template_im,relationship_bbox,placed_entities1,placed_entities2 = draw_relationship(self,template_im,cluster1_center,cluster2_center,entity_configuration,placed_entities1,placed_entities2,cluster1_shape,cluster2_shape)


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

                        cluster_shape = copy.deepcopy(base_shape)
                        min_x = cluster1_center[0] - (cluster1_shape[0]/2)
                        min_y = cluster1_center[1] - (cluster1_shape[1]/2)
                        max_x = cluster1_center[0] + (cluster1_shape[0]/2)
                        max_y = cluster1_center[1] + (cluster1_shape[1]/2)
                        cluster_shape['points'] = [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]
                        cluster_shape['ID'] = element_indx
                        cluster_shape['label'] = str(element_indx) + ":cluster:" + shapes_1_str
                        element_indx += 1
                        shapes.append(cluster_shape)




                    shapes_2_str = '['
                    if len(shapes_2) == 1:
                        shapes_2_str = str(shapes_2[0]['ID'])
                    else:
                        for shape in shapes_2:
                            shapes_2_str = shapes_2_str + str(shape['ID']) + ','
                        shapes_2_str = shapes_2_str[:-1] + ']'

                        cluster_shape = copy.deepcopy(base_shape)
                        min_x = cluster2_center[0] - (cluster2_shape[0]/2)
                        min_y = cluster2_center[1] - (cluster2_shape[1]/2)
                        max_x = cluster2_center[0] + (cluster2_shape[0]/2)
                        max_y = cluster2_center[1] + (cluster2_shape[1]/2)
                        cluster_shape['points'] = [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]
                        cluster_shape['ID'] = element_indx
                        cluster_shape['label'] = str(element_indx) + ":cluster:" + shapes_2_str
                        element_indx += 1
                        shapes.append(cluster_shape)


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


        # template_im = copy.deepcopy(template_im)
        # line_num = np.random.randint(0,20)
        line_num = 10
        shape = template_im.shape
        for line_idx in range(line_num):
            x1 = np.random.randint(0,shape[1])
            y1 = np.random.randint(0,shape[0])
            x2 = np.random.randint(0,shape[1])
            y2 = np.random.randint(0,shape[0])

            tmp_color = tuple([int(np.random.randint(0,200)),int(np.random.randint(0,200)),int(np.random.randint(0,200))])
        
            

            template_im = cv2.polylines(template_im, np.int32([np.array([[x1,y1],[x2,y2]])]), False, tmp_color, 1)


        # save json and new image
        im_dir = "output_test"
        image_path = str(sample_idx) + ".png"
        cv2.imwrite(os.path.join(im_dir, image_path), template_im)
        imageHeight = template_im.shape[0]
        imageWidth = template_im.shape[1]
        template_label_file = label_file.LabelFile()
        template_label_file.make_it(os.path.join(im_dir,str(sample_idx) + ".json"),shapes,image_path,imageHeight,imageWidth)

        training_images.append(template_im)
        training_labels.append(template_label_file)

    return training_images,training_labels

# write a function that loads the dataset into detectron2's standard format
def get_annotation_dicts(images, labels, category_list, anno_type = 'regular'):

    #go through all label files
    dataset_dicts = []

    for label_idx in range(len(images)):

        img = images[label_idx]
        imgs_anns = labels[label_idx]
        
        height, width = img.shape[:2]
        del img
        
        #declare a dict variant to save the content
        record = {}

        # record["file_name"] = filename
        # record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for anno in imgs_anns.shapes:
            #assert not anno["label"]
            #anno = anno["label"]
            poly_points = np.array(anno['points'],np.float32).reshape((-1 , 2))
            #rotated_rect = cv2.minAreaRect(poly_points)
            px = poly_points[:, 0]
            py = poly_points[:, 1]
            # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            # poly = list(itertools.chain.from_iterable(poly))
            try:
                component = list(anno['component'])
            except:
                component = []

            try:
                #only extract valid annotations
                category_id = imgs_anns.generate_category_id(anno,category_list)
            except Exception as e:
                #print(str.format('file: %s arises error: %s when generating category_id') % str(e))
                continue
            try:
                #LabelFile.normalize_shape_points(anno)
                if anno_type == 'regular':
                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "component": component,
                        "category_id": category_id,
                        "iscrowd": 0}
                if anno_type == 'rotated':
                    obj = {
                        "bbox": anno['rotated_box'],
                        "bbox_mode": BoxMode.XYWHA_ABS,
                        "component": component,
                        "category_id": category_id,
                        "iscrowd": 0}
                objs.append(obj)
            except Exception as e:
                print(str.format('file: arises error: %s when parsing box') % (str(e)))
                continue


        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

class my_args:
  def __init__(self, classes, num_samples):
    self.classes = classes
    self.num_samples = num_samples

if __name__ == "__main__":

    train_args = my_args({0:0.8,2:0.2},8)
    images,labels = get_batch(train_args)
    category_list = ['activate','gene','inhibit']
    # convert annotationformat
    reformatted_annotation = get_annotation_dicts(images, labels, category_list, anno_type = 'regular')

    train_instances = []
    for idx, record in enumerate(reformatted_annotation):
        bboxes = []
        class_ids = []
        objs = record["annotations"]
        for obj in objs:
            bboxes.append(obj["bbox"])
            class_ids.append(obj['category_id'])
        bboxes = torch.tensor(bboxes)
        bboxes = Boxes(bboxes)
        class_ids = torch.tensor(class_ids)
        image_shape = images[idx].shape
        train_instances.append(Instances((image_shape[0],image_shape[1]),gt_boxes=bboxes,gt_classes=class_ids))

    dataset = []
    for idx in range(len(images)):
        dataset.append({"image":torch.from_numpy(images[idx]).view(3,images[idx].shape[0],-1),'instances':train_instances[idx]})
    