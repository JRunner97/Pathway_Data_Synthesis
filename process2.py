import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import cv2
import copy
import math
 
# change thickness of line by adding 1 to knew set

X_ORIENTATION = 0
Y_ORIENTATION = 1

def test_it():

    # Dataset
    x_span = np.array([100, 200, 300, 400])
    y_span = np.array([100, 120, 120, 100])
    
    X_Y_Spline = make_interp_spline(x_span, y_span)
    
    # Returns evenly spaced numbers
    # over a specified interval.
    X_ = np.linspace(x_span.min(), x_span.max(), 500)
    Y_ = X_Y_Spline(X_)


    img = np.ones([500,500,3])
    img *= 255

    X_ = np.round(X_, 0).astype(int)
    Y_ = np.round(Y_, 0).astype(int)

    points = np.stack((X_,Y_)).T
    points = np.unique(points, axis=0)

    for x,y in points:
        img[y,x,0] = 0
        img[y,x,1] = 0
        img[y,x,2] = 0

    img = img / 255

    # cv2.imwrite('color_img.jpg', img)
    cv2.imshow("image", img)
    cv2.waitKey(0)


    print(x_span)
    tri_source = [x_span[3],y_span[3]]


    pt1 = (tri_source[0], tri_source[1]+5)
    pt2 = (tri_source[0]+10, tri_source[1])
    pt3 = (tri_source[0], tri_source[1]-5)

    triangle_cnt = np.array( [pt1, pt2, pt3] )

    cv2.drawContours(img, [triangle_cnt], 0, (0,0,0), -1)

    cv2.imshow("image", img)
    cv2.waitKey(0)



def draw_spline(img,x_span,y_span,thickness,orientation):

    
    X_Y_Spline = make_interp_spline(x_span, y_span)
    
    # Returns evenly spaced numbers
    # over a specified interval.
    X_ = np.linspace(x_span.min(), x_span.max(), 500)
    Y_ = X_Y_Spline(X_)
    
    X_ = np.round(X_, 0).astype(int)
    Y_ = np.round(Y_, 0).astype(int)

    base_points = np.stack((X_,Y_)).T
    base_points = np.unique(base_points, axis=0)

    base_points_shape = base_points.shape
    print(base_points_shape)

    if orientation is X_ORIENTATION:
        for x,y in base_points:
            img = cv2.circle(img, (x,y), thickness, (0,0,0), -1)
    else:
        for x,y in base_points:
            img = cv2.circle(img, (y,x), thickness, (0,0,0), -1)

    lag = 10

    base_points_y = base_points[:,1]
    y_max_idx = base_points_y.shape[0] - 1
    end_slope = (base_points_y[y_max_idx] - base_points_y[y_max_idx-lag])*-1

    print("end point")
    print(base_points_y[y_max_idx])

    print("lag point")
    print(base_points_y[y_max_idx-lag])

    return img, end_slope


def draw_arrowhead(img,x_span,y_span,tip_slope):

    lag = 10


    # TODO:: small slope makes 10 lag used above to get slope estimate drift
    # i.e. lengths for base and tip are scaled less when slope is large 
    # can adjust slope estimate to fix or set slope threshold for changing angle (prob need to change slope estimate to work for different angles)
    # slope == 1 => deg == 45

    tri_source = [x_span[3],y_span[3]]

    # img = cv2.circle(img, tuple(tri_source), 3, (0,255,0), -1)

    arrowhead_base_slpe = -1/tip_slope

    # returned in radians
    tip_deg = math.atan(tip_slope)
    base_deg = math.atan(arrowhead_base_slpe)

    print("slope")
    print(tip_slope)
    print("deg")
    print(tip_deg)

    # print(arrowhead_base_slpe)
    # print(base_deg)

    tip_len = 5

    # not really lag # pixels, since use unique op
    tip_rise = tip_len * math.sin(tip_deg)
    tip_run = (lag * tip_rise) / tip_slope
    tip_rise = math.floor(tip_rise)
    tip_run = math.floor(tip_run)


    print("tip rise run")
    print(tip_rise)
    print(tip_run)

    
    base_len = 15

    base_rise = base_len * math.sin(base_deg)
    base_run = (lag * base_rise) / tip_slope
    base_rise = math.floor(base_rise)
    base_run = math.floor(base_run)

    # print(base_rise)
    # print(base_run)


    pt1 = (tri_source[0]-base_rise, tri_source[1]-base_run)
    pt2 = (tri_source[0]+tip_run, tri_source[1]-tip_rise)
    pt3 = (tri_source[0]+base_rise, tri_source[1]+base_run)

    triangle_cnt = np.array( [pt1, pt2, pt3] )

    cv2.drawContours(img, [triangle_cnt], 0, (0,0,0), -1)

    # img = cv2.circle(img, pt2, 3, (0,0,255), -1)
    # img = cv2.circle(img, pt1, 3, (255,0,0), -1)
    # img = cv2.circle(img, pt3, 3, (255,0,0), -1)


    return img


if __name__ == "__main__":

    img = np.ones([500,500,3])
    img *= 255


    # Dataset
    x_span = np.array([100, 200, 300, 400])
    y_span = np.array([100, 120, 120, 100])

    thickness = 1
    img, end_slope = draw_spline(img,x_span,y_span,thickness,X_ORIENTATION)
    img = draw_arrowhead(img,x_span,y_span,end_slope)


    img = img / 255
    cv2.imshow("image", img, )
    cv2.waitKey(0)