import numpy as np
import os
import cv2
import threading
import math
import string
import random
from PIL import ImageFont, ImageDraw, Image
import json

ELLIPSE = 0
RECT = 1
NO_SHAPE = 2

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))


def draw_text(self,c_entity,slice_shape):

    """

        draws text in shape from specified font
            
    """

    w1 = slice_shape[0]
    h1 = slice_shape[1]

    img = np.ones((h1,w1,3), np.int32)
    self.slice_background = (np.random.randint(200,255),np.random.randint(200,255),np.random.randint(200,255))
    img *= self.slice_background
    img = img.astype(np.uint8)

    self.textbox_background = (np.random.randint(200,255),np.random.randint(200,255),np.random.randint(200,255))
    self.text_color = (255 - self.textbox_background[0], 255 - self.textbox_background[1], 255 - self.textbox_background[2],0)

    current_center = [int(w1/2),int(h1/2)]

    label = c_entity['label']
    w = c_entity['width']
    h = c_entity['height']

    # location is center
    # x1,y1 is top left corner
    x1 = current_center[0] - int(round(w/2)) - self.text_margin
    y1 = current_center[1] - int(round(h/2)) - self.text_margin

    if c_entity['type'] == ELLIPSE:
        img = cv2.ellipse(img, tuple(current_center), (math.floor(w*.80),h), 0, 0, 360, self.textbox_background, -1)
        if np.random.randint(2):
            border_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            img = cv2.ellipse(img, tuple(current_center), (math.floor(w*.80),h), 0, 0, 360, border_color, self.textbox_border_thickness)
    elif c_entity['type'] == RECT:
        img = cv2.rectangle(img, (x1, y1), (x1 + w + (self.text_margin*2), y1 + h + (self.text_margin*2)), self.textbox_background, -1)
        if np.random.randint(2):
            border_color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            img = cv2.rectangle(img, (x1, y1), (x1 + w + (self.text_margin*2), y1 + h + (self.text_margin*2)), border_color, self.textbox_border_thickness)

    b,g,r,a = 0,255,0,0
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x1 + self.text_margin, y1 + h + self.text_margin), label, font=c_entity['font'], fill=self.text_color, anchor='lb')
    img = np.array(img_pil)

    return img

def set_text_config(self):

    # save .tff to this folder
    self.font_folder = "font_folder"
    font_style_list = os.listdir(self.font_folder)
    self.font_style = random.choice(font_style_list)
    self.font_size = random.randint(8, 20)
    # TODO:: may need to change this dependent on text area size
    self.text_margin = random.randint(1, 5)
    self.textbox_border_thickness = random.randint(0, 2)

    return self

class ocr_child_thread(threading.Thread):
    def __init__(self,copyID,name):
        threading.Thread.__init__(self)
        self.copyID = copyID
        self.name = name
        f = open('exHUGO_latest.json')
        self.gene_dict = json.load(f)

    def run(self):

        """

        create text slice/window and save label to .txt file

        """

        self = set_text_config(self)

        # get random word and randomly capitalize
        # cluster_str_len = np.random.randint(1,20)
        # cluster_label = randomword(cluster_str_len)
        cluster_label = random.choice(self.gene_dict)
        cluster_label = ''.join(random.choice((str.upper, str.lower))(c) for c in cluster_label)

        # load font and get text size
        fontpath = os.path.join(self.font_folder, self.font_style)    
        font = ImageFont.truetype(fontpath, self.font_size)
        (c_w1, c_h1) = font.getsize(cluster_label)

        # initialize text
        c_entity = {}
        c_entity['label'] = cluster_label
        c_entity['font'] = font
        c_entity['width'] = c_w1
        c_entity['height'] = c_h1
        c_entity['center'] = [0,0]
        c_entity['type'] = random.choice([RECT, ELLIPSE, NO_SHAPE])

        # scale text size to get slice size
        x_dim = np.random.randint(5,20) + c_entity['width'] + self.text_margin
        y_dim = np.random.randint(5,20) + c_entity['height'] + self.text_margin
        slice_shape = [x_dim,y_dim]

        img = draw_text(self,c_entity,slice_shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

        # save txt and new image
        im_dir = "output_test"
        image_path = str(self.copyID) + ".png"
        cv2.imwrite(os.path.join(im_dir, image_path), thresh)
        file1 = open(os.path.join(im_dir, str(self.copyID) + ".txt"),"w")
        file1.write(c_entity['label'])
        file1.close() 



class ocr_parent_thread(threading.Thread):
    def __init__(self, threadID,name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):

        """

        Start 4 threads for generating x# samples from each parent

        """
        
        # how many images per template
        stop_child_flag = False
        num_copies = 10
        for child_idx in range(0,num_copies):

            # children in here
            child_thread_idx = child_idx * 4
            # other parent threads
            child_thread_idx = self.threadID*num_copies + child_thread_idx

            if stop_child_flag:
                break

            child_thread0 = ocr_child_thread(child_thread_idx,"child0")
            child_thread1 = ocr_child_thread(child_thread_idx+1,"child1")
            child_thread2 = ocr_child_thread(child_thread_idx+2,"child2")
            child_thread3 = ocr_child_thread(child_thread_idx+3,"child3")

            child_thread0.start()
            if (child_idx*4) + 1 > num_copies:
                stop_child_flag = True
                continue
            else:
                child_thread1.start()
            if (child_idx*4) + 2 > num_copies:
                stop_child_flag = True
                continue
            else:
                child_thread2.start()
            if (child_idx*4) + 3 > num_copies:
                stop_child_flag = True
                continue
            else:
                child_thread3.start()

            child_thread0.join()
            child_thread1.join()
            child_thread2.join()
            child_thread3.join()

def generate_ocr_text():

    """

    Start multiple threads for generating samples

    """

    # loop through all templates
    num_parents = 20
    stop_flag = False

    for parent_idx in range(0,num_parents-1,4):

        if stop_flag:
            break

        thread0 = ocr_parent_thread(parent_idx,"thread-0")
        thread1 = ocr_parent_thread(parent_idx+1,"thread-1")
        thread2 = ocr_parent_thread(parent_idx+2,"thread-2")
        thread3 = ocr_parent_thread(parent_idx+3,"thread-3")

        thread0.start()
        if parent_idx + 1 > num_parents:
            stop_flag = True
            continue
        else:
            thread1.start()
        if parent_idx + 2 > num_parents:
            stop_flag = True
            continue
        else:
            thread2.start()
        if parent_idx + 3 > num_parents:
            stop_flag = True
            continue
        else:
            thread3.start()

        thread0.join()
        thread1.join()
        thread2.join()
        thread3.join()

if __name__ == "__main__":

    generate_ocr_text()
