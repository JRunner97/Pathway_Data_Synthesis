class Synthetic_Shape:
    def __init__(self, center, label):
        self.center = center
        self.label = label
        self.font_size = random.randint(8, 20)
        self.text_margin = random.randint(5, 15)
        self.text_thickness = random.randint(1, 2)
        self.textbox_background = (0,0,230)
        self.textbox_border_thickness = random.randint(0, 2)

        shape_image = random.choice(os.listdir("shape_images"))
        image = cv2.imread(os.path.join("shape_images",shape_image))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.cnt = cnts[0][1]
        self.source_image_dims = image.shape

    def draw_shape(self,img,center):

        # get zero centered points
        cols, rows = self.get_points()

        # send to new center
        rows = rows + center[1]
        cols = cols + center[0]

        pts = np.stack((cols,rows),axis=-1).astype(np.int32)

        img = cv2.fillPoly(img, [pts], self.textbox_background)
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
