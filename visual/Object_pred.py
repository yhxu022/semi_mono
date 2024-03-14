
import numpy as np

class Object3d_pred(object):
    def __init__(self, det):
        # data = label_file_line.split(" ")
        data = det
        data[1:] = [float(x) for x in data[1:]]
        # extract label, truncation, occlusion
        self.type = ['Pedestrian', 'Car', 'Cyclist'][int(data[0])]  # 'Car', 'Pedestrian', ...
        self.alpha = data[1]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[2]  # left
        self.ymin = data[3]  # top
        self.xmax = data[4]  # right
        self.ymax = data[5]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[6]  # box height
        self.w = data[7]  # box width
        self.l = data[8]  # box length (in meters)
        self.t = (data[9], data[10], data[11])  # location (x,y,z) in camera coord.
        self.ry = data[12]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]