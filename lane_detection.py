import cv2
import numpy as np
import edge_detection as edge
import matplotlib.pyplot as plt

class Lane:
    def _init_(self,orig_frame):
        self.orig_frame=orig_frame
        self.lane_line_markings=None
        self.warped_frame=None
        self.transformation_matrix=None
        self.inv_transformation_matrix=None

        self.orig_image_size=self.orig_frame.shape[::-1][1:]
        width=self.orig_image_size[0]
        height=self.orig_image_size[1]
        self.width=width
        self.height=height


        self.roi_points=np.float32([
            (274,184),
            (0,337),
            (575,337),
            (371,184)
        ])

        self.padding=int(0.25*width)
        self.desired_roi_points=np.float32([
            [self.padding,0],
            [self.padding,self.orig_image_size[1]],
            [self.orig_image_size[0]-self.padding, self.orig_image_size[1]],
            [self.orig_image_size[0]-self.padding,0]
        ])




