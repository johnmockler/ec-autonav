import cv2
import numpy as np
from matplotlib import pyplot as plt
import math as m
MAP = '/home/john/Pictures/simpleTest.png'
#MAP = '/home/john/Pictures/ENC_test.png'
TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])

MIN_LENGTH = 15.00
dtype = [('x',int), ('y',int)]
arrows = []

#process map
map_image = cv2.imread(MAP,1)
map_threshed = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)
corners = cv2.goodFeaturesToTrack(map_mask,500,0.5,10)
corners = np.int0(corners)
np.sort(corners,axis=1)

class ArrowPair():

    def __init__(self, x, y, pair=None, dist=1000):
        self.x = x
        self.y = y
        self.pair = pair
        self.dist = dist
    
    def draw(self):
        cv2.line(map_image,(self.x,self.y),(self.pair.x,self.pair.y),(0,255,0),thickness=3, lineType=8)

