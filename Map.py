import cv2
import numpy as np
from matplotlib import pyplot as plt

IMAGEPATH = '/home/john/Pictures/testing_image.png'

LAND_MIN = np.array([26, 0, 20])
LAND_MAX = np.array([28, 100, 255])

SHALLOW_BOUND_MIN = np.array([20, 100, 100])
SHALLOW_BOUND_MAX = np.array([22, 255, 255])

HI_DEPTH_MIN = np.array([0, 0, 250])
HI_DEPTH_MAX = np.array([5, 255, 255])

MED_DEPTH_MIN = np.array([89, 0, 100])
MED_DEPTH_MAX = np.array([92, 100, 255])

WATER_BOUND_MIN = np.array([101, 100, 100])
WATER_BOUND_MAX = np.array([103, 255, 255])

LO_DEPTH_MIN = np.array([93, 0, 100])
LO_DEPTH_MAX = np.array([95, 150, 255])

SHALLOW_DEPTH_MIN = np.array([56, 0, 100])
SHALLOW_DEPTH_MAX = np.array([58, 150, 255])

TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 255, 255])


class MyMap:

    def __init__(self):
        self.image = cv2.imread(IMAGEPATH,1)
        dim = self.image.shape
        print(dim)
        self.map = np.zeros(dim[0], dim[1])
        self.hsv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.kernel = np.ones((5,5),np.uint8)


    def loadMap(self):
        # FIND CONTOURS
        landMask = cv2.inRange(self.hsv_img, LAND_MIN, LAND_MAX)
        self.land_contour, self.land_hierarchy = cv2.findContours(landMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shallowMask = cv2.inRange(hsv_img, SHALLOW_DEPTH_MIN, SHALLOW_DEPTH_MAX)
        shallowBorderMask = cv2.inRange(hsv_img, SHALLOW_BOUND_MIN, SHALLOW_BOUND_MAX)
        shallowMaskAll = cv2.bitwise_or(shallowMask, shallowBorderMask)
        self.shallow_contour, self.shallow_hierarchy = cv2.findContours(shallowMaskAll, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lowMask = cv2.inRange(self.hsv_img, LO_DEPTH_MIN, LO_DEPTH_MAX)
        self.low_contour, self.low_hierarchy = cv2.findContours(lowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        medMask = cv2.inRange(self.hsv_img, MED_DEPTH_MIN, MED_DEPTH_MAX)
        medBorderMask = cv2.inRange(self.hsv_img, WATER_BOUND_MIN, WATER_BOUND_MAX)
        medMaskAll = cv2.bitwise_or(medMask, medBorderMask)
        medMaskNoGaps = cv2.morphologyEx(medMaskAll, cv2.MORPH_CLOSE, self.kernel)
        self.med_contour, self.med_hierarchy = cv2.findContours(medMaskNoGaps, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        highMask = cv2.inRange(self.hsv_img, HI_DEPTH_MIN, HI_DEPTH_MAX)
        self.high_contour, self.high_hierarchy = cv2.findContours(highMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        trafficMask = cv2.inRange(self.hsv_img, TRAFFIC_MIN1, TRAFFIC_MAX1)
        self.traffic_contour, self.traffic_hierarchy = cv2.findContours(trafficMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        or1 = cv2.bitwise_or(shallowMaskAll, lowMask)
        or2 = cv2.bitwise_or(or1, medMaskAll)
        or3 = cv2.bitwise_or(or2, highMask)
        or4 = cv2.bitwise_or(or3, trafficMask)
        or5 = cv2.bitwise_or(or4, landMask)

        # filter out buoys and random lines
        yellowMask = cv2.inRange(self.hsv_img, YELLOW_BUOY_MIN, YELLOW_BUOY_MAX)
        greenMask = cv2.inRange(self.hsv_img, GREEN_BUOY_MIN, GREEN_BUOY_MAX)
        silverMask = cv2.inRange(self.hsv_img, SILVER_LINE_MIN, SILVER_LINE_MAX)
        redMask = cv2.inRange(self.hsv_img, RED_LINE_MIN, RED_LINE_MAX)

        or6 = cv2.bitwise_or(or5, yellowMask)
        or7 = cv2.bitwise_or(or6, greenMask)
        or8 = cv2.bitwise_or(or7, silverMask)
        or9 = cv2.bitwise_or(or8, redMask)

        obstacleMask = cv2.bitwise_not(or9)
        self.obstacle_contour, self.obstacle_hierarchy = cv2.findContours(obstacleMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    def isObstacle(self, x, y):
        land = cv2.pointPolygonTest(self.land_contour,(x,y),False)
        shallow = cv2.pointPolygonTest(self.shallow_contour,(x,y),False)
        med = cv2.pointPolygonTest(self.med_contour,(x,y),False)
        obstacle = cv2.pointPolygonTest(self.obstacle_contour,(x,y),False)
        return land or shallow or med or obstacle