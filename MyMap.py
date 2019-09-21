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

SILVER_LINE_MIN = np.array([0, 0, 150])
SILVER_LINE_MAX = np.array([0, 0, 200])

RED_LINE_MIN = np.array([0, 250, 175])
RED_LINE_MAX = np.array([0, 255, 255])

GREEN_BUOY_MIN = np.array([59, 200, 200])
GREEN_BUOY_MAX = np.array([61, 255, 255])

YELLOW_BUOY_MIN = np.array([25, 250, 250])
YELLOW_BUOY_MAX = np.array([27, 255, 255])


class MyMap:

    def __init__(self):
        self.image = cv2.imread(IMAGEPATH, 1)
        dim = self.image.shape
        print(dim)
        self.map = np.zeros((dim[0], dim[1]))
        self.hsv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.kernel = np.ones((5, 5), np.uint8)
        self.land_contour = 0
        self.shallow_contour = 0
        self.obstacle_contour = 0
        self.med_contour = 0
        self.high_contour = 0
        self.traffic_contour = 0


    def loadMap(self):
        # FIND CONTOURS
        landMask = cv2.inRange(self.hsv_img, LAND_MIN, LAND_MAX)
        self.land_contour, land_hierarchy = cv2.findContours(landMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shallowMask = cv2.inRange(self.hsv_img, SHALLOW_DEPTH_MIN, SHALLOW_DEPTH_MAX)
        shallowBorderMask = cv2.inRange(self.hsv_img, SHALLOW_BOUND_MIN, SHALLOW_BOUND_MAX)
        shallowMaskAll = cv2.bitwise_or(shallowMask, shallowBorderMask)
        self.shallow_contour, shallow_hierarchy = cv2.findContours(shallowMaskAll, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lowMask = cv2.inRange(self.hsv_img, LO_DEPTH_MIN, LO_DEPTH_MAX)
        self.low_contour, low_hierarchy = cv2.findContours(lowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        medMask = cv2.inRange(self.hsv_img, MED_DEPTH_MIN, MED_DEPTH_MAX)
        medBorderMask = cv2.inRange(self.hsv_img, WATER_BOUND_MIN, WATER_BOUND_MAX)
        medMaskAll = cv2.bitwise_or(medMask, medBorderMask)
        medMaskNoGaps = cv2.morphologyEx(medMaskAll, cv2.MORPH_CLOSE, self.kernel)
        self.med_contour, med_hierarchy = cv2.findContours(medMaskNoGaps, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        highMask = cv2.inRange(self.hsv_img, HI_DEPTH_MIN, HI_DEPTH_MAX)
        self.high_contour, high_hierarchy = cv2.findContours(highMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        trafficMask = cv2.inRange(self.hsv_img, TRAFFIC_MIN1, TRAFFIC_MAX1)
        self.traffic_contour, traffic_hierarchy = cv2.findContours(trafficMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

    def returnMap(self):
        return self.map

    def isObstacle(self, y, x):


        for cnt in self.land_contour:
            land = cv2.pointPolygonTest(cnt,(x,y),False)
            if land > 0:
                return 1

        for cnt in self.shallow_contour:
            shallow = cv2.pointPolygonTest(cnt,(x,y),False)
            if shallow > 0:
                return 1

        for cnt in self.low_contour:
            low = cv2.pointPolygonTest(cnt, (x, y), False)
            if low > 0:
                return 1

        for cnt in self.med_contour:
            med = cv2.pointPolygonTest(cnt,(x,y),False)
            if med > 0:
                return 1

        for cnt in self.obstacle_contour:
            obstacle = cv2.pointPolygonTest(cnt,(x,y),False)
            if obstacle > 0:
                return 1

        return 0

    def printPath(self, path):
        try:
            for coord in path:
                self.image[coord[0], coord[1]] = [255, 0, 0]

            plt.imshow(self.image, cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            plt.show()
        except:
            print('path not found!')