import cv2
import numpy as np
from matplotlib import pyplot as plt
import Arrow

IMAGEPATH = '/home/john/Pictures/simpleTest.png'
ARROW1 = '/home/john/Pictures/arrow_example.png'
ARROW2 = '/home/john/Pictures/broken_arrow.png'

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

#for light purple
TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])

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
        self.map = np.zeros((dim[0], dim[1]))
        self.hsv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.kernel = np.ones((5, 5), np.uint8)
        self.land_contour = 0
        self.shallow_contour = 0
        self.obstacle_contour = 0
        self.med_contour = 0
        self.high_contour = 0
        self.traffic_contour = 0
        self.traffic_mask = 0

    #update this function to only search a given boundary?
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

        self.trafficMask = cv2.inRange(self.hsv_img, TRAFFIC_MIN1, TRAFFIC_MAX1)
        trafficBlur = cv2.GaussianBlur(trafficMask,(3,3),0)
        self.traffic_contour, traffic_hierarchy = cv2.findContours(trafficBlur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

    #update to more efficiently search for obstacles
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
    
    #find all arrows in an image
    #need to update to find large arrows
    def findArrows(self):
        arrow1, arrow1_area = self.loadModelImage(ARROW1, TRAFFIC_MIN1, TRAFFIC_MAX1,0)
        arrow2, arrow2_area = self.loadModelImage(ARROW2, TRAFFIC_MIN1, TRAFFIC_MAX1,1)
        arrows = []

        for contour in self.traffic_contour:
                if ((cv2.contourArea(contour) <= 1.5*arrow1_area and cv2.contourArea(contour) >= 0.8*arrow1_area and cv2.matchShapes(contour,arrow1[0], 3,0.0) < 1) 
                or (cv2.contourArea(contour) <= 1.7*arrow2_area and cv2.contourArea(contour) >= 0.2*arrow2_area  and cv2.matchShapes(contour, arrow2[0],1,0.0) < 5)):
                    
                    moment = cv2.moments(contour)
                    cx = int(moment['m10']/moment['m00'])
                    cy = int(moment['m01']/moment['m00'])
                    (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
                    arrow = Arrow.Arrow(cx, cy, angle)

                    #for test 
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(self.image,[box],0,(0,0,255),2)
                    #[X, Y, W, H] = cv2.boundingRect(contour)
                    #cv2.rectangle(self.image, (X,Y), (X+W, Y+H), (255, 0, 0), 2)
                    self.image[cy, cx] = [255, 0, 0]




                    print(angle)
                    arrows.append(arrow)
        plt.imshow(self.image,cmap='gray')
        plt.show()
        return arrows
    
    #backup to find arrows using good features to track algorithm
    def findArrows2(self):
        #with corners, next step is to verify whether area around corners denotes an arrow
        corners = cv2.goodFeaturesToTrack(self.traffic_mask,25,0.01,10)
        corners = np.int0(corners)


    #find arrow direction
    def arrowDirection(self):
        return -1

    #finds traffic buffer zones
    def findTrafficZones(self):
        return -1

    #finds traffic boundaries
    def findTrafficBounds(self):
        return -1


    def loadModelImage(self, impath,min_color_range, max_color_range, mode):
        try:
            image = cv2.imread(impath,1)
            hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            masked_image = cv2.inRange(hsv_image, min_color_range, max_color_range)

            if mode == 1:
                canny_image = cv2.Canny(masked_image,0,150)
                blurred_image = cv2.blur(canny_image,(2,2))
                ret, threshed_image = cv2.threshold()
                contour, heirarchy = cv2.findContours(threshed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            else:
                blurred_image = cv2.blur(masked_image, (2,2))
                contour, heirarchy = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            contour_area = cv2.contourArea(contour[0])

            return contour, contour_area   
        
        except:
            return [-1], -1

    def printPath(self, path):
        try:
            for coord in path:
                self.image[coord[0], coord[1]] = [255, 0, 0]

            plt.imshow(self.image, cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            plt.show()
        except:
            print('path not found!')