import cv2
import numpy as np
from matplotlib import pyplot as plt
import PointHandler as pt 
import MapObject as obj 
from include import kdtree
import math as m

#constants
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
TRAFFIC_MIN_ALL = np.array([150, 10, 10])
TRAFFIC_MAX_ALL = np.array([151, 255, 255])

TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])

TRAFFIC_MIN2 = np.array([150, 200, 10])
TRAFFIC_MAX2 = np.array([151, 255, 255])

SILVER_LINE_MIN = np.array([0, 0, 150])
SILVER_LINE_MAX = np.array([0, 0, 200])

RED_LINE_MIN = np.array([0, 250, 175])
RED_LINE_MAX = np.array([0, 255, 255])

GREEN_BUOY_MIN = np.array([59, 200, 200])
GREEN_BUOY_MAX = np.array([61, 255, 255])

YELLOW_BUOY_MIN = np.array([25, 250, 250])
YELLOW_BUOY_MAX = np.array([27, 255, 255])

class MyMap():

    def __init__(self, impath=r'C:\Users\jmock\Documents\Projekt Arbeit Images\opencvtest.png', offset=(0, 0)):
        self.impath = impath
        self.offset = offset
        self.image = cv2.imread(impath, 1)
        #self.obstacle_image = np.zeros((self.image.shape[0],self.image.shape[1],3), np.uint8)
        self.arrows = []

        self.load_map()



    def load_map(self):


        hsv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        land_mask = cv2.inRange(hsv_img, LAND_MIN, LAND_MAX)
        land_contour, land_hierarchy = cv2.findContours(land_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shallow_mask = cv2.inRange(hsv_img, SHALLOW_DEPTH_MIN, SHALLOW_DEPTH_MAX)
        shallow_border_mask = cv2.inRange(hsv_img, SHALLOW_BOUND_MIN, SHALLOW_BOUND_MAX)
        shallow_mask_all = cv2.bitwise_or(shallow_mask, shallow_border_mask)

        low_mask = cv2.inRange(hsv_img, LO_DEPTH_MIN, LO_DEPTH_MAX)
        low_contour, low_hierarchy = cv2.findContours(low_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        med_kernel = np.ones((5, 5), np.uint8)
        med_mask = cv2.inRange(hsv_img, MED_DEPTH_MIN, MED_DEPTH_MAX)
        med_border_mask = cv2.inRange(hsv_img, WATER_BOUND_MIN, WATER_BOUND_MAX)
        med_mask_all = cv2.bitwise_or(med_mask, med_border_mask)
        med_mask_no_gaps = cv2.morphologyEx(med_mask_all, cv2.MORPH_CLOSE, med_kernel)
        med_contour, med_hierarchy = cv2.findContours(med_mask_no_gaps, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        high_mask = cv2.inRange(hsv_img, HI_DEPTH_MIN, HI_DEPTH_MAX)
        high_contour, high_hierarchy = cv2.findContours(high_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        or1 = cv2.bitwise_or(shallow_mask_all, low_mask)
        or2 = cv2.bitwise_or(or1, med_mask_all)
        or3 = cv2.bitwise_or(or2, high_mask)
        or4 = cv2.bitwise_or(or3, land_mask)

        # filter out buoys and random lines
        yellow_mask = cv2.inRange(hsv_img, YELLOW_BUOY_MIN, YELLOW_BUOY_MAX)
        green_mask = cv2.inRange(hsv_img, GREEN_BUOY_MIN, GREEN_BUOY_MAX)
        silver_mask = cv2.inRange(hsv_img, SILVER_LINE_MIN, SILVER_LINE_MAX)
        red_mask = cv2.inRange(hsv_img, RED_LINE_MIN, RED_LINE_MAX)
        pink_mask = cv2.inRange(hsv_img, TRAFFIC_MIN_ALL, TRAFFIC_MAX_ALL)


        or5 = cv2.bitwise_or(or4, pink_mask)
        or6 = cv2.bitwise_or(or5, yellow_mask)
        or7 = cv2.bitwise_or(or6, green_mask)
        or8 = cv2.bitwise_or(or7, silver_mask)
        or9 = cv2.bitwise_or(or8, red_mask)

        obstacle_mask = cv2.bitwise_not(or9)
        
        #sailable area is area that excludes: land, obstacles, shallow water
        self.sailable_area = cv2.bitwise_or(obstacle_mask, land_mask)
        self.sailable_area = cv2.bitwise_or(self.sailable_area, shallow_mask_all)
        #save obstacle locations on obstacle image
        # for obst in list(self.obstacles.inorder()):
        #     cv2.circle(self.image,(obst.data.coords.x,obst.data.coords.y),2,(0,0,255),-1)
        #     cv2.circle(self.obstacle_image,(obst.data.coords.x,obst.data.coords.y),int(obst.data.radius),(0,255,0),-1)
        # cv2.drawContours(obstacle_image, land_contour, -1, (0,255,0), -1)
        # cv2.drawContours(obstacle_image, shallow_contour, -1, (0,255,0), -1)    

        traffic_mask_dark = cv2.inRange(hsv_img, TRAFFIC_MIN2, TRAFFIC_MAX2)
        traffic_mask_light = cv2.inRange(hsv_img, TRAFFIC_MIN1, TRAFFIC_MAX1)


        line_set = self.detect_lines(traffic_mask_dark)
        self.arrows, arrow_tree = self.detect_arrows(traffic_mask_light)
        self.process_arrows(self.arrows, line_set)

        return 0

    def detect_arrows(self, mask):
        CORNER_THRESHOLD = 0.6
        arrows = []
        corners = cv2.goodFeaturesToTrack(mask,500,CORNER_THRESHOLD,10)
        corners = np.int0(corners)

        for i in corners:
            x,y = i.ravel()
            coords = pt.Point(x,y)
            arrows.append(obj.Arrow(coords))
    
        arrows = self.filter_arrows(arrows)
        arrow_tree = kdtree.create(arrows)
        arrows = list(arrow_tree.inorder())

        return arrows, arrow_tree

        
    def detect_lines(self, mask):
        '''finds dark purple lines in image

        '''
        lineSet = []
        
        kernel = np.ones((4,4),np.uint8)

        #circle finder stuff
        #https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
        #http://www.bmva.org/bmvc/1989/avc-89-029.pdf
        DP = 1
        MIN_DIST = 10
        MAX_RADIUS = 40
        ACCUMULATOR_THRESH = 15 #lower = more circles found

        
        xformed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        #grayimg = cv2.cvtColor(map_mask, cv2.COLOR_BGR2GRAY)
        dilation = cv2.dilate(mask,kernel,iterations = 2)
        erosion = cv2.erode(dilation,kernel,iterations = 1)

        #BLOCK OUT CIRCLES FIRST:
        circles = cv2.HoughCircles(erosion,cv2.HOUGH_GRADIENT,DP,MIN_DIST, param2=ACCUMULATOR_THRESH,maxRadius=MAX_RADIUS)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(erosion,(i[0],i[1]),i[2]+6,0,-1)
        else:
            print('no circles found!')

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        RHO_RES = 1
        THETA_RES = np.pi/180
        THRESHOLD = 25 #lower = more lines default = 25
        MINLENGTH = 10


        lines = cv2.HoughLinesP(erosion,RHO_RES,THETA_RES,THRESHOLD, minLineLength=MINLENGTH)
        for line in lines:
            for x1, y1, x2, y2 in line:
                pt1 = pt.Point(x1,y1)
                pt2 = pt.Point(x2,y2)
                coords = pt.midpoint(pt1, pt2)
                direct = pt.direction(pt1, pt2)
                segment = (pt1, pt2)
                newLine = obj.Line(coords, direct, segment)
                lineSet.append(newLine)
        lineSet = kdtree.create(lineSet)
        
        return lineSet
    
    
    def process_arrows(self, arrows, lineSet):

        for arrow in arrows:
            nearbyLines = lineSet.search_knn((arrow.data.coords.x,arrow.data.coords.y), 6)
            closest_dist = 1000
            for neighbor in nearbyLines:
                neighbor_dist = abs(pt.perpendicular_dist(neighbor[0].data.segment[0], neighbor[0].data.segment[1], arrow.data.coords))

                if m.isnan(neighbor_dist):
                    neighbor_dist = abs(pt.distance(arrow.data.coords,neighbor[0].data.coords))

                if neighbor_dist <= closest_dist:
                    closest_line = neighbor 
                    closest_dist = neighbor_dist

            #CALCULATE DIRECTION
            pt_closest= pt.closest_point(closest_line[0].data.segment[0], closest_line[0].data.segment[1], arrow.data.coords)
            arrow_dir = pt.direction(arrow.data.coords,pt_closest) - m.pi/2.0

            arrow_dir = pt.scale_angle(arrow_dir)

            arrow.data.direction = arrow_dir

    def process_obstacles(self, contour_array):
        obstacle_list = []
        for cnt in contour_array:
            #find centroid
            try:
                m = cv2.moments(cnt)
                cx = int(m['m10']/m['m00'])
                cy = int(m['m01']/m['m00'])
                coords = pt.Point(cx,cy)

                #find radius
                area = cv2.contourArea(cnt)
                radius = np.sqrt(2*area/np.pi)
                obstacle_list.append(obj.Obstacle(coords, radius))
            except:

                next



        return kdtree.create(obstacle_list)
    
    def filter_arrows(self, arrows):
        '''takes a list of points, finds arrows within the Minimum distance and replaces them with their midpoint
        Parameters
        ----------
        arrows: list of arrow objects

        Returns
        -------
        arrows: list list of arrow objects
        '''

        MIN_DISTANCE = 17.0

        for arrow in arrows:
            for next_arrow in arrows:
                dist = pt.distance(arrow.coords, next_arrow.coords)
                if dist != 0 and dist < MIN_DISTANCE:
                    arrow.coords = pt.midpoint(arrow.coords,next_arrow.coords)
                    arrows.remove(next_arrow)

        return arrows
    
    def is_obstacle(self, x, y):

        if self.obstacle_total[x,y] == 255:
            return 1
        else:
            return 0

    def get_map(self):
        map_matrix = self.sailable_area/255
        return map_matrix
    
    def query_point(self):
        gui = MapGUI()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', gui.save_point)

        while(gui.clicks < 2):
            cv2.imshow('image', self.image)
            if cv2.waitKey(20) & 0xFF == 27:
                break
           
        cv2.destroyAllWindows()

        return gui.points




    def print_path(self, path):
        try:
            for coord in path:
                self.image[coord[0], coord[1]] = [255, 0, 0]

            plt.imshow(self.image, cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            plt.show()
        except:
            print('path not found!')

class MapGUI:
    def __init__(self):
        self.points = []
        self.clicks = 0
    
    def save_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append(pt.Point(x,y))
            self.clicks += 1
