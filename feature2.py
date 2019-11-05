import cv2
import numpy as np
from matplotlib import pyplot as plt
import math as m
import ArrowObj

#MAP = '/home/john/Pictures/simpleTest.png'
MAP = '/home/john/Pictures/ENC_dataset.png'

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
TRAFFIC_MAX1 = np.array([151, 200, 255])

SILVER_LINE_MIN = np.array([0, 0, 150])
SILVER_LINE_MAX = np.array([0, 0, 200])

RED_LINE_MIN = np.array([0, 250, 175])
RED_LINE_MAX = np.array([0, 255, 255])

GREEN_BUOY_MIN = np.array([59, 200, 200])
GREEN_BUOY_MAX = np.array([61, 255, 255])

YELLOW_BUOY_MIN = np.array([25, 250, 250])
YELLOW_BUOY_MAX = np.array([27, 255, 255])

MIN_LENGTH = 15.00

arrows = []

#process map
map_image = cv2.imread(MAP,1)
hsv_img = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_threshed = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
#map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)
map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)

#corners = cv2.goodFeaturesToTrack(map_mask,500,0.02,10)
#corners = np.int0(corners)
#np.sort(corners,axis=1)

def detectArrows(mask):
    #pre-process map before input to function  
    corners = cv2.goodFeaturesToTrack(mask,500,0.02,10)
    corners = np.int0(corners)
    arrows = []

    for i in corners:
        xi,yi = i.ravel()
        arrow = Arro
        arrow = ArrowObj.ArrowObject(xi,yi)
        arrows.append(arrow)

    for i in arrows:
        for j in arrows:
            if i.


 


    #corners are probably arrows


def distance(x1,y1, x2,y2):
    d = m.sqrt((x1-x2)**2 + (y1-y2)**2)
    return d

def direction(x1,y1,x2,y2):
    dx = x1 - x2
    dy = y1 - y2
    theta = m.asin(dy/dx)
    return theta

#pairs items in an array based on their closest neighbor match closest neighbor then kick out if another is closers. for ones without pairs 
def midpoint(x1,x2, y1, y2):
    x = (x1 - x2)/2
    y = (y1 - y2)/2

    return x,y

def capture(x,y,num,img):
    l = 17
    x1 = x - l
    x2 = x + l
    y1 = y - l
    y2 = y + l
    arrow_img = img[y1:y2,x1:x2]
    #arrow_img = (255-arrow_img)
    arrow_threshed = cv2.cvtColor(arrow_img, cv2.COLOR_BGR2HSV)
    arrow_mask = cv2.inRange(arrow_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)
    #arrow_gray = cv2.cvtColor(arrow_mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('/home/john/Pictures/Others trainingset2/'+str(num)+'.png', arrow_mask)

def findArrows(array):

    paired = []
    unpaired = []

    for a in array:
        for b in array:
            dist = distance(a,b)
            if dist != 0 and dist < a.dist and dist < MIN_LENGTH:
                #array.remove(a)
                #array.remove(b)
                #a.pair = b
                #a.dist = dist
                #b.pair = a
                #b.dist = dist
                #paired.append(a)                               
                cx,cy = midpoint(a.x,b.x,a.y,b.y)
                a.x = cx
                a.y = cy
                array.remove(b)

            #cv2.line(map_image,(a.x,a.y),(b.x,b.y),(0,255,0),thickness=3, lineType=8)

    unpaired.append(array)

    return paired, unpaired
#update this function to only search a given boundary?
def loadMap():
    # FIND CONTOURS
    kernel = np.ones((5, 5), np.uint8)
    landMask = cv2.inRange(hsv_img, LAND_MIN, LAND_MAX)
    land_contour, land_hierarchy = cv2.findContours(landMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shallowMask = cv2.inRange(hsv_img, SHALLOW_DEPTH_MIN, SHALLOW_DEPTH_MAX)
    shallowBorderMask = cv2.inRange(hsv_img, SHALLOW_BOUND_MIN, SHALLOW_BOUND_MAX)
    shallowMaskAll = cv2.bitwise_or(shallowMask, shallowBorderMask)
    shallow_contour, shallow_hierarchy = cv2.findContours(shallowMaskAll, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lowMask = cv2.inRange(hsv_img, LO_DEPTH_MIN, LO_DEPTH_MAX)
    low_contour, low_hierarchy = cv2.findContours(lowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    medMask = cv2.inRange(hsv_img, MED_DEPTH_MIN, MED_DEPTH_MAX)
    medBorderMask = cv2.inRange(hsv_img, WATER_BOUND_MIN, WATER_BOUND_MAX)
    medMaskAll = cv2.bitwise_or(medMask, medBorderMask)
    medMaskNoGaps = cv2.morphologyEx(medMaskAll, cv2.MORPH_CLOSE, kernel)
    med_contour, med_hierarchy = cv2.findContours(medMaskNoGaps, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    highMask = cv2.inRange(hsv_img, HI_DEPTH_MIN, HI_DEPTH_MAX)
    high_contour, high_hierarchy = cv2.findContours(highMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    trafficMask = cv2.inRange(hsv_img, TRAFFIC_MIN1, TRAFFIC_MAX1)
    #trafficBlur = cv2.GaussianBlur(trafficMask,(3,3),0)
    traffic_contour, traffic_hierarchy = cv2.findContours(trafficMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    or1 = cv2.bitwise_or(shallowMaskAll, lowMask)
    or2 = cv2.bitwise_or(or1, medMaskAll)
    or3 = cv2.bitwise_or(or2, highMask)
    or4 = cv2.bitwise_or(or3, trafficMask)
    or5 = cv2.bitwise_or(or4, landMask)

    # filter out buoys and random lines
    yellowMask = cv2.inRange(hsv_img, YELLOW_BUOY_MIN, YELLOW_BUOY_MAX)
    greenMask = cv2.inRange(hsv_img, GREEN_BUOY_MIN, GREEN_BUOY_MAX)
    silverMask = cv2.inRange(hsv_img, SILVER_LINE_MIN, SILVER_LINE_MAX)
    redMask = cv2.inRange(hsv_img, RED_LINE_MIN, RED_LINE_MAX)

    or6 = cv2.bitwise_or(or5, yellowMask)
    or7 = cv2.bitwise_or(or6, greenMask)
    or8 = cv2.bitwise_or(or7, silverMask)
    or9 = cv2.bitwise_or(or8, redMask)

    obstacleMask = cv2.bitwise_not(or9)
    obstacle_contour, obstacle_hierarchy = cv2.findContours(obstacleMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return obstacle_contour, obstacleMask

def findObstacle():
    contour,mask = loadMap()
    num = 1
    for cnt in contour:
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])   
        capture(cx,cy,num,mask)
        num += 1

#findObstacle()
num = 1
for i in corners:

    x,y = i.ravel()
    #print(x)
    #print(y)
    #capture(x,y,num, map_image)
    #num += 1
    arrow = ArrowObj.ArrowPair(x,y)
    arrows.append(arrow)
    #print(str(x) + ','+ str(y))

    #cv2.circle(map_image,(x,y),3,255,-1)

for arr in arrows:
    try:
        capture(arr.x,arr.y,num,map_image)
        num +=1
    except:
        num = num

#paired, unpaired = findArrows(arrows) 
#print(paired)


#plt.imshow(map_image),plt.show()






