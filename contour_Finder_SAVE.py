import cv2
import numpy as np
from matplotlib import pyplot as plt

new_contours = []
score = []
MAP = '/home/john/Pictures/simpleTest.png'
ARROW = '/home/john/Pictures/arrow_example.png'
BROKEN = '/home/john/Pictures/broken_arrow.png'

TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])

#process map
map_image = cv2.imread(MAP,1)
map_threshed = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)
map_blur = cv2.GaussianBlur(map_mask,(3,3),0)
map_contour, map_hiearchy = cv2.findContours(map_blur,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)




#process example arrow
test_image = cv2.imread(ARROW, 1)
test_threshed = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
test_mask = cv2.inRange(test_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)
blur = cv2.blur(test_mask, (2,2))
#rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
#threshed = cv2.morphologyEx(testmask, cv2.MORPH_CLOSE, rect_kernel)
test_contour, test_heirarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
test_area = cv2.contourArea(test_contour[0])

#process broken arrow
#process example arrow
test_image2 = cv2.imread(BROKEN, 1)
test_threshed2 = cv2.cvtColor(test_image2, cv2.COLOR_BGR2HSV)
test_mask2 = cv2.inRange(test_threshed2, TRAFFIC_MIN1, TRAFFIC_MAX1)
test_mask3 = cv2.Canny(test_mask2,0,150)
blur2 = cv2.blur(test_mask3, (2,2))
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
threshed2 = cv2.morphologyEx(blur2, cv2.MORPH_CLOSE, rect_kernel)
ret,thresh1 = cv2.threshold(blur2,62,255,cv2.THRESH_BINARY)
test_contour2, test_heirarchy2 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
test_area2 = cv2.contourArea(test_contour2[0])

print(len(test_contour2))


#draw rectangles around matched contour
count = 0
for contour in map_contour:
    if ((cv2.contourArea(contour) <= 1.5*test_area and cv2.contourArea(contour) >= 0.8*test_area and cv2.matchShapes(contour,test_contour[0], 3,0.0) < 1) 
    or (cv2.contourArea(contour) <= 1.7*test_area2 and cv2.contourArea(contour) >= 0.2*test_area2  and cv2.matchShapes(contour, test_contour2[0],1,0.0) < 5)):
    #if cv2.contourArea(contour) >= 5*test_area:
        count += 1
        [X, Y, W, H] = cv2.boundingRect(contour)
        cv2.rectangle(map_image, (X,Y), (X+W, Y+H), (255, 0, 0), 2)
        new_contours.append(contour)


#cv2.drawContours(map_image, new_contours, -1, (0,255,0), 3)

print(count)
plt.imshow(map_image,cmap='gray')
plt.show()
