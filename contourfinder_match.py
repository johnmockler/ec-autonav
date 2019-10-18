import cv2
import numpy as np
from matplotlib import pyplot as plt

new_contours = []
score = []
IMAGEPATH = '/home/john/Pictures/ECN_test.png'
ARROW = '/home/john/Pictures/arrow_example.png'


TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])


image = cv2.imread(IMAGEPATH, 1)
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
trafficMask = cv2.inRange(hsv_img, TRAFFIC_MIN1, TRAFFIC_MAX1)
traffic_contour, traffic_hierarchy = cv2.findContours(trafficMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

testimage = cv2.imread(ARROW, 1)
tst_img = cv2.cvtColor(testimage, cv2.COLOR_BGR2HSV)
testmask = cv2.inRange(tst_img, TRAFFIC_MIN1, TRAFFIC_MAX1)
test_contour, test_heirarchy = cv2.findContours(testmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

test_area = cv2.contourArea(test_contour[0])

#cv2.drawContours(testimage, test_contour, -1, (0,255,0), 3)

#plt.imshow(testimage, cmap='gray')
#plt.show()

for contour in traffic_contour:
    if cv2.matchShapes(contour,test_contour[0], 1,0.0) < 1000 and (cv2.contourArea(contour))/test_area < 1.5 and (cv2.contourArea(contour))/test_area > 0.1:
        score.append(cv2.matchShapes(contour,test_contour[0], 1,0.0))
        new_contours.append(contour)

print(len(traffic_contour))
print(len(score))
print(score)
    #    score = 
#        new_contours.append(score)

#print(new_contours[2])

cv2.drawContours(image, new_contours, -1, (0,255,0), 3)
plt.imshow(image,cmap='gray')
plt.show()
