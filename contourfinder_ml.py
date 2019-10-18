import cv2
import numpy as np
from matplotlib import pyplot as plt

new_contours = []
score = []
IMAGEPATH = '/home/john/Pictures/trickyarrows.png'
ARROW = '/home/john/Pictures/arrow_example.png'


TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])


image = cv2.imread(IMAGEPATH, 1)
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
trafficMask = cv2.inRange(hsv_img, TRAFFIC_MIN1, TRAFFIC_MAX1)
traffic_contour, traffic_hierarchy = cv2.findContours(trafficMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

testimage = cv2.imread(ARROW, 1)
tst_img = cv2.cvtColor(testimage, cv2.COLOR_BGR2HSV)
testmask = cv2.inRange(tst_img, TRAFFIC_MIN1, TRAFFIC_MAX1)
test_contour, test_heirarchy = cv2.findContours(testmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

test_area = cv2.contourArea(test_contour[0])
#cv2.drawContours(image, traffic_contour, -1, (0,255,0), 3)
#plt.imshow(image, cmap='gray')
#plt.show()

rect = cv2.minAreaRect(test_contour[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(testimage, [box], 0, (0,0,255),2)


moment_box = cv2.moments(box)
moment_arrow = cv2.moments(test_contour[0])

cx_box = int(moment_box['m10']/moment_box['m00'])
cy_box = int(moment_box['m01']/moment_box['m00'])

cx_arrow = int(moment_arrow['m10']/moment_arrow['m00'])
cy_arrow = int(moment_arrow['m01']/moment_arrow['m00'])

slope = (cy_arrow - cy_box)/(cx_arrow - cx_box)


print(str(cx_box) + ', ' + str(cy_box))
print(str(cx_arrow)c + ', ' + str(cy_arrow))
print(slope)


rows,cols = testimage.shape[:2]
[vx,vy,x,y] = cv2.fitLine(test_contour[0], cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(testimage,(cols-1,righty),(0,lefty),(0,255,0),2)







plt.imshow(testimage, cmap='gray')
plt.show()



#for contour in traffic_contour:
#    cv2.drawContours(image, contour, -1, (0,255,0), 3)
#    plt.imshow(image, cmap='gray')
#    plt.show()



    #if cv2.matchShapes(contour,test_contour[0], 1,0.0) < 1000 and (cv2.contourArea(contour))/test_area < 1.5 and (cv2.contourArea(contour))/test_area > 0.1:
    #    score.append(cv2.matchShapes(contour,test_contour[0], 1,0.0))
    #    new_contours.append(contour)

print(len(traffic_contour))
print(len(score))
print(score)
    #    score = 
#        new_contours.append(score)

#print(new_contours[2])

#cv2.drawContours(image, new_contours, -1, (0,255,0), 3)
#plt.imshow(image,cmap='gray')
#plt.show()
