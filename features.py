import cv2
import numpy as np
from matplotlib import pyplot as plt

MAP = '/home/john/Pictures/simpleTest.png'
TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])

#process map
map_image = cv2.imread(MAP,1)
map_threshed = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)
corners = cv2.goodFeaturesToTrack(map_mask,50,0.6,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(map_image,(x,y),3,255,-1)

plt.imshow(map_image),plt.show()