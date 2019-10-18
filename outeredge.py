import cv2
import numpy as np
from matplotlib import pyplot as plt

MAP = '/home/john/Pictures/ENC_test.png'


TRAFFIC_MIN1 = np.array([150, 200, 10])
TRAFFIC_MAX1 = np.array([151, 255, 255])

map_image = cv2.imread(MAP,1)
map_threshed = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)
map_canny = cv2.Canny(map_mask, 0,3)
#map_blur = cv2.GaussianBlur(map_canny,(3,3),0)
map_blur = cv2.blur(map_mask,(3,3))
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
threshed2 = cv2.morphologyEx(map_blur, cv2.MORPH_CLOSE, rect_kernel)
#ret,thresh1 = cv2.threshold(blur2,62,255,cv2.THRESH_BINARY)
map_contour, map_hiearchy = cv2.findContours(threshed2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(len(map_contour))

cv2.drawContours(map_image, map_contour, -1, (0,255,0), 3)

plt.imshow(threshed2,cmap='gray')
plt.show()
