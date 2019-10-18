import cv2
import numpy as np
from matplotlib import pyplot as plt

new_contours = []
score = []

ARROW = '/home/john/Pictures/broken_arrow.png'

TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])


#process example arrow
test_image = cv2.imread(ARROW, 1)
test_threshed = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
test_mask = cv2.inRange(test_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)
test_mask2 = cv2.Canny(test_threshed,0,150)
blur = cv2.blur(test_mask2, (2,2))
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
threshed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, rect_kernel)
ret,thresh1 = cv2.threshold(blur,62,255,cv2.THRESH_BINARY)
test_contour, test_heirarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
test_area = cv2.contourArea(test_contour[0])
print(len(test_contour))


#draw rectangles around matched contour


cv2.drawContours(test_image, test_contour, -1, (0,255,0), 3)

plt.imshow(blur,cmap='gray')
plt.show()
