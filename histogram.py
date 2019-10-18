import cv2
import numpy as np
from matplotlib import pyplot as plt

new_contours = []
score = []
MAP = '/home/john/Pictures/ENC_test.png'
ARROW = '/home/john/Pictures/arrow_example.png'
BROKEN = '/home/john/Pictures/broken_arrow.png'

TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])


roi = cv2.imread(ARROW)
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

target = cv2.imread(MAP)
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)


# calculating object histogram
roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

# normalize histogram and apply backprojection
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,disc,dst)

# threshold and binary AND
ret,thresh = cv2.threshold(dst,50,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)

res = np.vstack((target,thresh,res))



plt.imshow(res,cmap='gray')
plt.show()
