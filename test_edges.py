import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/john/Pictures/ENC_test.png', 0)
edges = cv2.Canny(img,100,200)

plt.imshow(edges,cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()