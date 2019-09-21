import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/john/Pictures/ENC_test (copy).png', 0)

px = img[100,100,2]

print(px)