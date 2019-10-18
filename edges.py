import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('home/john/Pictures/testing_image.png')
edges = cv2.Canny(img,100,200)


cv2.imshow('edges',edges)


