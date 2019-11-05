import cv2
import numpy as np
from matplotlib import pyplot as plt

FLANN_INDEX_LSH = 6


MAP = '/home/john/Pictures/simpleTest.png'
TEMPLATE = '/home/john/Pictures/arrow4.png'
#MAP = '/home/john/Pictures/opencvtest.png'



TRAFFIC_MIN1 = np.array([150, 10, 10])
TRAFFIC_MAX1 = np.array([151, 200, 255])

map_image = cv2.imread(TEMPLATE,1)
map_image = cv2.resize(map_image, (0,0), fx=2, fy=2)
hsv_img = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_threshed = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
map_mask = cv2.inRange(map_threshed, TRAFFIC_MIN1, TRAFFIC_MAX1)

map_image2 = cv2.imread(MAP,1)
hsv_img2 = cv2.cvtColor(map_image2, cv2.COLOR_BGR2HSV)
map_threshed2 = cv2.cvtColor(map_image2, cv2.COLOR_BGR2HSV)
map_mask2 = cv2.inRange(map_threshed2, TRAFFIC_MIN1, TRAFFIC_MAX1)


orb = cv2.ORB_create(patchSize=2)

kp1 = orb.detect(map_mask,None)
print(kp1)

kp1, des1 = orb.compute(map_mask, kp1)

print(kp1)

kp2 = orb.detect(map_mask2,None)

kp2, des2 = orb.compute(map_mask2, kp2)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2search_params = dict(checks=50)   # or pass empty dictionary
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(map_image,kp1,map_image2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()



#img2 = cv2.drawKeypoints(map_mask,kp,color=(0,255,0),flags=0)
#plt.imshow(img2),plt.show()


# corners = cv2.goodFeaturesToTrack(map_mask,500,0.25,1)
# corners = np.int0(corners)

# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(map_image,(x,y),3,255,-1)

# plt.imshow(map_image),plt.show()