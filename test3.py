import cv2
import  numpy as np
import matplotlib as plt
img = np.zeros((1080, 1920, 3), np.uint8)
area1 = np.array([[250, 200], [300, 100], [750, 800], [100, 1000]])
area2 = np.array([[1000, 200], [1500, 200], [1500, 400], [1000, 400]])
cv2.polylines(img, [area1], True, 255, 2)
# cv2.fillPoly(img, [area1, area2],   255)
cv2.fillPoly(img, [area1 ],   255)

# plt.imshow(img)
cv2.imshow("img",img)
cv2.waitKey(0)