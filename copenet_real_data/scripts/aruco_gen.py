import numpy as np
import matplotlib as mpl
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt


aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

fig = plt.figure()
nx = 4
ny = 3
for i in range(1, nx*ny+1):
    ax = fig.add_subplot(ny,nx, i)
    img = aruco.drawMarker(aruco_dict,i, 600)
    plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")
