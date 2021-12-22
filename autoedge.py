#adding a line  for testing
import argparse
import cv2
from matplotlib import pyplot as plt
from imutils import perspective
from imutils import contours
import imutils
import numpy as np
from scipy.spatial import distance as dist
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, required=True,
# 	help="path to input image")
# ap.add_argument("-s", "--scharr", type=int, default=0,
# 	help="path to input image")
# args = vars(ap.parse_args())

# load the image, convert it to grayscale, and display the original
# grayscale image
'''image = cv2.imread('/home/qualitas/Documents/Projects/TVS_OD/Images/Pic_2021_04_20_141036_blockId#48648.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
scharr=0

# set the kernel size, depending on whether we are using the Sobel
# operator of the Scharr operator, then compute the gradients along
# the x and y axis, respectively
ksize = -1 if scharr > 0 else 3
gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
# the gradient magnitude images are now of the floating point data
# type, so we need to take care to convert them back a to unsigned
# 8-bit integer representation so other OpenCV functions can operate
# on them and visualize them
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)
# combine the gradient representations into a single image
combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
# show our output images
cv2.imshow("Sobel/Scharr X", gX)
cv2.imshow("Sobel/Scharr Y", gY)
cv2.imshow("Sobel/Scharr Combined", combined)
cv2.waitKey(0)'''

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# load the image, convert it to grayscale, and display the original
# grayscale image
image = cv2.imread('/home/qualitas/Documents/Projects/TVS_OD/Images/Pic_2021_04_22_182513_blockId#19197.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
scharr=0
# set the kernel size, depending on whether we are using the Sobel
# operator of the Scharr operator, then compute the gradients along
# the x and y axis, respectively
ksize = -1 if scharr > 0 else 3
gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
# the gradient magnitude images are now of the floating point data
# type, so we need to take care to convert them back a to unsigned
# 8-bit integer representation so other OpenCV functions can operate
# on them and visualize them
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)
# combine the gradient representations into a single image
combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
# show our output images
# cv2.imshow("Sobel/Scharr X", gX)
# cv2.imshow("Sobel/Scharr Y", gY)
# cv2.imshow("Sobel/Scharr Combined", combined)
# cv2.waitKey(0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(combined, 60, 110)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# plt.imshow(edged)
# plt.show()
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts,)
pixelsPerMetric = None

con_dict={}
# for c in cnts:
#     con_dict[c]=cv2.contourArea(c)

sort_dict=sorted(cnts,key=lambda x:cv2.contourArea(x))


# loop over the contours individually
#for c in sort_dict:
	# if the contour is not sufficiently large, ignore it
	#if cv2.contourArea(c) < 100:
		#continue
c=sort_dict[-1]
print(cv2.contourArea(c))
# compute the rotated bounding box of the contour
orig = image.copy()
box = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
box = np.array(box, dtype="int")
# order the points in the contour such that they appear
# in top-left, top-right, bottom-right, and bottom-left
# order, then draw the outline of the rotated bounding
# box
box = perspective.order_points(box)
cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
# loop over the original points and draw them
for (x, y) in box:
    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)


# unpack the ordered bounding box, then compute the midpoint
# between the top-left and top-right coordinates, followed by
# the midpoint between bottom-left and bottom-right coordinates
(tl, tr, br, bl) = box
(tltrX, tltrY) = midpoint(tl, tr)
(blbrX, blbrY) = midpoint(bl, br)
# compute the midpoint between the top-left and top-right points,
# followed by the midpoint between the top-righ and bottom-right
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)
# draw the midpoints on the image
cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
# draw lines between the midpoints
cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
    (255, 0, 255), 2)
cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
    (255, 0, 255), 2)

# compute the Euclidean distance between the midpoints
dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
# if the pixels per metric has not been initialized, then
# compute it as the ratio of pixels to supplied metric
# (in this case, inches)
#if pixelsPerMetric is None:
    #pixelsPerMetric = dB / args["width"]


# compute the size of the object
dimA = dA #/ pixelsPerMetric
dimB = dB #/ pixelsPerMetric
# draw the object sizes on the image
cv2.putText(orig, "{:.1f}in".format(dimA),
    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
    0.65, (255, 255, 255), 2)
cv2.putText(orig, "{:.1f}in".format(dimB),
    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
    0.65, (255, 255, 255), 2)
cv2.putText(orig, "{:.1f}".format(cv2.contourArea(c)),
    (int(trbrX + 100), int(trbrY)+40), cv2.FONT_HERSHEY_SIMPLEX,
    0.90, (255, 255, 255), 2)
# show the output image
plt.imshow(orig)
plt.show()
