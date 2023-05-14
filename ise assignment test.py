import cv2
import numpy as np
from matplotlib import pyplot as plt

#Notes for Each Image
# For isePhoto4 set rssize to (800,1050)
#
# For isePhoto6 set dilation iteration to 2
#
# For isePhoto7 set threshold to 174
#
# For isePhoto8 set threshold to 145
#
# For isePhoto9 set threshold to 160 dan iteration to 2
#
# For isePhoto10 set threshold to 120 and make area search to >1000

#Function to rank areas in descending order
def rank_areas(contours):
    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            areas.append(area)
    areas.sort(reverse=True)
    return areas

# Read image
image = cv2.imread('isePhoto2.jpg')
image = cv2.resize(image, (400, 640))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
_, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

# erosion and dilation
kernel = np.ones((7, 7), np.uint8)

dilation = cv2.dilate(thresh, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

triangle = 0
rectangular = 0
pentagon = 0
round = 0
total_object = 0

areas = rank_areas(contours)

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:
        total_object += 1
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        cv2.drawContours(image, contour, -1, (255, 0, 255), 4)

        if len(approx) == 3:
            triangle += 1
            shape = "Triangle"
        elif len(approx) == 4:
            rectangular += 1
            shape = "Rectangle"
        elif len(approx) == 5:
            pentagon += 1
            shape = "Pentagon"
        else:
            round += 1
            shape = "Round"

        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(image, f"{shape}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        rank = areas.index(area) + 1
        cv2.putText(image, f"Size Ranking: {rank} - Area: {area}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Show the image
cv2.imshow('Image', image)
cv2.imshow('gray', gray)
cv2.imshow('threshold', thresh)
cv2.imshow('dilation', dilation)

print("triangular : ", triangle)
print("rectangular : ", rectangular)
print("pentagon : ", pentagon)
print("round : ", round)
print("total objects :", total_object)

cv2.waitKey(0)
cv2.destroyAllWindows()