import typing as tp
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from utils import is_dash,is_road_line, draw_lines, cluster_lines, fill_between_lines, fill_lines, pointsum_in_line, polar2cartesian, points_in_line

image = cv.imread("./roads/road10.png")
roi = image[220:,300:900,:]
print(image.shape)
cv.namedWindow("original_img")
cv.imshow("original_img", roi)
cv.waitKey(0)
cv.destroyAllWindows()

gray_img = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
cv.namedWindow("gray_img")
cv.imshow("gray_img", gray_img)
cv.waitKey(0)
cv.destroyAllWindows()

th, tl = 250, 40 #default value for image 1: 850, 410
edges = cv.Canny(roi, tl, th, apertureSize=3)
cv.namedWindow("edges"),
cv.imshow("edges", edges)
cv.waitKey(0)
cv.destroyAllWindows()

lines1_polar = cv.HoughLines(edges, rho=0.8, theta=np.pi/180, threshold=40, min_theta=0, max_theta=np.pi*5/12)
#since theta varies from 0=vertical to np.pi and we want that theta belongs to roughly -50deg,+50deg we need to split lines calc
lines2_polar = cv.HoughLines(edges, rho=0.8, theta=np.pi/180, threshold=40, min_theta=np.pi*7/12, max_theta=np.pi)
#best values for road1b: rho=0.8, theta=np.pi/180, threshold=80, min_theta=np.pi*7/12, max_theta=np.pi

line_img, mask1 = draw_lines(roi, lines1_polar[:,0])
line_img, mask2 = draw_lines(line_img, lines2_polar[:,0])
cv.namedWindow("line_img"),
cv.imshow("line_img", line_img)
cv.waitKey(0)
cv.destroyAllWindows()
    

clusters_1 = cluster_lines(lines1_polar[:,0], rho_thresh= 3, theta_thresh=np.deg2rad(6))
clusters_2 = cluster_lines(lines2_polar[:,0], rho_thresh= 3, theta_thresh=np.deg2rad(6))

rep_lines_1 = np.array([
    [c[:,0].mean(), c[:,1].mean()]
    for c in clusters_1
])
rep_lines_2 = np.array([
    [c[:,0].mean(), c[:,1].mean()]
    for c in clusters_2
])

rep_lines = np.vstack((rep_lines_1, rep_lines_2))




cluster_lines_img, mask1 = draw_lines(roi, rep_lines)


cv.namedWindow("cluster_lines_img")
cv.imshow("cluster_lines_img", cluster_lines_img)
cv.waitKey(0)
cv.destroyAllWindows()

street_lines = []

for line in rep_lines:
    if  is_road_line(line, roi, offset=10, min_delta_L=5):
        street_lines.append(line)
print(street_lines)
street_lines = np.array(street_lines)
print("lol", street_lines)
street_lines_img, street_line_mask = draw_lines(roi, street_lines)

cv.namedWindow("street_lines_img")
cv.imshow("street_lines_img", street_lines_img)
cv.waitKey(0)
cv.destroyAllWindows()