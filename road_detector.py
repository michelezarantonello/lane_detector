import typing as tp
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from utils import is_dash, draw_lines, cluster_lines, fill_between_lines, fill_lines, pointsum_in_line, polar2cartesian, points_in_line

image = cv.imread("./roads/road1.png")
roi = image[450:,:,:]


cv.namedWindow("original_img")
cv.imshow("original_img", image)
cv.waitKey(0)
cv.destroyAllWindows()
gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.namedWindow("gray_img")
cv.imshow("gray_img", gray_img)
cv.waitKey(0)
cv.destroyAllWindows()



#cv.namedWindow("road1b")
#def on_trackbar(_):
#    tl = cv.getTrackbarPos("tl", "road1b") # lower threshold ->  pixels below tl wont be considered edges
#    th = cv.getTrackbarPos("th", "road1b") # higher threshold ->  pixels above it will be considered edgesb
#    # pixels between tl and th are edges if they can be connected to pixels above th
#    edges = cv.Canny(image, tl, th, apertureSize=3)
#    cv.imshow("road1b", edges)
#cv.createTrackbar("th", "road1b", 0, 2000, on_trackbar)
#cv.createTrackbar("tl", "road1b", 0, 1000, on_trackbar) #th=850 tl=400 for road1b
#on_trackbar(0)
#cv.waitKey(0)
#cv.destroyAllWindows()
# print("image shape: ", image.shape)

th, tl = 850, 410 #default value for image 1: 850, 410. General 500,200
edges = cv.Canny(image, tl, th, apertureSize=3)
cv.namedWindow("edges"),
cv.imshow("edges", edges)
cv.waitKey(0)
cv.destroyAllWindows()

lines1_polar = cv.HoughLines(edges, rho=0.8, theta=np.pi/180, threshold=80, min_theta=0, max_theta=np.pi*5/12)
#since theta varies from 0=vertical to np.pi and we want that theta belongs to roughly -50deg,+50deg we need to split lines calc
lines2_polar = cv.HoughLines(edges, rho=0.8, theta=np.pi/180, threshold=80, min_theta=np.pi*7/12, max_theta=np.pi)
#best values for road1b: rho=0.8, theta=np.pi/180, threshold=80, min_theta=np.pi*7/12, max_theta=np.pi

line_img, mask1 = draw_lines(image, lines1_polar[:,0])
line_img, mask2 = draw_lines(line_img, lines2_polar[:,0])
cv.namedWindow("line_img"),
cv.imshow("line_img", line_img)
cv.waitKey(0)
cv.destroyAllWindows()
    

clusters_1 = cluster_lines(lines1_polar[:,0], rho_thresh=5, theta_thresh=np.deg2rad(5))
clusters_2 = cluster_lines(lines2_polar[:,0], rho_thresh=5, theta_thresh=np.deg2rad(5))

rep_lines_1 = np.array([
    [c[:,0].mean(), c[:,1].mean()]
    for c in clusters_1
])
rep_lines_2 = np.array([
    [c[:,0].mean(), c[:,1].mean()]
    for c in clusters_2
])

rep_lines = np.vstack((rep_lines_1, rep_lines_2))




cluster_lines_img, mask1 = draw_lines(image, rep_lines)


cv.namedWindow("cluster_lines_img")
cv.imshow("cluster_lines_img", cluster_lines_img)
cv.waitKey(0)
cv.destroyAllWindows()



dashed_lines = []
continuous_lines = []
max_gap_between_points = 30 #30
max_mean_gap_between_points = 1.15 #1.15
for line in rep_lines:
    if(is_dash(line, edges)) and np.max(points_in_line(line, edges)) > max_gap_between_points and np.mean(points_in_line(line, edges)) > max_mean_gap_between_points:
        dashed_lines.append(line)
    else:
        continuous_lines.append(line)


dashed_lines = np.array(dashed_lines)
continuous_lines = np.array(continuous_lines)
# for line in dashed_lines:
#     print("the mean dashed lines gaps are: ", np.mean(points_in_line(line, edges)))

# for line in continuous_lines:
#     print("the mean continuous_lines gaps are: ", np.mean(points_in_line(line, edges)))

if len(dashed_lines>0):
    dashed_lines_img, mask1 = draw_lines(image, dashed_lines)

# dashed_lines_img = fill_lines(dashed_lines_img, mask1)


#filled_lines_img1 = fill_lines(line_img, mask1)
#filled_lines_img = fill_lines(filled_lines_img1, mask2)


cv.namedWindow("dashed_lines_img") 
cv.imshow("dashed_lines_img", dashed_lines_img)
cv.waitKey(0)
cv.destroyAllWindows()


if len(continuous_lines>0):
    continuous_lines_img, mask1 = draw_lines(image, continuous_lines)


cv.namedWindow("continuous_lines_img")
cv.imshow("continuous_lines_img", continuous_lines_img)
cv.waitKey(0)
cv.destroyAllWindows()


#to detect if lines are continuous or segmented i tried to count number of points in the lines -> low number means
#segmented lines, high number of points in the line -> continuous line
#->approach not robust due to "hard code"->number of points in the line depends on where the foto was taken, output of canny,etc
#idea: use of houghLinesP or LineSegmentDetector


#idea now is to intersect each detected line with the bottom row of the image and the top row of the roi. 
#by doing so we will be able to define the street lines as segment which can be then used to find lanes. 
y_top = 240 #arbitrary values
y_bottom = image.shape[0] - 1 # = 629
#intersection of line with top and bottom part of the image/roi
line_segments = []
for line in rep_lines:
    x_bottom = int((line[0] - y_bottom * np.sin(line[1])) / np.cos(line[1])) 
    x_top = int((line[0] - y_top * np.sin(line[1])) / np.cos(line[1]))
    line_segments.append([(x_bottom, y_bottom), (x_top, y_top)])
# print("line segments are: ", line_segments)
def get_x_bottom(line_seg):
    return line_seg[0][0]
line_segments.sort(key=get_x_bottom)
# print("sorted line segments are: ", line_segments)
max_xbottom_gap = 290
lanes = []
for i in range(len(line_segments)-1):
    if np.abs(line_segments[i][0][0] - line_segments[i+1][0][0]) > max_xbottom_gap:
        lanes.append((line_segments[i], line_segments[i+1]))
print("lanes are: ", lanes)

lanes_polygon = []
for lane in lanes:
    lanes_polygon.append(np.array([lane[0][0], lane[0][1], lane[1][1], lane[1][0]], dtype=np.int32))
print("lanes_polygon are: ", lanes_polygon)
lane_mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv.fillPoly(lane_mask, [lanes_polygon[1]], 255) #to fill every polygone (pay attention because intersection will merge together cv.fillPoly(lane_mask, lanes_polygon, 255))
lane_img = image.copy()
lane_img[lane_mask == 255] = (0, 255, 0)

cv.namedWindow("lane_img")
cv.imshow("lane_img", lane_img)
cv.waitKey(0)
cv.destroyAllWindows()

#YOLO doc: https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html

# net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# ln = net.getUnconnectedOutLayersNames()

# blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
# net.setInput(blob)
# outputs = net.forward(ln)

# print(outputs)
# for out in outputs:
#     print(out.shape)

WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None
vehicles_box = [] #has (x,y,w,h)

# Load names of classes and get random colors
classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# determine the output layer
ln = net.getUnconnectedOutLayersNames()

def load_image(path):
    global img, img0, outputs, ln
    vehicles_box.clear()

    img0 = cv.imread(path)
    img = img0.copy()
    
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    # combine the 3 output groups into 1 (10647, 85)
    # large objects (507, 85)
    # medium objects (2028, 85)
    # small objects (8112, 85)
    outputs = np.vstack(outputs)

    post_process(img, outputs, 0.3)
    cv.imshow('window',  img)
    cv.waitKey(0)

def post_process(img, outputs, conf=0.3):
    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            p1 = int(x + w//2), int(y + h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # cv.rectangle(img, p0, p1, WHITE, 1)
    nms_thresh = 0.4
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, nms_thresh)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if classes[classIDs[i]] in {"car", "bus", "truck", "motorbike"}:
                vehicles_box.append((x,y,w,h))

# def trackbar(x):
#     global img, img0, outputs

#     if img0 is None or outputs is None:
#         return
#     conf = x/100
#     img = img0.copy()
#     post_process(img, outputs, conf)
#     cv.displayOverlay('window', f'confidence level={conf}')
#     cv.imshow('window', img)

cv.namedWindow('window')
# cv.createTrackbar('confidence', 'window', 50, 100, trackbar)
load_image('./roads/road13.png')
#best confidence level = 0.15
cv.destroyAllWindows()

print(vehicles_box)#xywh

#check in which lane cars are
# Dictionary to store vehicles per lane
vehicles_in_lanes = {i: [] for i in range(len(lanes_polygon))}

for box in vehicles_box:
    x, y, w, h = box
    x_center = x + w // 2  # get the center x of the vehicle
    
    # Check which lane this vehicle belongs to
    for lane_idx, poly in enumerate(lanes_polygon):
        x_min = min(poly[:, 0])
        x_max = max(poly[:, 0])
        
        if x_min <= x_center <= x_max:
            vehicles_in_lanes[lane_idx].append(box)
            break 
print("VEHICLES_IN_LANES", vehicles_in_lanes)

#check if car is too close 
Y_THRESHOLD_ALLARM = 450
X_THRESHOLD_ALLARM = 100
for box in vehicles_box:
    x, y, w, h = box
    y_bottom = y + h #lowest point of the car (rear wheels more or less)
    x_center = x + w // 2
    if abs(x_center - image.shape[1]/2) < X_THRESHOLD_ALLARM and y_bottom > Y_THRESHOLD_ALLARM:
        print("VEHICLE IN FRONT OF YOU, BRAKE!")
    # for x axis i assumed that my car is positioned in the middle (along x axis) of my image      
    if (x_center - image.shape[1]/2) > X_THRESHOLD_ALLARM and y_bottom > Y_THRESHOLD_ALLARM: 
        print("VEHICLE ON YOUR RIGHT, PAY ATTENTION!")
    if (x_center - image.shape[1]/2) < - X_THRESHOLD_ALLARM and y_bottom > Y_THRESHOLD_ALLARM: 
        print("VEHICLE ON YOUR LEFT, PAY ATTENTION!")