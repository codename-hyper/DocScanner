import cv2
import numpy as np


def preProcess(img):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGrey, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 80, 200)
    kernal = np.ones((5, 5), np.int32)
    imgDilate = cv2.dilate(imgCanny, kernal, iterations=5)
    imgErode = cv2.erode(imgDilate, kernal, iterations=2)
    return imgErode


def getContour(img):
    maxArea = 0
    big_corner = 0
    big_approx = np.array([])
    contour, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for con in contour:
        area = cv2.contourArea(con)
        if area > 5000:
            # cv2.drawContours(imgContour, con, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(con, True)
            approx = cv2.approxPolyDP(con, 0.02 * perimeter, True)
            corners = len(approx)
            if area > maxArea and corners == 4:
                # x, y, w, h = cv2.boundingRect(approx)
                # cv2.rectangle(imgContour,(x,y),(x+w,y+h),(255,0,0),3)
                maxArea = area
                big_corner = corners
                big_approx = approx
    cv2.drawContours(imgContour, big_approx, -1, (255, 0, 0), 20)
    return big_approx


def warp(img, big_approx):
    big_approx = order(big_approx)
    pt1 = np.float32(big_approx)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    return cv2.warpPerspective(img, matrix, (width, height))


def order(points):
    points = points.reshape(4, 2)
    new_points = np.zeros((4, 1, 2), np.int32)
    sum_points = points.sum(axis=1)
    # return sum_points
    new_points[0] = points[np.argmin(sum_points)]
    new_points[3] = points[np.argmax(sum_points)]
    diff_point = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff_point)]
    new_points[2] = points[np.argmax(diff_point)]
    return new_points


width, height = 1280, 720
# cap = cv2.VideoCapture(0)
# cap.set(3, width)
# cap.set(4, height)
# cap.set(10, 150)

while True:
    # success, img = cap.read()
    img = cv2.resize(cv2.imread('Resources/certi.jpg'), (width, height))
    image = preProcess(img)
    imgContour = img.copy()
    x = getContour(image)
    result = warp(img, x)
    # print(x)
    # print(x.shape)

    cv2.imshow('output', result)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
