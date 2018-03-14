import cv2 as cv
import numpy as np


def displayImage(image) :
    cv.imshow('output', image);
    cv.waitKey(0);
    cv.destroyAllWindows();

def get_contour_areas(contours):
    # returns the areas of all contours as list
    all_areas = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        all_areas.append(area)
    return all_areas

def preProcessing(image):

    image = cv.adaptiveThreshold(image,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,25, 2)
    #ret,image = cv.threshold(image,150,255,cv.THRESH_BINARY);


    kernel = np.ones((3, 3), np.uint8)
    image = cv.erode(image, kernel, iterations=3)
    kernel = np.ones((3, 3), np.uint8)
    image = cv.dilate(image, kernel, iterations=5)
    #displayImage(image);


    return image;

def drawBillOutLine(input , orig):
    cnt = getBillOutLine(input)
    cv.drawContours(orig, [cnt], -1, (0, 255, 0), 1);
    return orig;

def getBillOutLine(input):
    _, contours, hierarchy = cv.findContours(input, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    cnt = sorted_contours[0];
    accuracy = 0.1 * cv.arcLength(cnt, True)
    cnt = cv.approxPolyDP(cnt, accuracy, True)
    return cnt;



#main
image = cv.imread('bill.jpg');
image = cv.resize(image, (0, 0), fx=0.1, fy=0.1);

gray = cv.cvtColor( image, cv.COLOR_RGB2GRAY );


input = gray.copy();
preprocessed = preProcessing(input);
output = drawBillOutLine(preprocessed,image.copy())
displayImage(output);

#crop the bill out of the image
cnt = getBillOutLine(preprocessed);
print cnt[0] , cnt[1] , cnt[2] ,cnt[3]
if(len(cnt)==4):
    #it is a rectangle
    mask = np.zeros(gray.shape, np.int32);
    poly = np.array([cnt],np.int32);
    cv.fillPoly(mask,poly,255);
    #be wary of the dtype and the size of the arrays

    test = np.array(input.copy(),np.int32)
    test = cv.bitwise_and(test,mask);
    #you cannot display an image of type int32
    test = np.array(test,np.uint8)
    displayImage(test);