# import the necessary packages
import numpy as np
import argparse
import cv2

def detect_barcode(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=1)
    closed = cv2.dilate(closed, None, iterations=1)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    img, cnts, hier = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    else:
        return [False, 0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    return [True,np.int0(cv2.boxPoints(rect))]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="path to the image file")
    # args = vars(ap.parse_args())
    #
    # load the image and convert it to grayscale
    # image = cv2.imread(args["image"])

    image = cv2.imread('E:\\barcodes\\data\\tmp583977-clean.png')
    found, box = detect_barcode(image)
    # draw a bounding box arounded the detected barcode and display the
    # image
    print(box)
    print(type(box))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 1)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
