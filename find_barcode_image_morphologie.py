# import the necessary packages
import numpy as np
import argparse
import cv2

def noisy(image):
      row,col= image.shape
      mean = 0
      var = 0.3
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy

def detect_barcode_morp(image):
    # image preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray",gray)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    kernel = np.ones((15, 15), np.uint8)
    bh = cv2.morphologyEx(blur,cv2.MORPH_BLACKHAT,kernel)
    tresh=int(np.max(bh)*0.50)
    ret, threshed = cv2.threshold(bh, tresh, 255, cv2.THRESH_BINARY)
#    img, cnts, hier = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    biggest = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
#    box = cv2.minAreaRect(biggest)
#    box=np.int0( box[0:2])
#    print("Box:",box)
#    line_width=np.sqrt((box[0][0])*([1][0])+(box[0][1])*(box[0][1]))
#    print("line width:",line_width)

    closed=threshed
    closed = cv2.dilate(closed, None, iterations=10)
    closed = cv2.erode(closed, None, iterations=10)

    img, cnts, hier = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    for c in cnts:
#        box=cv2.minAreaRect(c)
#        pnt1=(int(box[0][0]), int(box[0][1]))
#        pnt2=(int(box[1][0]), int(box[1][1]))
#        cv2.rectangle(image, pnt1, pnt2, color=(0, 255, 0), thickness=2)

    biggest = sorted(cnts, key=cv2.contourArea, reverse=False)[0]
    box = cv2.minAreaRect(biggest)
    return [True,box]

if __name__ == "__main__":
    image = cv2.imread('E:\\barcodes\\data\\05102009108.jpgbarcodeOrig.png')
    found, box = detect_barcode_morp(image)
    angle = box[2]
    pnt1=(int(box[0][0]), int(box[0][1]))
    pnt2=(int(box[1][0]), int(box[1][1]))
    print(angle)
    print("p1 "+str(pnt1)," p2 "+str(pnt2))
    #cv2.drawContours(image, [box], -1, (255, 0, 0), 1)
    cv2.rectangle(image, pnt1, pnt2, color=(0, 255, 0), thickness=2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # draw a bounding box arounded the detected barcode and display the
    # image
    # print(box)
    #cv2.drawContours(image, [box], -1, (255, 0, 0), 1)

