import numpy as np
import cv2
import zxing
from find_barcode_image_gradients import detect_barcode
from pyzbar.pyzbar import decode
from os import listdir
from os.path import isfile, join

mypath = 'E:\\barcodes\\video_data\\'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

frame_width = 1920
frame_height = 1080
fourcc = cv2.VideoWriter_fourcc(*'XVID')
for f in onlyfiles:
    print(f)
    camera = cv2.VideoCapture(mypath + f)
    output = cv2.VideoWriter(mypath + f+".avi", fourcc, 20.0, (640, 480))

    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        output.write(frame)

    camera.release()
    output.release()
cv2.destroyAllWindows()
