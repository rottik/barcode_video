import numpy as np
import cv2
import zxing
from find_barcode_image_gradients import detect_barcode
from find_barcode_image_morphologie import detect_barcode_morp
from pyzbar.pyzbar import decode
from os import listdir
from scipy.signal import wiener
from skimage import restoration
from os.path import isfile, join

def euklid_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

def get_angle(box):
    if(type(box)==int):
        return 0
    # najdi pevny bod ve vesmiru
    point = [999999,999999]
    point_index = 0
    tmp_box = box
    min_dist = euklid_dist([0,0], point)
    for p in tmp_box:
        ed = euklid_dist([0,0], point)
        if ed < min_dist:
            min_dist = ed
            point = p

    dists = np.zeros((4,1))
    index = 0
    for p in tmp_box:
        dists[index] = euklid_dist(point, p)
        index += 1

    idiag = np.argmax(dists)
    diag = tmp_box[idiag]
    np.delete(tmp_box,idiag)
    ilong = np.argmax(dists)
    long = tmp_box[ilong]
    np.delete(tmp_box,ilong)
    ishort = np.argmax(dists)
    short = tmp_box[ishort]

    angle = np.arccos(np.array(point) - np.array(short)) * 180 / np.pi
    #if angle > 180:
    #    angle -= 180
    return angle

def crop_to_box(frame,box):
    min_x = np.min([x[0] for x in box])
    max_x = np.max([x[0] for x in box])
    min_y = np.min([x[1] for x in box])
    max_y = np.max([x[1] for x in box])

    if max_y-min_y <= 0:
        return frame
    if max_x-min_y <= 0:
        return frame

    if max_x > min_x and max_y > min_y:
        barcode = frame[min_y : max_y , min_x : max_x ]
        return barcode
    return frame

def decode_from_box(frame,box):
    code_zxing = ""
    code_zbar = ""
    barcode = crop_to_box(frame,box)
    barcode = cv2.cvtColor(barcode, cv2.COLOR_BGR2GRAY)

    try:
        cv2.imwrite("tmp.png", barcode)
        zxing_code = str(reader.decode("tmp.png").parsed)
        code_zxing = str(zxing_code)
    except:
        code_zxing = ""

    try:
        zbarcode = decode(barcode)
        if (len(zbarcode) > 0):
            code_zbar = str(zbarcode[0][0])
    except:
        code_zbar = ""

# scikit wiener
    #wie=wiener(barcode,noise=0.01)
    #cv2.imshow("wiener",wie)

# skimage wiener
    #psf = np.ones((3, 3)) / 9
    #deconv = restoration.wiener(barcode, psf,1)
    #cv2.imshow("wiener filtered", deconv)

    barcode = cv2.equalizeHist(barcode)

    # dolni propust - odstani sum
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(barcode, -1, kernel)

    # horni propust - zvyrazni zmeny
    #kernel = np.array([[-2,-1, 6, -1, -2], [-2,-1, 6, -1, -2], [-2,-1, 7, -1, -2], [-2,-1, 6, -1, -2], [-2,-1, 6, -1, -2]])
    kernel = np.array([[-1, 3,-1],[-1,4,-1],[-1,3,-1]])
    im = cv2.filter2D(dst, -1, kernel)

    #im=cv2.erode(im,np.ones((3,3)))
    #im=cv2.dilate(im,np.ones((3,3)))
    #im=cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)*255
    im = cv2.equalizeHist(im)
    try:
        if im.shape[0] > 0 and im.shape[1] > 0:
            cv2.imshow("sharp",cv2.resize(im,(300, 200), interpolation = cv2.INTER_CUBIC))
    except AttributeError:
        print("att error")

    code_enh_zxing = ""
    code_enh_zbar = ""
    try:
        cv2.imwrite("tmp.png", im)
        #zxing_code = str(reader.decode("tmp.png").parsed)
        code_enh_zxing = str(zxing_code)
    except:
        code_enh_zxing = ""

    try:
        zbarcode = decode(im)
        if (len(zbarcode) > 0):
            code_enh_zbar = str(zbarcode[0][0])
    except:
        code_enh_zbar = ""

    return [code_zxing, code_zbar, code_enh_zxing, code_enh_zbar]

def insert_data(frame, angle, zxingline, zbarline, cnt_zbar, cnt_zxing, cnt_enh_zbar, cnt_enh_zxing):
    if zxingline != "":
        codel1 = "ZXING:" + zxingline
    else:
        codel1 = "ZXING:---"

    if zbarline != "":
        codel2 = "ZBAR:" + zbarline
    else:
        codel2 = "ZBAR:---"

    textsize = [0, 0]
    font_face = cv2.FONT_HERSHEY_DUPLEX
    textsize2 = cv2.getTextSize(codel2, font_face, 2, 1)
    textsize1 = cv2.getTextSize(codel1, font_face, 2, 1)
    textsize[0] = np.max([textsize1[0][0], textsize2[0][0]])
    textsize[1] = textsize1[0][1] + textsize2[0][1]
    cv2.rectangle(frame, (5, 5), (textsize[0] + 50, 110), color=(255, 255, 255), thickness=-2)
    cv2.putText(frame, codel1, (30, 50), font_face, fontScale=1.8, color=(0, 0, 0))
    cv2.putText(frame, codel2, (30, 100), font_face, fontScale=1.8, color=(0, 0, 0))
    # cv2.putText(frame, "Angle:" + str(angle), (30, 150), font_face, fontScale=1.8, color=(0, 0, 0))
    cv2.putText(frame, "ZXing count :" +str(cnt_zxing), (30, 200), font_face, fontScale=1.8, color=(0, 0, 0))
    cv2.putText(frame, "ZBar count:" + str(cnt_zbar), (30, 250), font_face, fontScale=1.8, color=(0, 0, 0))
    cv2.putText(frame, "Filtred ZXing count :" + str(cnt_enh_zxing), (30, 300), font_face, fontScale=1.8, color=(0, 0, 0))
    cv2.putText(frame, "Filtred ZBar count:" + str(cnt_enh_zbar), (30, 350), font_face, fontScale=1.8, color=(0, 0, 0))
    return frame

if(__name__=="__main__"):
    reader = zxing.BarCodeReader()
    mypath = 'E:\\barcodes\\video_data\\'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    frame_width = 1920
    frame_height = 1080
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter("output.avi", fourcc, 20.0, (int(frame_width), int(frame_height)))
    for f in onlyfiles:
        print(f)
        camera = cv2.VideoCapture(mypath + f)

        cnt = 0
        cnt_zxing=0
        cnt_zbar=0
        cnt_enh_zxing=0
        cnt_enh_zbar=0
        zxing_code = ""
        zbar_code = ""
        prev_barcode = np.zeros((5, 5))
        while True:
            (grabbed, frame) = camera.read()
            if not grabbed:
                break
            found, box = detect_barcode(frame)
            angle=0
            if found:
                [codel1, codel2, code_enh_l1, code_enh_l2] = decode_from_box(frame, box)

            if codel1 != "":
                cnt_zxing += 1
                if zxing_code == "":
                    zxing_code = codel1
            if codel2 != "":
                cnt_zbar += 1
                if zbar_code == "":
                    zbar_code = codel2

            if code_enh_l1 != "":
                cnt_zxing += 1

            if code_enh_l2 != "":
                cnt_enh_zbar += 1

            frame = insert_data(frame, angle, codel1, codel2, cnt_zbar, cnt_zxing, cnt_enh_zbar, cnt_enh_zxing)
            # out.write(frame)

            if type(box) != int:
                if len(box) > 0:
                    cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
            cv2.imshow("Video", frame)
            cnt += 1

            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break
        camera.release()
        print("ZXing:" + str(cnt_zxing) + "\t" + str((100 * cnt_zxing) / cnt) + "\t" + zxing_code)
        print("ZBar :" + str(cnt_zbar) + "\t" + str((100 * cnt_zbar) / cnt) + "\t" + zbar_code)
        print("ZXing:" + str(cnt_enh_zxing) + "\t" + str((100 * cnt_zxing) / cnt) + "\t" + zxing_code)
        print("ZBar :" + str(cnt_enh_zbar) + "\t" + str((100 * cnt_zbar) / cnt) + "\t" + zbar_code)
    # out.release()
    cv2.destroyAllWindows()
