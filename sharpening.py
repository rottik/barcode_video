#from find_barcode_image_gradients import detect_barcode
#from find_barcode_image_morphologie import detect_barcode_morp
from find_barcode_video import crop_to_box
from scipy.signal import wiener
from skimage import restoration
from scipy.signal import convolve2d as conv2
import numpy as np
import cv2

image = cv2.imread('E:\\barcodes\\data\\Foto(751).jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray",gray)
box = [[350,250],[350,600],[820,250],[820,600]]
barcode = crop_to_box(gray,box)
cv2.imshow("barcode",barcode)

# barcode=cv2.equalizeHist(barcode)

fil=wiener(barcode,noise=0.5)
cv2.imshow("scipy.signal.wiener",fil)

cv2.imshow("equalized hist",cv2.equalizeHist(barcode))

# dolni propust - odstani sum
kernel = np.ones((3, 3), np.float32) / 9
dst = cv2.filter2D(barcode, -1, kernel)
cv2.imshow("low pass", dst)

psf = np.ones((5, 5)) / 25
cv2.imshow("wiener", restoration.wiener(barcode, psf, 1100))
#cv2.imshow("unsupervised_wiener", restoration.unsupervised_wiener(barcode, psf))
cv2.imshow("richardson_lucy", restoration.richardson_lucy(barcode, psf, 5))
cv2.imshow("unwrap_phase", restoration.unwrap_phase(barcode))
cv2.imshow("denoise_tv_chambolle", restoration.denoise_tv_chambolle(barcode))
cv2.imshow("denoise_bilateral", restoration.denoise_bilateral(barcode,multichannel=False))
cv2.imshow("wavelet", restoration.denoise_wavelet(barcode))
cv2.imshow("nl_means", restoration.denoise_nl_means(barcode,multichannel=False))

# horni propust - zvyrazni zmeny
# kernel = np.array([[-2,-1, 6, -1, -2], [-2,-1, 6, -1, -2], [-2,-1, 7, -1, -2], [-2,-1, 6, -1, -2], [-2,-1, 6, -1, -2]])
kernel = np.array([[-1, 3, -1], [-1, 4, -1], [-1, 3, -1]])
im = cv2.filter2D(dst, -1, kernel)
cv2.imshow("low+high pass",im)

cv2.waitKey(0)
cv2.destroyAllWindows()