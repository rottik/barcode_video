#from find_barcode_image_gradients import detect_barcode
#from find_barcode_image_morphologie import detect_barcode_morp
from find_barcode_video import crop_to_box
from scipy.signal import wiener
from skimage import restoration
from scipy.signal import convolve2d as conv2
import numpy as np
import cv2
from pyzbar.pyzbar import decode


def estimate_PSF(image):
    h1=np.hamming(image.shape[0])
    h2d = h1[:,np.newaxis] * np.hamming(image.shape[1])
    windowed = image * h2d
    spect = np.round(np.fft.fft2(windowed),9)
    # mozna normovat
    cepstrum = np.real(np.fft.fft2(np.log(1+ np.abs(spect)** 2)))
    c0v = cepstrum[0,:]
    #print(c0v)
    R = 0;
    for r in range(1,len(c0v)):
        if (c0v[r] < 0):
            R = r;
            break
    kernel = np.zeros((R,R))
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            if(x*x + y*y < R*R):
                kernel[x,y]=0.5*np.pi*R*R
    return kernel

# hodne mazly
image = cv2.imread('E:\\barcodes\\data\\Foto(751).jpg')
box = [[350,250],[350,600],[820,250],[820,600]]

image = cv2.imread('E:\\barcodes\\data\\tmp583977-noise.png')
box = [[0,0],[0,600],[820,0],[820,600]]
# cv2.imshow("gray",gray)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("gray:"+str(decode(gray)))
barcode = crop_to_box(gray,box)
print("barcode:"+str(decode(barcode)))
cv2.imshow("gray",gray)
barcode = crop_to_box(gray,box)
# hranovy detektory ne - laplacian, sobel

# barcode=cv2.equalizeHist(barcode)

#fil=wiener(barcode,noise=0.5)
#cv2.imshow("scipy.signal.wiener",fil)

cv2.imshow("equalized hist",cv2.equalizeHist(barcode))
print("equal:"+str(decode(cv2.equalizeHist(barcode))))
# dolni propust - odstani sum
#kernel = np.ones((3, 3), np.float32) / 9
#dst = cv2.filter2D(barcode, -1, kernel)
#cv2.imshow("low pass", dst)

psf = np.ones((5, 5)) / 25
# estimate PSF

psf = estimate_PSF(barcode)
#print(psf)

wiener=restoration.wiener(cv2.equalizeHist(barcode), psf, 500)
#wiener=255-wiener
#wiener = np.max(np.max(wiener))*wiener
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
closed = cv2.morphologyEx(wiener, cv2.MORPH_CLOSE, kernel)
#wiener = cv2.equalizeHist(wiener)
#wiener = cv2.dilate(wiener,psf)
blackhat = cv2.morphologyEx(wiener, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("closed", closed)
cv2.imshow("equalized hist",(gray+blackhat))
cv2.imshow("wiener", wiener)
#cv2.imshow("unsupervised_wiener", restoration.unsupervised_wiener(barcode, psf))
#cv2.imshow("richardson_lucy", restoration.richardson_lucy(barcode, psf, 5))
#cv2.imshow("unwrap_phase", restoration.unwrap_phase(barcode))
#cv2.imshow("denoise_tv_chambolle", restoration.denoise_tv_chambolle(barcode))
#cv2.imshow("denoise_bilateral", restoration.denoise_bilateral(barcode,multichannel=False))
#cv2.imshow("wavelet", restoration.denoise_wavelet(barcode))
#cv2.imshow("nl_means", restoration.denoise_nl_means(barcode,multichannel=False))

# horni propust - zvyrazni zmeny
# kernel = np.array([[-2,-1, 6, -1, -2], [-2,-1, 6, -1, -2], [-2,-1, 7, -1, -2], [-2,-1, 6, -1, -2], [-2,-1, 6, -1, -2]])
#kernel = np.array([[-1, 3, -1], [-1, 4, -1], [-1, 3, -1]])
#im = cv2.filter2D(dst, -1, kernel)
#cv2.imshow("low+high pass",im)

cv2.waitKey(0)
cv2.destroyAllWindows()