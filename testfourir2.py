
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/pi/Desktop/spectrsal star2.jpg',1)
#img = cv2.imread('/home/pi/Desktop/ladder3.jpg',1)
#remove annotation by changing top rows to all black


img[0:60,0:3280] = [0,0,0]

gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




#img = cv2.imread('messi5.jpg',0)

dft = cv2.dft(np.float32(gimg),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()





cv2.imshow('image',gimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

f = np.fft.fft2(gimg)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])






plt.show()

img.close()

"""
