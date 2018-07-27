# shows video and captures image using picmaera and opencv
# from https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/ 
#  
# press q to quit 

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np


def nothing(x):
    pass


cv2.namedWindow("RGB")
cv2.createTrackbar ('line#',"RGB",0,479,nothing)


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
    help='Path to video file (if not using camera)')
parser.add_argument('-c', '--color', type=str, default='rgb',
    help='Color space: "gray" (default) or "rgb"')
parser.add_argument('-b', '--bins', type=int, default=640,
    help='Number of bins per channel (default 16)')
parser.add_argument('-w', '--width', type=int, default=0,
    help='Resize video to specified width in pixels (maintains aspect)')
args = vars(parser.parse_args())




#set filename/resolution
#resolution size 4:3 options: (1920,1088),(1640,1232),(640,480)
# note (3280,2464) provides 'out of resources' 

# Configure VideoCapture class instance for using camera or file input.
if not args.get('file', False):
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture(args['file'])

color = args['color']
bins = args['bins']
resizeWidth = args['width']


font = cv2.FONT_HERSHEY_COMPLEX

SIZE = (640,480)
FILEOUT = '/home/pi/Desktop/testimage1.jpg'

 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (SIZE)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(SIZE))
                                    
 
# allow the camera to warmup
time.sleep(0.1)





color = args['color']

bins = args['bins']
resizeWidth = args['width']

# Initialize plot.
fig, ax = plt.subplots()
if color == 'rgb':
    ax.set_title('Line intensity ')
else:
    ax.set_title('Line Intensity(grayscale)')
ax.set_xlabel('line #')
ax.set_ylabel('Intensity')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 3
alpha = 0.5
if color == 'rgb':
    lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha)
    lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha)
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha)
else:
    lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)
ax.set_xlim(0, bins-1)
ax.set_ylim(0, 256)
plt.ion()
plt.show()



 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
    image = frame.array
    image1= image


    
    # Resize frame to width, if specified.
    if resizeWidth > 0:
        (height, width) = image.shape[:2]
        resizeHeight = int(float(resizeWidth / width) * height)
        image = cv2.resize(image, (resizeWidth, resizeHeight),interpolation=cv2.INTER_AREA)
 
    # Normalize histograms based on number of pixels per frame.
    numPixels = np.prod(image.shape[:2])
    if color == 'rgb':
#       print (numPixels)
#        (b, g, r) = cv2.split(image)

       B = image[:,:,0]
       G = image[:,:,1]
       R = image[:,:,2]


       B1 = B
       G1 = G
       R1 = R



       y1=300
       y2=400

       x1=11
       x2=639





  #      pixels = B[y1:y2, x1:x2]

       pixelsB = B1[y1,]
       pixelsG = G1[y1,]
       pixelsR = R1[y1,]
 #       Y = np.arange (y1)
 #       X= np.arange (x1,x2)
 #       X,Y = np.meshgrid(X,Y)



 #       histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
 #       print (histogramR)
 #       histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
#        histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
       lineR.set_ydata(pixelsR)
       lineG.set_ydata(pixelsG)
       lineB.set_ydata(pixelsB)

       y1 =cv2.getTrackbarPos ('line#',"RGB")

        
        
    else:
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       cv2.imshow('Grayscale', gray)
       histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
       lineGray.set_ydata(histogram)

   

    #create trackbar


#    cv2.putText(image1,str(y1), (250,250),font,4,(0,0,255))

    cv2.line(image1,(1,y1),(640,y1),(0,255,0),1)
    cv2.imshow('RGB', image1)

    fig.canvas.draw()
# clear the stream in preparation for the next frame
#press q to quit (several times)
	
    rawCapture.truncate(0)
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
capture.release()
cv2.destroyAllWindows()

"""

	# show the frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
"""






















	

cv2.imwrite(FILEOUT,image)
camera.close()
cv2.waitKey(0)
