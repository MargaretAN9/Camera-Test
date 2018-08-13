# shows video and captures image using picmaera and opencv
# from https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/ 
#  
# press q to quit 
#436nm HSV H:115-146, S:98-225  V:251:255
#546nm HSV H:44-74, S:0-255  V:228:255




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



y1=300
y2=400

x1=11
x2=639



def throttle_ocr(image):
    img = image[0:480,0:640]
	# lower and upper ranges for green pixels, format BGR
    lower = np.array([44,0,228])
    upper = np.array([74,255,255])
    resmask = cv2.inRange(img,lower,upper)

    V=resmask[y1,:]
 #   print(len(V),V)
    A=len(V)-1
    while A>=0:
        if V[A]!=0:
            print (V[A])
            return V[A]
        A-=1
        
    return 0    
        


#        print (resmask.shape)

    
#	count = np.count_nonzero(res)
#	return count 



#def throttle_ocr(image,coords):
##	img = images[coords[1]:coords[3],coords[0]:coords[2]]
    # lower and upper ranges for green pixels, format BGR
#	lower = np.array([0,110,0])
#	upper = np.array([90,200,90])
#	res = cv2.inRange(img,lower,upper)
#	count = np.count_nonzero(res)
#	return count 














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
fig, ax1 = plt.subplots()
if color == 'rgb':
    ax1.set_title('Line intensity ')
else:
    ax1.set_title('Line Intensity(grayscale)')
ax1.set_xlabel('line #')
ax1.set_ylabel('Intensity')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 3
alpha = 0.5
if color == 'rgb':
    lineR, = ax1.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha)
    lineG, = ax1.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha)
    lineB, = ax1.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha)
else:
    lineGray, = ax1.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)
ax1.set_xlim(0, bins-1)
ax1.set_ylim(0, 256)


ax2=ax1.twiny()

#newlabel = [273.15,290,310,330,350,373.15] # labels of the xticklabels: the position in the new x-axis
#k2degc = lambda t: t-273.15 # convert function: from Kelvin to Degree Celsius
#newpos   = [k2degc(t) for t in newlabel]   # position of the xticklabels in the old x-axis
#ax2.set_xticks(newpos)
#ax2.set_xticklabels(newlabel)

ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
ax2.spines['bottom'].set_position(('outward', 36))
ax2.set_xlabel('Spectrum (nm)')
#ax2.set_xlim(ax1.get_xlim())
ax2.set_xlim(100,bins-1)







fig.tight_layout()




plt.ion()


plt.show()



 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
    image = frame.array
    image1= image
    HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    

    y1 =cv2.getTrackbarPos ('line#',"RGB")
    
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


       throttle_ocr(HSV) 
        
        
    else:
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       cv2.imshow('Grayscale', gray)
       histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
       lineGray.set_ydata(histogram)

   

    #create trackbar


#    cv2.putText(image1,str(y1), (250,250),font,4,(0,0,255))
    fig.canvas.draw()
    cv2.line(image1,(1,y1),(640,y1),(0,255,0),1)
    cv2.imshow('RGB', image)


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









