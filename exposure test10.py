from picamera import PiCamera
from time import sleep
from fractions import Fraction
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
#camera = PiCamera(resolution=(1280,720 ))
camera = PiCamera(resolution=(1280, 720),framerate=(1),sensor_mode=3)
#camera = PiCamera(framerate=(30))

#capture standard picture

camera.capture('/home/pi/Desktop/testexposure.jpg')

img = cv2.imread('/home/pi/Desktop/testexposure.jpg')
img = cv2.resize(img, (0, 0), None, .25, .25)

topLeft = cv2.resize(img, (0, 0), None, .25, .25)




#capture loop settings 

   
# 10 sec is max for V2, 


for x in range (1,4):
    camera.shutter_speed = 100000*x

    for y in range (1,3):
        camera.iso = 100*y


    
        camera.capture('img1.jpg')
        img1 = cv2.imread('img1.jpg')
        cv2.putText(img1,"{0}".format(x),(5,50), font, 2, (0,255,0), 7, cv2.LINE_AA)
        img1 = cv2.resize(img1, (0, 0), None, .25, .25)
        img = np.concatenate((img, img1), axis=1)
        row1=img
#    img = np.concatenate((img, img1), axis=1)


        print ((x),(y),(camera.shutter_speed/100000),(camera.iso))  

   

        
 #   img=img2        


   


cv2.imshow("image2", img)
#cv2.imwrite("/home/pi/Desktop/vis.jpg", vis)
camera.close()
cv2.waitKey(0)
cv2.destroyAllWindows()


