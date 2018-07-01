from picamera import PiCamera
from time import sleep
from fractions import Fraction
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
#camera = PiCamera(resolution=(1280,720 ))
camera = PiCamera(resolution=(1280, 720),framerate=(Fraction (1,6)),sensor_mode=3)
#camera = PiCamera(resolution=(1280, 720),framerate=(1),sensor_mode=3)

#camera = PiCamera(framerate=(30))



#set wait time should be min 5 sec 

waitime = 30

xrange = 11


sleep(waitime)

camera.capture('/home/pi/Desktop/testexposure1.jpg')


 
#ISO values between 0(auto),100,200,300,400,500,600,700,800
#awb gains between 0.0 and 8.0 (typical gains between 0.9 and 1.9)
print("X Y Time(secs) ISO")

for y in range (1,10):

    for x in range (1,xrange):


        if x == 1:
            #capture seed picture
            camera.shutter_speed = 100000*x
            camera.iso = 100*y
            sleep(waitime)
            camera.capture('/home/pi/Desktop/testexposure.jpg')

    
            img = cv2.imread('/home/pi/Desktop/testexposure.jpg')
            cv2.putText(img,"{0},{1}".format(x,y),(5,50), font, 2, (0,255,0), 9, cv2.LINE_AA)

            img = cv2.resize(img, (0, 0), None, .25, .25)
            print ((x),(y),"{0:10s}".format(str(camera.shutter_speed/1000000)),(camera.iso))
 #          print ((x),(y),"{0:10s}".format(str(round(camera.shutter_speed/1000000))),(camera.iso)) 
        else:

            
            camera.shutter_speed = 100000*x
            camera.iso = 100*y
            sleep(waitime)
            camera.capture('img1.jpg')
            img1 = cv2.imread('img1.jpg')
            cv2.putText(img1,"{0},{1}".format(x,y),(5,50), font, 2, (0,255,0), 9, cv2.LINE_AA)
            img1 = cv2.resize(img1, (0, 0), None, .25, .25)
            img = np.concatenate((img, img1), axis=1)
            print ((x),(y),"{0:10s}".format(str(camera.shutter_speed/1000000)),(camera.iso))
     #      print ((x),(y),"{0:10s}".format(str(round(camera.shutter_speed/1000000))),(camera.iso))  
           

    if y==1:
        img3=img
    else:
        
         img3= np.concatenate((img3, img), axis=0)
    


for x in range (1,xrange):
            
    if x == 1:
        #capture seed picture
        camera.shutter_speed = 0
        camera.iso = 0
        sleep(waitime)
        camera.capture('/home/pi/Desktop/testexposure3.jpg')
        print ("test2")
    
        img4 = cv2.imread('/home/pi/Desktop/testexposure3.jpg')
        cv2.putText(img4,"auto",(5,50), font, 2, (0,255,0), 9, cv2.LINE_AA)

        img4 = cv2.resize(img4, (0, 0), None, .25, .25)

    else:
        print ("test")    
        camera.shutter_speed = 0
        camera.iso = 0
        sleep(waitime)
        camera.capture('img5.jpg')
        img5 = cv2.imread('img5.jpg')
        cv2.putText(img5,"auto",(5,50), font, 2, (0,255,0), 9, cv2.LINE_AA)
        img5 = cv2.resize(img5, (0, 0), None, .25, .25)
        img4 = np.concatenate((img4, img5), axis=1)

        print ((x),(y),"{0:10s}".format(str(round(camera.shutter_speed/100000))),(camera.iso))  
            



img3 = np.concatenate((img3, img4), axis=0)



   

#img3 = cv2.resize(img3, (0, 0), None, .25, .25)

cv2.imshow("image2", img3)
cv2.imwrite("/home/pi/Desktop/img3.jpg", img3)
camera.close()
cv2.waitKey(0)
cv2.destroyAllWindows()



#Displaying a Matplotlib Scale Image
#Imports a file and displays labels (x,y, title} and resolution tick marks
#requires matplotlib




#Enter input file name
filein = "/home/pi/Desktop/img3.jpg"


image = mpimg.imread(filein)
plt.imshow(image)



#if you want  title only
#plt.axis ("off")

plt.xlabel('shutter_speed')
plt.ylabel('iso')


plt.title('Public Lab Raspberry Pi Test')

plt.show()


