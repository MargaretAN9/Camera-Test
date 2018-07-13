


from picamera import PiCamera
from time import sleep
from fractions import Fraction
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


font = cv2.FONT_HERSHEY_SIMPLEX
#camera = PiCamera(resolution=(1280,720 ))
#camera = PiCamera(resolution=(1280, 720),framerate=(Fraction (1,8)),sensor_mode=3)
camera = PiCamera(resolution=(1280, 720),framerate=(.3),sensor_mode=3)

#camera = PiCamera(framerate=(30))



#set wait time should be min 5 sec 

waitime = 4

xrange = 5

print ("test")



#capture seed picture


camera.shutter_speed = 0
camera.iso = 0
sleep(waitime)
camera.capture('/home/pi/Desktop/testexposure.jpg')
print ("test1")
    
img = cv2.imread('/home/pi/Desktop/testexposure.jpg')
#cv2.putText(img,"{0},{1}".format(x,y),(5,50), font, 2, (0,255,0), 9, cv2.LINE_AA)

img = cv2.resize(img, (0, 0), None, .25, .25)
#print ((x),(y),"{0:10s}".format(str(camera.shutter_speed/10000000)),(camera.iso))
#print ((x),(y),"{0:10s}".format(str(round(camera.shutter_speed/10000000))),(camera.iso)) 


times = [.1,1,10,300]


for i in range (1,xrange):
    
    sleep(waitime)
    camera.shutter_speed = int(10000*(times[i-1]))
    print (i,(camera.shutter_speed/1000000))
    camera.iso = 100
    print ("test2")
   
    camera.capture('/home/pi/Desktop/img{0:02d}.jpg'.format(i))
    img1 = cv2.imread('/home/pi/Desktop/img{0:02d}.jpg'.format(i))
    img1 = cv2.resize(img1, (0, 0), None, .25, .25)
    img = np.concatenate((img,img1), axis=1)



#cv2.imwrite("/home/pi/Desktop/img2.jpg", img)



cv2.imshow("img", img)


#cv2.imwrite("/home/pi/Desktop/img", img)


sleep(1)


cv2.waitKey(0)

#cv2.destroyAllWindows()
camera.framerate = 30
sleep(.3)


camera.close()

"""    
cv2.destroyAllWindows()

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




#Displaying a Matplotlib Scale Image
#Imports a file and displays labels (x,y, title} and resolution tick marks
#requires matplotlib




#Enter input file name
filein = "/home/pi/Desktop/img3.jpg"


image = mpimg.imread("/home/pi/Desktop/img3.jpg")
plt.imshow(image)



#if you want  title only
#plt.axis ("off")

plt.xlabel('shutter_speed')
plt.ylabel('iso')


plt.title('Public Lab Raspberry Pi Test')

plt.show()

def readImagesAndTimes():
  
  times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)
  
  filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]

  images = []
  for filename in filenames:
    im = cv2.imread(filename)
    images.append(im)
  
  return images, times

if __name__ == '__main__':
  # Read images and exposure times
  print("Reading images ... ")

  images, times = readImagesAndTimes()
  
"""
