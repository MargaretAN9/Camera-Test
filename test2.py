#test camera program -tested with Raspberrty Pi camera v2.1
#program provides xx sec alignment preview and records jpg image 
# application: align spectrometer or focus microscope 
# annotates with filename and datetime

from picamera import PiCamera,Color
from time import sleep
import datetime as dt

#set filename/resolution/video time (sec)
#resolution size 4:3 options: (3280,2464),(1920,1080),(1640,1232),(640,480)


filename = '/home/pi/Desktop/testimage1.jpg'
SIZE = (3280,2464)
vidtime = 7.9

camera = PiCamera()

camera.start_preview(alpha=255)

#camera.annotate_background = picamera.Color('black')
camera.annotate_background = Color('blue')
camera.annotate_foreground = Color('yellow')
camera.annotate_text = filename + "     " + dt.datetime.now().strftime ('%Y-%m-%d %H:%M:%S')

# Now fix the values
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = g

iso = camera.iso
awbmode = camera.awb_mode
speed = camera.shutter_speed
resolution = camera.resolution
framerate = camera.framerate
awbgains = camera.awb_gains


print (vidtime)
print ({iso:.3f}, speed,resolution, framerate,awbmode,awbgains,g)














#camera.start_preview()
sleep(vidtime)
 

camera.resolution = (SIZE)
print (iso, speed,resolution, framerate,awbmode,awbgains)
camera.capture(filename)
print (iso, speed,resolution, framerate,awbmode,awbgains)
camera.stop_preview()

