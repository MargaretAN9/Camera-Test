#The program  records a video  from a Raspberry Pi camera

# use omxplayer to read video




from picamera import PiCamera
from time import sleep
from picamera import PiCamera, Color
from picamera import PiCamera, Color
import datetime as dt

# set parameters
FILENAME = '/home/pi/videodemo.h264'

camera = PiCamera()

camera.annotate_background = Color('blue')
camera.annotate_foreground = Color('yellow')
sleep(1)
camera.start_preview()

camera.start_recording(FILENAME)


camera.annotate_text = " Public Lab Microscope Test" 
sleep(5)



#   camera.contrast = 5



camera.stop_recording()    
camera.stop_preview()




