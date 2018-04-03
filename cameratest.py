#test camera program -tested with Raspberrty Pi camera v2.1
#program provides 60 sec alignment preview and records 1920x1080 jpg image on desktop


from picamera import PiCamera
from time import sleep

camera = PiCamera()

camera.start_preview(alpha=255)

camera.start_preview()
sleep(60)
camera.capture('/home/pi/Desktop/testimage1.jpg')

camera.stop_preview()
