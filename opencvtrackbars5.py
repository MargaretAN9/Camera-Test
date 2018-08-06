#Trackbar Picamera manual control

#ESC to quit



import time
import numpy as np
import cv2
import picamera
import picamera.array
from fractions import Fraction

def nothing(x):
    pass


ISO_number = [100,200,320,400,500,640,800]
exposure_number = ['auto','off','night', 'nightpreview', 'backlight','spotlight', 'sports','snow','beach','verylong','fixedfps','antishake','fireworks']
effect_number = ['none','negative','solarize', 'sketch', 'denoise','emboss', 'oilpaint','hatch','gpen','pastel','watercolor','film','blur']


font=cv2.FONT_HERSHEY_SIMPLEX

cv2.namedWindow("Trackbars")

cv2.namedWindow("Public Lab")
cv2.createTrackbar ('Red Gain',"Trackbars",0,80,nothing)
cv2.createTrackbar ('Blue Gain',"Trackbars",0,80,nothing)
cv2.createTrackbar ('Frame Rate',"Trackbars",5,60,nothing)
cv2.createTrackbar ('Contrast',"Trackbars",0,200,nothing)
cv2.createTrackbar  ('Brightness',"Trackbars",0,100,nothing)
cv2.createTrackbar  ('ISO',"Trackbars",0,6,nothing)
cv2.createTrackbar  ('Exposure',"Trackbars",0,11,nothing)
cv2.createTrackbar  ('Saturation',"Trackbars",0,200,nothing)
cv2.createTrackbar  ('Sharpness',"Trackbars",0,200,nothing)
cv2.createTrackbar  ('Effects',"Trackbars",0,12,nothing)


# Create a VideoCapture object
cap = cv2.VideoCapture(0)

width = 544
#frame_height = int(cap.get(4))
height= 400


#frame_width = int(cap.get(3))

frame_width = 544
#frame_height = int(cap.get(4))
frame_height= 400

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

#twice height and width

#set video writer MPEG4 =XVID
out = cv2.VideoWriter('/home/pi/Desktop/NDVItest20.mp4',cv2.VideoWriter_fourcc('X','V','I','D'), 10, (1088,800),1)
#set video writer (MJPG=avi) option
#out = cv2.VideoWriter('/home/pi/Desktop/NDVItestwithtrackbar.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1088,800),1)
#set video writer H264 option
#out = cv2.VideoWriter('/home/pi/Desktop/output.h264',cv2.VideoWriter_fourcc('H','2','6','4'), 10, (1088,800),1)



def label(image, text):
    """
    Labels the given image with the given text
    """
    return cv2.putText(image, text, (0, 50), font, 2, (255,255,255),4)


def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-95%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out






#fastie colormap
def fastieColorMap(ndvi) :

    fastie = np.zeros((256, 1, 3), dtype=np.uint8)
    fastie[:, 0, 2] = [255, 250, 246, 242, 238, 233, 229, 225, 221, 216, 212, 208, 204, 200, 195, 191, 187, 183, 178, 174, 170, 166, 161, 157, 153, 149, 145, 140, 136, 132, 128, 123, 119, 115, 111, 106, 102, 98, 94, 90, 85, 81, 77, 73, 68, 64, 60, 56, 52, 56, 60, 64, 68, 73, 77, 81, 85, 90, 94, 98, 102, 106, 111, 115, 119, 123, 128, 132, 136, 140, 145, 149, 153, 157, 161, 166, 170, 174, 178, 183, 187, 191, 195, 200, 204, 208, 212, 216, 221, 225, 229, 233, 238, 242, 246, 250, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 151, 146, 141, 136, 131, 126, 121, 116, 111, 106, 101, 96, 91, 86, 81, 76, 71, 66, 61, 56, 66, 77, 87, 98, 108, 119, 129, 140, 131, 122, 113, 105, 96, 87, 78, 70, 61, 52, 43, 35, 26, 17, 8, 0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
    fastie[:, 0, 1] = [255, 250, 246, 242, 238, 233, 229, 225, 221, 216, 212, 208, 204, 200, 195, 191, 187, 183, 178, 174, 170, 166, 161, 157, 153, 149, 145, 140, 136, 132, 128, 123, 119, 115, 111, 106, 102, 98, 94, 90, 85, 81, 77, 73, 68, 64, 60, 56, 52, 56, 60, 64, 68, 73, 77, 81, 85, 90, 94, 98, 102, 106, 111, 115, 119, 123, 128, 132, 136, 140, 145, 149, 153, 157, 161, 166, 170, 174, 178, 183, 187, 191, 195, 200, 204, 208, 212, 216, 221, 225, 229, 233, 238, 242, 246, 250, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 151, 146, 141, 136, 131, 126, 121, 116, 111, 106, 101, 96, 91, 86, 81, 76, 71, 66, 61, 56, 66, 77, 87, 98, 108, 119, 129, 140, 147, 154, 161, 168, 175, 183, 190, 197, 204, 211, 219, 226, 233, 240, 247, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 249, 244, 239, 233, 228, 223, 217, 212, 207, 201, 196, 191, 185, 180, 175, 170, 164, 159, 154, 148, 143, 138, 132, 127, 122, 116, 111, 106, 100, 95, 90, 85, 79, 74, 69, 63, 58, 53, 47, 42, 37, 31, 26, 21, 15, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fastie[:, 0, 0] = [255, 250, 246, 242, 238, 233, 229, 225, 221, 216, 212, 208, 204, 200, 195, 191, 187, 183, 178, 174, 170, 166, 161, 157, 153, 149, 145, 140, 136, 132, 128, 123, 119, 115, 111, 106, 102, 98, 94, 90, 85, 81, 77, 73, 68, 64, 60, 56, 52, 56, 60, 64, 68, 73, 77, 81, 85, 90, 94, 98, 102, 106, 111, 115, 119, 123, 128, 132, 136, 140, 145, 149, 153, 157, 161, 166, 170, 174, 178, 183, 187, 191, 195, 200, 204, 208, 212, 216, 221, 225, 229, 233, 238, 242, 246, 250, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 151, 146, 141, 136, 131, 126, 121, 116, 111, 106, 101, 96, 91, 86, 81, 76, 71, 66, 61, 56, 80, 105, 130, 155, 180, 205, 230, 255, 239, 223, 207, 191, 175, 159, 143, 127, 111, 95, 79, 63, 47, 31, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239]


    color = cv2.LUT(ndvi, fastie)

  
    return color;

#begin camera collection

def run():
    with picamera.PiCamera() as camera:

        # Set the camera parameters
        x = 400
 #       camera.resolution = (int(1.33 * x), x)
        camera.resolution = (480, x) 
        # Various optional camera settings below:
        camera.iso=800
        camera.framerate = 30
        camera.awb_mode = 'off'
        camera.exposure_mode = "off"
  #      camera.framerate = Fraction (1,6)
     #red/blue camera ratios from 0 to 8

        
 #       camera.awb_gains = (Red_gain,Blue_gain)

        # Need to sleep to give the camera time to get set up properly
        time.sleep(1)

        with picamera.array.PiRGBArray(camera) as stream:
            # Loop constantly
            while True:
                # Grab data from the camera, in colour format
                # NOTE: This comes in BGR rather than RGB, which is important
                # for later!
                camera.capture(stream, format='bgr', use_video_port=True)
                image = stream.array

                image1=image 
                Red_gain =cv2.getTrackbarPos ("Red Gain","Trackbars")  
                Blue_gain =cv2.getTrackbarPos ("Blue Gain","Trackbars")
                Frame_rate =cv2.getTrackbarPos ("Frame Rate","Trackbars")
                Contrast =cv2.getTrackbarPos ('Contrast',"Trackbars")   
                Brightness =cv2.getTrackbarPos ('Brightness',"Trackbars")
                ISO =cv2.getTrackbarPos ('ISO',"Trackbars")
                Exp=cv2.getTrackbarPos ('Exposure',"Trackbars")
                Saturation=cv2.getTrackbarPos ('Saturation',"Trackbars")
                Sharpness=cv2.getTrackbarPos ('Sharpness',"Trackbars")
                Effects=cv2.getTrackbarPos ('Effects',"Trackbars")

                
                print(ISO,"Is the ISO")
                
  #              print (Frame_rate)
                
                camera.awb_gains = (Red_gain/10,Blue_gain/10)
                camera.framerate = Frame_rate
                camera.contrast = Contrast-100
                camera.brightness = Brightness
                camera.exposure_mode = exposure_number[Exp]
                camera.saturation = Saturation-100
                camera.sharpness = Sharpness-100
                camera.image_effect = effect_number[Effects]
 #               camera.iso = ISO_number[Exp]
                print (camera.iso)

                #start video capture
                ret, image = cap.read()
                # Calculate the NDVI


         

    
                
#                ndvifastie = fastieColorMap(ndvi)

#               #format red 

 #               zeros = np.zeros (r.shape[:2], dtype ="uint8")
  #              r= cv2.merge(([zeros,zeros,r]))

                # Do the labelling
                label(image1, 'RGB')







                # Combine ready for display
 #               combined = disp_multiple(image1,ndvifastie,r, ndvijet)



 #               write video


 #               cv2.putText(combined,str(Red_gain/10),(750,50),font,2,(0,0,256),4)
 #               cv2.putText(combined,str(Blue_gain/10),(200,50),font,2,(256,0,0),4)
  #              out.write(combined)


                # Display
                cv2.imshow('Public Lab', image1)

                stream.truncate(0)

                #  press ESC to break 
                c = cv2.waitKey(7) % 0x100
                if c == 27:
                    break

    # cleanup or things will get messy
    cv2.destroyAllWindows()
    cap.release()
    out.release()

if __name__ == '__main__':
    run()


"""       


# colorbar fastie

                rows,cols,channels = colorbar.shape

                roi = colorbar[0:rows, 0:cols ]
                                
                # Now create a mask of logo and create its inverse mask also
                img2gray = cv2.cvtColor(colorbar,cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                # Now black-out the area of logo in ROI
                img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

                
                # Take only region of logo from logo image.
                img2_fg = cv2.bitwise_and(colorbar,colorbar,mask = mask)
                
                
                # Put logo in ROI and modify the main image
                dst = cv2.add(img1_bg,img2_fg)

             
                combined[775:(775+rows), 25:(25+cols)] = dst


# colorbar jet

                rows,cols,channels = colorbarjet.shape

                roi = colorbarjet[0:rows, 0:cols ]
                                
                # Now create a mask of logo and create its inverse mask also
                img2gray = cv2.cvtColor(colorbarjet,cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                # Now black-out the area of logo in ROI
                img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

                
                # Take only region of logo from logo image.
                img2_fg = cv2.bitwise_and(colorbarjet,colorbarjet,mask = mask)
                
                
                # Put logo in ROI and modify the main image
                dst = cv2.add(img1_bg,img2_fg)

             
                combined[775:(775+rows), 720:(720+cols)] = dst

"""



