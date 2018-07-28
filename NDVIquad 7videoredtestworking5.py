
#the program displays (and records) an R/G/B/NDVI quad video.   
#NDVI equations from https://github.com/robintw/RPiNDVI/blob/master/ndvi.py

#store file on desktop

#note you are creating big data files





import time
import numpy as np

import cv2
import picamera
import picamera.array


# Create a VideoCapture object
cap = cv2.VideoCapture(0)




#frame_width = int(cap.get(3))

frame_width = 544
#frame_height = int(cap.get(4))
frame_height= 400

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

#twice height and width

#set video writer (MJPG=avi), (H264=h264) formats

#out = cv2.VideoWriter('/home/pi/Desktop/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1064,800),1)
out = cv2.VideoWriter('/home/pi/Desktop/output.h264',cv2.VideoWriter_fourcc('H','2','6','4'), 10, (1064,800),1)

def disp_multiple(im1=None, im2=None, im3=None, im4=None):
    """
    Combines four images for display.

    """
    height, width = im1.shape

    combined = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)
    combined[0:height, 0:width, :] = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
    combined[height:, 0:width, :] = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    combined[0:height, width:, :] = cv2.cvtColor(im3, cv2.COLOR_GRAY2RGB)
    combined[height:, width:, :] = im4



    return combined


def label(image, text):
    """
    Labels the given image with the given text
    """
    return cv2.putText(image, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),4)


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


def run():
    with picamera.PiCamera() as camera:
        # Set the camera resolution
        x = 400
        camera.resolution = (int(1.33 * x), x)
        # Various optional camera settings below:
        # camera.framerate = 5
        # camera.awb_mode = 'off'
        # camera.awb_gains = (0.5, 0.5)

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

         
                

                # Get the individual colour components of the image
                b, g, r = cv2.split(image)


                #start video capture
                ret, image = cap.read()
                # Calculate the NDVI

                # Bottom of fraction
                bottom = (r.astype(float) + g.astype(float))
                bottom[bottom == 0] = 0.01  # Make sure we don't divide by zero!

                ndvi = (r.astype(float) - g) / bottom
                ndvi = contrast_stretch(ndvi)
                ndvi = ndvi.astype(np.uint8)
                ndvi = cv2.applyColorMap(ndvi, cv2.COLORMAP_JET)

                # Do the labelling
                label(b, 'Blue')
                label(g, 'Green')
                label(r, 'NIR')
                label(ndvi, 'NDVI')


                # Combine ready for display
                combined = disp_multiple(b, g, r, ndvi)

                
  #              write video

                out.write(combined)


                # Display
                cv2.imshow('combined', combined)

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





