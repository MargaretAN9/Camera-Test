import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

# Loading exposure images into a list
img_fn = ["img01.jpg", "img02.jpg", "img03.jpg", "img04.jpg"]
img_list = [cv2.imread(fn) for fn in img_fn]
exposure_times = np.array([0.0001, 0.001, 0.01, 0.1], dtype=np.float32)
print ("test")







# Merge exposures to HDR image
merge_debvec = cv2.createMergeDebevec()
hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy())
merge_robertson = cv2.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())
print ("test1")

# Tonemap HDR image
tonemap1 = cv2.createTonemapDurand(gamma=2.2)
res_debvec = tonemap1.process(hdr_debvec.copy())
tonemap2 = cv2.createTonemapDurand(gamma=1.3)
res_robertson = tonemap2.process(hdr_robertson.copy())
print ("test2")


# Exposure fusion using Mertens
merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

print ("test3")


# Convert datatype to 8-bit and save
res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

cv2.imwrite("/home/pi/Desktop/ldr_debvec.jpg", res_debvec_8bit)
cv2.imwrite("/home/pi/Desktop/ldr_robertson.jpg", res_robertson_8bit)
cv2.imwrite("/home/pi/Desktop/fusion_mertens.jpg", res_mertens_8bit)

# Estimate camera response function (CRF)
cal_debvec = cv2.createCalibrateDebevec()
crf_debvec = cal_debvec.process(img_list, times=exposure_times)
hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy(), response=crf_debvec.copy())
cal_robertson = cv2.createCalibrateRobertson()
crf_robertson = cal_robertson.process(img_list, times=exposure_times)
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy(), response=crf_robertson.copy())
imgplot = plt.imshow(crf_robertson)

# Obtain Camera Response Function (CRF)
calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(img_list, exposure_times)
#imgplot = plt.imshow(responseDebevec)
#plt.imshow(responseDebevec)
#plt.show()

dimensions=responseDebevec.shape
print (dimensions)
#creates (256 1 3) array 



"""
#cv2.imwrite("/home/pi/Desktop/responseDebevec", responseDebevec)
plt.figure(figsize=(10,10))
plt.plot(responseDebevec,range(256), 'rx')
#plt.plot(gg,range(256), 'gx')
#plt.plot(gb,range(256), 'bx')
plt.ylabel('pixel value Z')
plt.xlabel('log exposure X')
plt.show()


cv2.imwrite("/home/pi/Desktop/hdr_debvec.jpg", hdr_debvec)
cv2.imwrite("/home/pi/Desktop/hdr_robertson.jpg", hdr_robertson)
cv2.imwrite("/home/pi/Desktop/crf_robertson.jpg", crf_robertson)
cv2.imwrite("/home/pi/Desktop/crf_debvec.jpg", crf_debvec)







"""
