import pylab
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pydicom


ImageFile = pydicom.dcmread('C:/meeting/2D/test/test/S4-A4-0018.dcm')
pylab.imshow(ImageFile.pixel_array,cmap=pylab.cm.bone) 
pylab.show()
cv2.waitKey()
plt.imsave('C:/meeting/2D/test/'+'unadjusted_18.tiff',ImageFile.pixel_array)
