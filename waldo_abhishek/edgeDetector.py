from pylab import imshow, show
import matplotlib as plt
import mahotas
import mahotas.demos
import numpy as np

# loading the image
wally = mahotas.demos.load('wally')

# getting float type value 
# float values are better to use
wfloat = wally.astype(float)
 
# splitting image into red, green and blue channel
r, g, b = wfloat.transpose((2, 0, 1))
 
# white channel
w = wfloat.mean(2)
 
# pattern of wally shirt
# pattern + 1, +1, -1, -1 on vertical axis
pattern = np.ones((24, 16), float)
for i in range(2):
    pattern[i::4] = -1
 
# convolve with the red minus white
# increase the response where shirt is
v = mahotas.convolve(r-w, pattern)
 
# getting maximum value 
mask = (v == v.max())
 
# creating mask to tone down the image 
# except the region where wally is
mask = mahotas.dilate(mask, np.ones((48, 24)))
 
# subtraction mask from the wally
np.subtract(wally, .8 * wally * ~mask[:, :, None], 
                   out = wally, casting ='unsafe')

# show the new image
imshow(wally)
show()