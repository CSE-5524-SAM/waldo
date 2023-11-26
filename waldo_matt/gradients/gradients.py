import math
from skimage import io
import skimage
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

template = io.imread("inputs/waldoW129noise.png") #waldo with grey
#remove alpha channel
template=template[:,:,:3]

template_rows = template.shape[0]
template_cols = template.shape[1]
template_data = []

#make template greyscale
Im = skimage.color.rgb2gray(template)

#create derivative masks and convolve on template
gx_mask = np.array([-1,0,1]).reshape((1,3))
gy_mask = np.transpose(gx_mask)

gxIm = scipy.ndimage.convolve(Im, gx_mask, mode='nearest')
gyIm = scipy.ndimage.convolve(Im, gy_mask, mode='nearest')

# plt.imshow(gxIm, cmap="grey")
# plt.show()
#
# plt.imshow(gyIm, cmap="grey")
# plt.show()

#find the gradient histogram for template image
template_gradients = [0,0,0,0,0,0,0,0]
for r in range(template_rows):
    for c in range(template_cols):
        dx = gxIm[r,c]
        dy = gyIm[r,c]
        mag = math.sqrt(dx ** 2 + dy**2)
        if mag>0:
            dir = np.arctan2(dy, dx)
            deg = math.degrees(dir)
            #shift gradients clockwise 45/2 degrees
            deg -= (45 + 45/2)
            #make degrees positive
            deg += 360
            deg = deg % 360
            template_gradients[math.floor(deg/45)] += mag
#normalize gradient histogram
template_gradients = template_gradients / np.linalg.norm(template_gradients)


#read field image
field = io.imread("inputs/smoothed_field_small.png") #wa waldo with blacknoise
#remove alpha channel
field=field[:,:,:3]

field_rows = field.shape[0]
field_cols = field.shape[1]

#find data for the image field used to find the gradients
field_grey = skimage.color.rgb2gray(field)
#convolve derivative masks
field_gx = scipy.ndimage.convolve(field_grey, gx_mask, mode='nearest')
field_gy = scipy.ndimage.convolve(field_grey, gy_mask, mode='nearest')
#find magnitude of gradient at each pixel
field_mag = (field_gx**2 + field_gy**2)**.5
#find dir of gradient at each pixel
field_dir = np.degrees(np.arctan2(field_gx, field_gy))
#rotate
field_deg = (field_dir - (45 + 45/2) + 360) % 360
#find bin index for each pixel
field_gradient_histogram_bin = np.floor(field_deg/45).astype(int)

#at each pixel of search field image, provide 8 bins for gradients
field_grad_histograms = np.zeros((field_rows,field_cols,8))
#for each pixel, place the maginitude of the gradient in the corrsponding histogram bin
for r in range(field_rows):
    for c in range(field_cols):
        field_grad_histograms[r,c, field_gradient_histogram_bin[r,c]] = field_mag[r,c]

#iterate through all possible windows in search field
gradients_list = []
for r in range(field_rows - template_rows):
    for c in range(field_cols - template_cols):
        print(r,c)
        #sum the gradient histograms for each pixel over the entire window to get histogram of gradients for window
        gradients = np.sum(field_grad_histograms[r:r+template_rows, c:c+template_cols], axis=(0,1))
        #print(gradients)
        #normalize histogram gradient vector
        gradients = gradients / np.linalg.norm(gradients)

        #find the euclidean distance between template and window gradient histograms
        dif = np.sum((template_gradients - gradients)**2)**.5
        gradients_list.append((dif, r, c))

#find closest matching gradient histogram
gradients_list.sort(key=lambda x: x[0])
best_dif, bestr, bestc = gradients_list[0]

#
field_match = field[bestr:bestr+template_rows, bestc:bestc+template_cols]
plt.imshow(field_match)
plt.savefig("/results/gradient_best_match.png")
plt.show()
