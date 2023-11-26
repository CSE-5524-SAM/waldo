import math
import random
from skimage import io
import numpy as np
import scipy
import matplotlib.pyplot as plt

def generate_random_values():
    return [1-random.randint(0, 255)/2 for _ in range(3)]

template = io.imread("input/waldo_body.png")
#remove alpha channel
template=template[:,:,:3]

template_rows = template.shape[0]
template_cols = template.shape[1]
template_data = []

for i in range(template_rows):
    #start from left
    for c in range(template_cols):
        iswhite = template[i, c][0] > 235 and template[i, c][1] > 235 and template[i, c][2]
        if iswhite:
            template[i,c] = [255/2,255/2,255/2]  #generate_random_values()
        else:
            break
    #start from right
    for c in range(template_cols-1, -1, -1):
        iswhite = template[i, c][0] > 235 and template[i, c][1] > 235 and template[i, c][2]
        if iswhite:
            template[i, c] = [255/2,255/2,255/2]  #generate_random_values()
        else:
            break

plt.imshow(template)
plt.axis("image")
io.imsave("input/waldoW129noise.png", template)
plt.show()