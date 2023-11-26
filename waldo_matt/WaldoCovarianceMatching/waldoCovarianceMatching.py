import math
from skimage import io
import numpy as np
import scipy
import matplotlib.pyplot as plt

field = io.imread("input/smoothed_small_search_field1.png")
field=field[:,:,:3]
r = field.shape[0]
c = field.shape[1]

template = io.imread("input/waldoWblacknoise.png") #wa waldo with blacknoise
#remove alpha channel
template=template[:,:,:3]

template_rows = template.shape[0]
template_cols = template.shape[1]
template_data = []

for tr in range(template_rows):
    for tc in range(template_cols):
        #create feature vector @ each pixel [x, y, R, G, B]
        temp = [tc, tr]
        temp.extend(template[tr,tc])
        template_data.append(temp)

#normalize template_data
template_data = np.array(template_data)
mean = np.mean(template_data, axis=0)
template_data = template_data - mean
#find the template/model covariance matrix
modelCovMatrix = np.matmul(np.transpose(template_data), template_data)/(template_rows*template_cols)


#find the closest match covariance matrix for a corresponding window in the Waldo search puzzle
min_dif = float('inf')
coord = (0,0)
#ansImg contains a difference value of the search field window and the model covariance matrix at that pixel
ansImg = np.zeros((r,c))
difs = []

field_data = np.zeros((r,c,5))
for i in range(r):
    for j in range(c):
        temp = [j,i]
        temp.extend(field[i,j])
        field_data[i,j] = temp

#weight the edges of the template less because its mostly background
window_edge_penalty = np.ones((template_rows,template_cols,1))
window_edge_penalty[:,0:20] = 1/100
window_edge_penalty[:,-20:] = 1/100
window_edge_penalty = window_edge_penalty.reshape(template_rows * template_cols, 1)

#iterate through possible match windows
for fr in range(r-template_rows):
    for fc in range(c-template_cols):
        print(fr,fc)
        window = field_data[fr:fr+template_rows, fc:fc+template_cols]
        window = window.reshape(template_rows*template_cols, 5)
        #fix r, c datapoints
        window[:,0] = window[:,0] - fc
        window[:,1] = window[:,1] - fr

        #normalize pixel data of the window
        window = np.array(window)
        mean = np.mean(window, axis=0)
        window = window - mean
        window = window*window_edge_penalty

        #calculate the correlation matrix
        windowCov = np.matmul(np.transpose(window), window) / (template_rows * template_cols)

        #find the difference between CovModel and CovCand
        gValues, gVectors = scipy.linalg.eig(modelCovMatrix, windowCov)
        # get real parts of complex eigenvalues
        real_parts = np.real(gValues)
        dif = 0
        for ev in gValues:
            if ev == 0:
                continue
            dif += (math.log(abs(ev)))**2
        dif = math.sqrt(dif)
        ansImg[fr, fc] = dif
        if dif<min_dif:
            min_dif = dif
            coord = (fr,fc)

        difs.append((dif,fr,fc))


#find the best match window (smallest difference)
difs.sort(key = lambda x:x[0])
print(difs[0:10])

print("Minumum Cov Difference:", min_dif, " @ (r,c): ", coord)
plt.imshow(ansImg, cmap='gray')
plt.axis("image");
plt.savefig('results/match_bnoise/cov_differences.png');
plt.show()

#print out the ten best matches
for i in range(10):
    dif,r,c = difs[i]
    match = field[r:r+template_rows,c:c+template_cols]
    plt.imshow(match)
    #plt.savefig(f'match_bnoise/match_patch_{i+1}th.png');
    plt.show()


















