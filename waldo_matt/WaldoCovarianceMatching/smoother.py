import matplotlib.pyplot as plt
from skimage import io, filters

field = io.imread("input/small_search_field1.png")
r = field.shape[0]
c = field.shape[1]

sigma = 1.0  # Adjust the standard deviation of the Gaussian kernel
smoothed_image = filters.gaussian(field, sigma=sigma)
plt.imshow(smoothed_image)
plt.axis("image");
plt.imsave('input/smoothed_small_search_field1.png', smoothed_image);
plt.show()