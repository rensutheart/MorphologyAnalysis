import ImageAnalysis
import numpy as np

from skimage import data, io
from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_isodata, threshold_mean, gaussian, apply_hysteresis_threshold,  unsharp_mask
from skimage.exposure import rescale_intensity

from skimage.exposure import histogram

import pandas as pd

import matplotlib.pyplot as plt

img = ImageAnalysis.loadGenericImage("testIm.tiff")
bins = 256

# ImageAnalysis.plotStackHist(img)

counts, centers = histogram(img, nbins=bins)
#remove 'black'
counts = counts[1:]
centers = centers[1:]

movingAverageFrame=10
df = pd.DataFrame(counts)
movingAverage = df.rolling(movingAverageFrame, center=True).mean()[0]


# normalise
counts = counts/np.sum(counts)
movingAverage = movingAverage/np.sum(movingAverage)

# first derivitive
gradient = []
gradient2 = []
for i in range(0,len(movingAverage)-1):
    gradient.append(movingAverage[i+1] - movingAverage[i])

for i in range(0,len(counts)-1):
    gradient2.append(counts[i+1] - counts[i])

plt.plot(gradient2, label='raw')
plt.plot(gradient, label='ma')
plt.show()


lambda_val = 0.00001

max_location = np.argmax(counts)

counts2 = []
# TODO: this assumes that counts start at 0
for i in range(0, counts.shape[0]):
    counts2.append(counts[i] + lambda_val * (np.abs(max_location - i))/bins)
counts2 = np.array(counts2)

# plt.plot(counts, label="initial")
# plt.plot(counts2, label="updated")
# plt.show()

updated_gradient = []

for i in range(0,len(counts2)-1):
    updated_gradient.append(counts2[i+1] - counts2[i])

# plt.plot(gradient2, label='raw')
# plt.plot(updated_gradient, label='updated_gradient')
# plt.show()

threshold_location = np.argmax(np.array(gradient2)>0)

binary_image = img[0] > threshold_location

plt.imshow(binary_image)
plt.show()



thresh = threshold_otsu(img)
bin_otsu = img[0] > thresh

plt.imshow(bin_otsu)
plt.show()