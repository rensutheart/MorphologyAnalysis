import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd

def labeledImageToStack(labeledImage):
    outputStack = []

    for i in range(0, np.max(labeledImage)):
        outputStack.append((labeledImage == i) + 0)

    return outputStack


labeledImage = io.imread("C:\\RESEARCH\\MEL\\MEL_Output\\labelsF1.tiff")

labeledImageStack = labeledImageToStack(labeledImage)

plt.imshow(labeledImageStack[0][:,:,0])
plt.show()


originalImage = io.imread("C:\\RESEARCH\\MEL\\Pre-processed\\Con001.tif - T=0.tif")

import pandas as pd
eventInfo = pd.read_csv("C:\\RESEARCH\\MEL\\MEL_Output\\EventLocations1.csv")
for event in eventInfo.iterrows():
    print(event[1]["Loc_X"])


stackF1_3 = np.stack((originalImage,) * 3, axis=-1)


eventInfoPaths = ["C:\\RESEARCH\\MEL\\MEL_Output\\EventLocations1.csv"]
eventInfoPaths[0][eventInfoPaths[0].rfind("\\")+1:]