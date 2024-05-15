
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skimage.transform import rescale, rotate
from scipy.misc import imresize

from time import time

from skimage import data, io

import os
from os import listdir
from os.path import isfile, join

import matplotlib
# Say, "the default sans-serif font is Arial"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"



folder = 2

# NEW
#path = "D:\\PhD\\MEL_Output_new\\ReOutput_June2019\\AverageOutput\\{}\\".format(folder)
#imagePath = "D:\\PhD\\ForPaperNewSample\\{}\\Preproc\\".format(folder)
#outputPath =  "D:\\PhD\\MEL_Output_new\\ReOutput_June2019\\AverageOutput\\{}\\Panel\\".format(folder)

# OLD
#path = "D:\\PhD\\MEL_Output\\ReOutput_OLD_June2019\\AverageOutput\\{}\\".format(folder)
#imagePath = "D:\\PhD\\ForPaperOldSamples\\{}\\Preproc\\".format(folder)
#outputPath =  "D:\\PhD\\MEL_Output\\ReOutput_OLD_June2019\\AverageOutput\\{}\\Panel\\".format(folder)


#imagesPaths = [(imagePath+f) for f in listdir(imagePath) if isfile(join(imagePath, f)) and f.endswith("tif")]
#imagesPathsOverlaid = [(path+f) for f in listdir(path) if isfile(join(path, f)) and f.endswith("tif")]\

# NEW
#num = folder
#structurePath = "D:\\PhD\\ForPaperNewSample\\{}-structure1\\".format(num)
#outputPath = "D:\\PhD\MEL_Output_new\\ReOutput_June2019_reduced\\AverageOutput\\{}-structure1\\Panel\\".format(num)
#fusePath = "D:\\PhD\MEL_Output_new\\ReOutput_June2019_reduced\\AverageOutput\\{}-structure1\\Fusion\\".format(num)
#fragPath = "D:\\PhD\MEL_Output_new\\ReOutput_June2019_reduced\\AverageOutput\\{}-structure1\\Fission\\".format(num)
#depPath = "D:\\PhD\MEL_Output_new\\ReOutput_June2019_reduced\\AverageOutput\\{}-structure1\\Depolarization\\".format(num)
#overlaidPath = "D:\\PhD\MEL_Output_new\\ReOutput_June2019_reduced\\AverageOutput\\{}-structure1\\".format(num)

#OLD
num = folder
structurePath = "D:\\PhD\\ForPaperOldSamples\\{}-structure1\\".format(num)
outputPath = "D:\\PhD\MEL_Output\\ReOutput_OLD_June2019\\AverageOutput\\{}-structure1\\Panel\\".format(num)
fusePath = "D:\\PhD\MEL_Output\\ReOutput_OLD_June2019\\AverageOutput\\{}-structure1\\Fusion\\".format(num)
fragPath = "D:\\PhD\MEL_Output\\ReOutput_OLD_June2019\\AverageOutput\\{}-structure1\\Fission\\".format(num)
depPath = "D:\\PhD\MEL_Output\\ReOutput_OLD_June2019\\AverageOutput\\{}-structure1\\Depolarization\\".format(num)
overlaidPath = "D:\\PhD\MEL_Output\\ReOutput_OLD_June2019\\AverageOutput\\{}-structure1\\".format(num)

structurePaths = [(structurePath+f) for f in listdir(structurePath) if isfile(join(structurePath, f)) and f.endswith("tif")]
fusePaths = [(fusePath+f) for f in listdir(fusePath) if isfile(join(fusePath, f)) and f.endswith("tif")]
fragPaths = [(fragPath+f) for f in listdir(fragPath) if isfile(join(fragPath, f)) and f.endswith("tif")]
depPaths = [(depPath+f) for f in listdir(depPath) if isfile(join(depPath, f)) and f.endswith("tif")]
overlaidPaths = [(overlaidPath+f) for f in listdir(overlaidPath) if isfile(join(overlaidPath, f)) and f.endswith("tif")]

if not os.path.exists(outputPath):
    os.makedirs(outputPath)  
    
def rotateStack(inStack, angle):
    outStack = []
    for im in inStack:
#        temp = []
#        for i in range(0, 3):
#            temp.append(rotate(im[:,:,i],270, resize=True))
#    
#        temp = np.array(temp)
#        print(temp.shape)
#        outStack.append(np.moveaxis(temp, 0, -1))
        outStack.append(rotate(im,angle, resize=True))
    return np.array(outStack)
    
outputSum = None
for i in range(0, len(fusePaths)):
    print("Processing frame:", i)
    
    Frame1 = io.imread(structurePaths[i])
    Frame2 = io.imread(structurePaths[i+1])
    overlaid = io.imread(overlaidPaths[i])
    fuse = io.imread(fusePaths[i])
    frag = io.imread(fragPaths[i])
    dep = io.imread(depPaths[i])
    
    #pad the images with white to make them stand out
    padWidth = 2
    Frame1Padded = rotateStack(np.pad(Frame1, pad_width=((0,0),(padWidth,padWidth), (padWidth,padWidth), (0,0)), mode='constant', constant_values=255), 0)
    Frame2Padded = rotateStack(np.pad(Frame2, pad_width=((0,0),(padWidth,padWidth), (padWidth,padWidth), (0,0)), mode='constant', constant_values=255), 0)
    overlaidPadded = rotateStack(np.pad(overlaid, pad_width=((0,0),(padWidth,padWidth), (padWidth,padWidth), (0,0)), mode='constant', constant_values=255), 0)
    
#    fig =  plt.figure(figsize=(Frame1.shape[1]/100, Frame1.shape[2]/100), dpi=100)
#    (ax1, ax2, ax3) = fig.subplots(1,3,sharey=True)
#    
#    ax1.imshow(Frame1[0])
#    ax2.imshow(Frame2[0])
#    ax3.imshow(overlaid[0])
    

    frameText = str(i)
    if(i < 10):
        frameText = "0" + frameText
        
    output = np.dstack((Frame1, fuse, frag, dep, overlaid, Frame2))
    io.imsave("{}{}.tif".format(outputPath, frameText), (output).astype(np.uint8))   
    
    output2  = np.dstack((Frame1Padded, Frame2Padded, overlaidPadded))
    if(i > 0):
        outputSum = np.hstack((outputSum, output2))
    else:
        outputSum = output2

formattedOutput = (outputSum*255).astype(np.uint8)
io.imsave("{}Total{}.tif".format(outputPath, frameText), formattedOutput)      
