'''
Install: (indented should install automatically)
pip install tensorflow
	pip install numpy
pip install scikit-image
	pip install scipy
	pip install pillow
	pip install tifffile
	pip install matplotlib
pip install pandas
pip install trimesh
pip install czifile
pip install lxml
pip install ExifRead
pip install opencv-python
pip install pyglet
pip install glooey

'''

from scipy.stats import multivariate_normal
from scipy import signal
from skimage.transform import rescale
import ImageAnalysis
import Morphology

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import SliceViewer

from os import listdir, path, makedirs
from os.path import isfile, join


from skimage import measure
from scipy.spatial import ConvexHull
from scipy.ndimage import zoom
import pandas as pd
import csv
import math

from skimage import io

import trimesh

import pyglet
import glooey
import GUI


saveResults = True
personReviewName = "Rensu"

outputPath = "C:\\RESEARCH\\MEL\\James_MEL_Data\\Validated(MEL2)\\"
filePath = "C:\\RESEARCH\\MEL\\James_MEL_Data\\Output\\Con001\\"

labeledImageTimelapsePaths = [filePath + "LabelsF1_0.tif", filePath + "LabelsF2_0.tif"]
originalImageTimelapsePaths = ["C:\\RESEARCH\\MEL\\James_MEL_Data\\\decovolutionFrames\\Con001_1_frame_1_05_06.tif","C:\\RESEARCH\\MEL\\James_MEL_Data\\\decovolutionFrames\\Con001_1_frame_2_05_06.tif"]
eventInfoPaths = [filePath + "0.csv"]

# outputPath = "C:\\RESEARCH\\MEL\\Validated\\"
# filePath = "C:\\RESEARCH\\MEL\\MEL_Output\\"

# labeledImageTimelapsePaths = [filePath + "labelsF1.tiff", filePath + "labelsF2.tiff"]
# originalImageTimelapsePaths = ["C:\\RESEARCH\\MEL\\Pre-processed\\Con001.tif - T=0.tif","C:\\RESEARCH\\MEL\\Pre-processed\\Con001.tif - T=1.tif"]
# eventInfoPaths = [filePath + "EventLocations1.csv"]

# labeledImageTimelapsePaths = [f for f in listdir(filePath) if isfile(join(filePath, f)) and (f.endswith("czi"))]


for i in range(0, len(labeledImageTimelapsePaths)):
    print(i, ' ', labeledImageTimelapsePaths[i])

startFileIndex = 0
startFrame = 0
print('Start file: ', labeledImageTimelapsePaths[startFileIndex])

#0,1,2,7,8,
#11, 12
rerunIndices = [0] #,15,16] #0,1,2,3,4,5,6,7,8,9,10,11,
# rerunIndices = [18,19,20,21,22,23,24] # 11, 8,9,10,17,14,15,0,1,2,3,4,5,6,7,12,13


# 0 unasissgned, 1 Nothing, 2 Fuse, 3 Fragment, 4 depolarize, 5 Frag-Fuse
class transType:
    UNASSIGNED = 0
    NOTHING = 1
    FUSE = 2
    FRAGMENT = 3
    DEPOLARIZE = 4
    UNCERTAIN = 5

def generateKernel(kernelSize=9, size=10, s=0.5, showKernel=False):
    # zDivFactor=2
    kernelZ = 1  # kernelSize//zDivFactor

    x, y = np.mgrid[-1.0:1.0:2 / size, -1.0:1.0:2 / size]

    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([x.flat, y.flat])

    mu = np.array([0.0, 0.0])

    sigma = np.array([s, s])
    covariance = np.diag(sigma ** 2)

    out = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    zOut = signal.gaussian(kernelZ, kernelZ * 0.2)
    zOut = zOut / zOut[kernelZ // 2]
    if showKernel:
        plt.figure()
        plt.plot(zOut)

    # Reshape back to a (30, 30) grid.
    out = out.reshape(x.shape)
    kernel = rescale(out, kernelSize / size, multichannel=False)
    normalizationNum = kernel[kernelSize // 2, kernelSize // 2]
    kernel = kernel / normalizationNum

    fullKernel = []

    for i in range(0, kernelZ):
        iKernel = kernel * zOut[i]
        fullKernel.append(iKernel)
        if showKernel:
            plt.imshow(iKernel, vmin=0, vmax=1)
            plt.show()

    return np.array(fullKernel)

def generateRGBkernels(kernelSize = 10):

    splat_kernel = generateKernel(kernelSize, size=kernelSize, s=0.4)
    kernel_zeros = np.zeros_like(splat_kernel)
    red_kernel = np.swapaxes(np.stack((splat_kernel, kernel_zeros, kernel_zeros))[:, 0, :, :], 0, 2)
    green_kernel = np.swapaxes(np.stack((kernel_zeros, splat_kernel, kernel_zeros))[:, 0, :, :], 0, 2)
    blue_kernel = np.swapaxes(np.stack((kernel_zeros, kernel_zeros, splat_kernel))[:, 0, :, :], 0, 2)
    # print(red_kernel.shape)

    # 3 layers (*0.5 since each one repeats twice, and overlays)
    red_kernel_3D = np.swapaxes(np.stack((red_kernel * 0.2, red_kernel * 0.5, red_kernel * 0.2), axis=-1),2,3)
    green_kernel_3D = np.swapaxes(np.stack((green_kernel * 0.2, green_kernel * 0.5, green_kernel * 0.2), axis=-1),2,3)
    blue_kernel_3D = np.swapaxes(np.stack((blue_kernel * 0.2, blue_kernel * 0.5, blue_kernel * 0.2), axis=-1),2,3)

    # print(red_kernel_3D.shape)

    return (red_kernel_3D, green_kernel_3D, blue_kernel_3D)


def generateGradientImageFromLocation(originalStack, eventInfoReceived):
    (red_kernel_3D, green_kernel_3D, blue_kernel_3D) = generateRGBkernels(20)
    kernels = [green_kernel_3D, red_kernel_3D, blue_kernel_3D]
    outputStack = np.stack((originalStack,originalStack,originalStack), axis=-1)

    for event in eventInfoReceived.iterrows():
            location = [int(event[1]["Loc_X"]), int(event[1]["Loc_Y"]), int(event[1]["Loc_Z"])]
            thisLocation = [location[1], location[0], location[2]]
            outputStack, success = addKernelToOutputStack(kernels[int(event[1]["EventType"])], outputStack, thisLocation, originalStack)

    return outputStack


def checkEvent(typeEvent, label_1, label_2, assocInOtherFrame, location, xyScale=3, zScale=2, windowSize=100, binarize=True):
    global labelsF1_stack
    global labelsF2_stack
    global originalF1
    global originalF2
    
    stack4D_F1 = labelsF1_stack
    stack4D_F2 = labelsF2_stack

    sc = trimesh.Scene()

    thisLocation = location.copy()

    highlighedLabels = np.zeros((originalF1.shape[2], windowSize, windowSize*3, 3))
    # (x, y, depth, RGB)
    stackF1_3 = np.stack((originalF1,) * 3, axis=-1)
    stackF2_3 = np.stack((originalF2,) * 3, axis=-1)

    (red_kernel_3D, green_kernel_3D, blue_kernel_3D) = generateRGBkernels(10)
    #print(associatedLabelInSame)
    #print(location)
    l1 = int(label_1)
    l2 = int(label_2)
    assocInOtherFrame = int(assocInOtherFrame)
    if typeEvent == transType.FUSE:
        sphere = trimesh.creation.icosphere(1, 1/zScale, [0.0, 1.0, 0.0, 1.0])
        sphere.apply_scale([zScale, zScale, zScale])
        thisLocation = [thisLocation[1]*xyScale, thisLocation[0]*xyScale, thisLocation[2]*zScale]
        print('this location ' + str(thisLocation))
        print('location ' + str(location))
        sphere.apply_translation(thisLocation)
        sc.add_geometry(sphere)

        # This rescaling is necessary since some times pixels are directly next to each other and then the 3D mesh has discontinuities
        label1 = ImageAnalysis.rescaleStackXY_depthLast(np.int8(stack4D_F1[l1]), xyScale)
        label1 = (label1>np.min(label1))*1
        labelMesh1 = Morphology.fullStackToMesh(label1, [1, 1, zScale, 1])
        labelMesh1.visual.face_colors = [0.0, 0.5, 0.0, 0.5]
        sc.add_geometry(labelMesh1)

        label2 = ImageAnalysis.rescaleStackXY_depthLast(np.int8(stack4D_F1[l2]), xyScale)
        label2 = (label2>np.min(label2))*1
        labelMesh2 = Morphology.fullStackToMesh(label2, [1, 1, zScale, 1])
        labelMesh2.visual.face_colors = [0.5, 0.0, 0.0, 0.5]
        sc.add_geometry(labelMesh2)

        # Get the structure in the other frame
        combined = stack4D_F2[assocInOtherFrame]

        # structureList = np.intersect1d(associatedFromF1_to_F2[l1], associatedFromF1_to_F2[l2])
        # #print(structureList)
        # combined = np.zeros_like(stack4D_F2[0])
        # for structure in structureList:
        #     combined = np.maximum(combined, stack4D_F2[structure])

        combined = ImageAnalysis.rescaleStackXY_depthLast(np.int8(combined), xyScale)
        combined = (combined>np.min(combined))*1
        otherFrameMesh = Morphology.fullStackToMesh(combined, [1, 1, zScale, 1])
        otherFrameMesh.visual.face_colors = [0.0, 0.0, 0.0, 0.4]
        sc.add_geometry(otherFrameMesh)

        xCoord = int(location[1])
        yCoord = int(location[0])
        zCoord = int(location[2])

        #overlay the structures with a faint green
        structures = np.float64(stack4D_F1[l1])
        structures = np.stack((structures,) * 3, axis=-1)
        structures[:,:,:,0] = structures[:,:,:,2] = 0
        structures[:,:,:,1] = structures[:,:,:,1]*np.max(stackF1_3)*0.6

        structures2 = np.float64(stack4D_F1[l2])
        structures2 = np.stack((structures2,) * 3, axis=-1)
        structures2[:,:,:,1] = structures2[:,:,:,2] = 0
        structures2[:,:,:,0] = structures2[:,:,:,0]*np.max(stackF1_3)*0.6        

        labelOverlaid = np.maximum(stackF1_3*0.5, np.maximum(structures, structures2))

        labelOverlaid, success = addKernelToOutputStack(green_kernel_3D, labelOverlaid, [location[1], location[0], location[2]], stackF1_3) 
        # plt.imshow(np.max(labelOverlaid, axis=2))
        # plt.show()
        highlighedLabels[:,0:100, 0:100] = ImageAnalysis.padStackXY_depthLast(labelOverlaid, windowSize // 2)[:,xCoord:xCoord+windowSize,yCoord:yCoord+windowSize]
        highlighedLabels[:, 0:100, 100:200] = ImageAnalysis.padStackXY_depthLast(stackF1_3, windowSize // 2)[:,xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
        highlighedLabels[:, 0:100, 200:300] = ImageAnalysis.padStackXY_depthLast(stackF2_3, windowSize // 2)[:,xCoord:xCoord+windowSize,yCoord:yCoord+windowSize]
    elif typeEvent == transType.FRAGMENT:
        sphere = trimesh.creation.icosphere(1, 1 / zScale, [1.0, 0.0, 0.0, 1.0])
        sphere.apply_scale([zScale, zScale, zScale])
        thisLocation = [thisLocation[1]*xyScale, thisLocation[0]*xyScale, thisLocation[2]*zScale]
        print('location ' + str(thisLocation))
        sphere.apply_translation(thisLocation)
        sc.add_geometry(sphere)

        label1 = ImageAnalysis.rescaleStackXY_depthLast(np.int8(stack4D_F2[l1]), xyScale)
        label1 = (label1>np.min(label1))*1
        labelMesh1 = Morphology.fullStackToMesh(label1, [1, 1, zScale, 1])
        labelMesh1.visual.face_colors = [0.5, 0.0, 0.0, 0.5]
        sc.add_geometry(labelMesh1)

        label2 = ImageAnalysis.rescaleStackXY_depthLast(np.int8(stack4D_F2[l2]), xyScale)
        label2 = (label2>np.min(label2))*1
        labelMesh2 = Morphology.fullStackToMesh(label2, [1, 1, zScale, 1])
        labelMesh2.visual.face_colors = [0.0, 0.5, 0.0, 0.5]
        sc.add_geometry(labelMesh2)

        combined = stack4D_F1[assocInOtherFrame]
        # structureList = np.intersect1d(associatedFromF2_to_F1[l1], associatedFromF2_to_F1[l2])
        # # print(structureList)
        # combined = np.zeros_like(stack4D_F1[0])
        # for structure in structureList:
        #     combined = np.maximum(combined, stack4D_F1[structure])

        combined = ImageAnalysis.rescaleStackXY_depthLast(np.int8(combined), xyScale)
        combined = (combined>np.min(combined))*1
        otherFrameMesh = Morphology.fullStackToMesh(combined, [1, 1, zScale, 1])
        otherFrameMesh.visual.face_colors = [0.0, 0.0, 0.0, 0.4]
        sc.add_geometry(otherFrameMesh)

        xCoord = int(location[1])
        yCoord = int(location[0])
        zCoord = int(location[2])

        # overload the structures with a faint red
        structures = np.float64(stack4D_F2[l1])
        structures = np.stack((structures,) * 3, axis=-1)
        structures[:,:,:,1] = structures[:,:,:,2] = 0
        structures[:,:,:,0] = structures[:,:,:,0]*np.max(stackF2_3)*0.6

        structures2 = np.float64(stack4D_F2[l2])
        structures2 = np.stack((structures2,) * 3, axis=-1)
        structures2[:,:,:,0] = structures2[:,:,:,2] = 0
        structures2[:,:,:,1] = structures2[:,:,:,1]*np.max(stackF2_3)*0.6

        labelOverlaid = np.maximum(stackF2_3*0.5, np.maximum(structures, structures2))

        labelOverlaid, success = addKernelToOutputStack(red_kernel_3D, labelOverlaid,  [location[1], location[0], location[2]], stackF2_3)
        highlighedLabels[:, 0:100, 0:100] = ImageAnalysis.padStackXY_depthLast(labelOverlaid, windowSize // 2)[:,
                                            xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
        highlighedLabels[:, 0:100, 100:200] = ImageAnalysis.padStackXY_depthLast(stackF2_3, windowSize // 2)[:,
                                              xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
        highlighedLabels[:, 0:100, 200:300] = ImageAnalysis.padStackXY_depthLast(stackF1_3, windowSize // 2)[:,
                                              xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
    elif typeEvent == transType.DEPOLARIZE:
        sphere = trimesh.creation.icosphere(1, 1 / zScale, [0.0, 0.0, 1.0, 1.0])
        sphere.apply_scale([zScale, zScale, zScale])
        thisLocation = [thisLocation[1]*xyScale, thisLocation[0]*xyScale, thisLocation[2]*zScale]
        print('location ' + str(thisLocation))
        sphere.apply_translation(thisLocation)
        sc.add_geometry(sphere)

        label1 = ImageAnalysis.rescaleStackXY_depthLast(np.int8(stack4D_F1[l1]), xyScale)
        label1 = (label1>np.min(label1))*1
        labelMesh1 = Morphology.fullStackToMesh(label1, [1, 1, zScale, 1])
        labelMesh1.visual.face_colors = [0.0, 0.0, 0.5, 0.5]
        sc.add_geometry(labelMesh1)

        # I add a second sphere, since sometimes the first one is not rendered.
        sphere2 = trimesh.creation.icosphere(1, 1 / zScale, [0.0, 0.0, 1.0, 1.0])
        sphere2.apply_scale([zScale, zScale, zScale])
        sphere2.apply_translation(thisLocation)
        sc.add_geometry(sphere2)

        # structureList = associatedFromF1_to_F2[l1]
        # if(structureList.shape[0] != 0):
        #     print("There are not supposed to be associated structures")


        xCoord = int(location[1])
        yCoord = int(location[0])
        zCoord = int(location[2])

        # overload the structures with a faint blue
        structures = np.float64(stack4D_F1[l1])
        structures = np.stack((structures,) * 3, axis=-1)
        structures[:, :, :, 0] = structures[:, :, :, 1] = 0
        structures[:, :, :, 2] = structures[:, :, :, 2] * np.max(stackF1_3) * 0.6

        labelOverlaid = np.maximum(stackF1_3*0.5, structures)

        labelOverlaid, success = addKernelToOutputStack(blue_kernel_3D, labelOverlaid,  [location[1], location[0], location[2]], stackF1_3)
        highlighedLabels[:, 0:100, 0:100] = ImageAnalysis.padStackXY_depthLast(labelOverlaid, windowSize // 2)[:,
                                            xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
        highlighedLabels[:, 0:100, 100:200] = ImageAnalysis.padStackXY_depthLast(stackF1_3, windowSize // 2)[:,
                                              xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
        highlighedLabels[:, 0:100, 200:300] = ImageAnalysis.padStackXY_depthLast(stackF2_3, windowSize // 2)[:,
                                              xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
    else:
        print("The type must be either FUSE, FRAGMENT, or DEPOLARISE")
        return


    from scipy.spatial.transform import Rotation as R
    rotation = [0, 90, 0]
    rotationMatrix = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    transformationMatrix = np.eye(4, 4)
    transformationMatrix[0:3, 0:3] = rotationMatrix
    sc.apply_transform(transformationMatrix)

    return highlighedLabels, sc



def addKernelToOutputStack(kernelIn, outputStack, location, Frame1):
    kernel = kernelIn.copy()
    if(np.max(outputStack) > 1):
        kernel *= 255
        kernel = kernel.astype(np.uint8)
        outputStack = outputStack.astype(np.uint8)
    else:
        kernel = kernel.astype(np.float32)
        outputStack = outputStack.astype(np.float32)
    
    kernelSize = kernel.shape[1]
    kernelDepth = kernel.shape[2]
    location = np.around(location).astype(np.uint16)

    # ensure I stay within the bounds of the image
    startX = location[0] - kernelSize // 2
    endX = location[0] + kernelSize // 2 + kernelSize % 2  # for odd numbers
    startY = location[1] - kernelSize // 2
    endY = location[1] + kernelSize // 2 + kernelSize % 2  # for odd numbers
    startZ = location[2] - kernelDepth // 2
    endZ = location[2] + kernelDepth // 2 + kernelDepth % 2  # for odd numbers

    kernelStartX = max(0, -startX)  # if negative then remove that part
    kernelEndX = min(0, (Frame1.shape[0]) - endX) + kernelSize
    kernelStartY = max(0, -startY)
    kernelEndY = min(0, (Frame1.shape[1]) - endY) + kernelSize
    kernelStartZ = max(0, -startZ)
    kernelEndZ = min(0, (Frame1.shape[2]) - endZ) + kernelDepth

    startX = max(startX, 0)
    endX = min(endX, Frame1.shape[0])
    startY = max(startY, 0)
    endY = min(endY, Frame1.shape[1])
    startZ = max(startZ, 0)
    endZ = min(endZ, Frame1.shape[2])

    try:
        outputStack[startX:endX, startY:endY, startZ:endZ] = np.maximum(outputStack[startX:endX, startY:endY, startZ:endZ], kernel[kernelStartX:kernelEndX, kernelStartY:kernelEndY, kernelStartZ:kernelEndZ])
    except Exception as e:
        print(e)
        print("ERROR in addKernelToOutputStack. Sizes:")
        print("{} {}   {} {}   {} {} Kernel: {} {}   {} {}   {} {}".format(startX, endX, startY, endY, startZ, endZ,
                                                                           kernelStartX, kernelEndX, kernelStartY,
                                                                           kernelEndY, kernelStartZ, kernelEndZ))
        print(location)
        #        print("Error at {} to {}".format(label_index, withinAssociatedLabelsF1[label_index][0]))
        return outputStack, False

    return outputStack, True



def nextLabel():
    global eventInfo
    global label
    global sceneWidget
    global imageWidget
    global displayHBox
    global scaledStack
    global currentFrame  
    global fuseFragDep

    label += 1
    if(label >= len(eventInfo.index)):
        print("End of images")
        window.close()

    try:
        thisEvent = eventInfo.iloc[label]
    except:
        print("Could not read from eventInfo, might have reached end of list. label = " + str(label))
        return

   
    
    fuseFragDep = int(thisEvent["EventType"])
    print("fuseFragDep", fuseFragDep)
    print("label", label)

    
    if True:
        
        typeEvent = transType.NOTHING
        if fuseFragDep == 0:
            typeEvent = transType.FUSE
        elif fuseFragDep == 1:
            typeEvent = transType.FRAGMENT
        elif fuseFragDep == 2:
            typeEvent = transType.DEPOLARIZE

        location3D = [int(thisEvent["Loc_X"]), int(thisEvent["Loc_Y"]), int(thisEvent["Loc_Z"])]

        highlighedLabels, scene = checkEvent(typeEvent, thisEvent["Label_1"], thisEvent["Label_2"], thisEvent["AssocInOtherFrame"], location3D, 1)

        # SliceViewer.multi_slice_viewer(highlighedLabels)
        # scene.show()
        scaledStack = np.uint8(ImageAnalysis.contrastStretch(ImageAnalysis.rescaleStackXY_RGB(highlighedLabels, 2.5), 0, 100) * 255)
        # scaledStack = ImageAnalysis.binarizeStack(ImageAnalysis.rescaleStackXY(ImageAnalysis.preprocess(highlighedLabels), scaleFactor=2.5))
        currentFrame = 0

        displayHBox.remove(sceneWidget)
        sceneWidget = SceneWidget(scene)
        displayHBox.add_right(sceneWidget)
        imageWidget.update_image(scaledStack[currentFrame])

        #hbox.remove(infoLabel)
        text = "Fusion: {}/{}  Fission: {}/{}  Depolarisation: {}/{} Other: {}".format(
        fusionCorrect,len(eventInfo.loc[eventInfo['EventType'] == 0].index), fissionCorrect, len(eventInfo.loc[eventInfo['EventType'] == 1].index), depolarisationCorrect, len(eventInfo.loc[eventInfo['EventType'] == 2].index),otherIncorrect)
        print(text)
        #infoLabel = GUI.MyLabel(text)
        #hbox.add_right(infoLabel)

    # TODO Left over code?
    elif fuseFragDep + 1 < 3:
        fuseFragDep += 1
        label = -1
        currentFrame = 0

        nextLabel()



def incrementYes(widget=None):
    global fuseFragDep
    global label
    global fusionCorrect
    global fissionCorrect
    global depolarisationCorrect
    yesList.append((fuseFragDep, label))
    if(fuseFragDep == 0): #Fusion
        fusionCorrect += 1
    elif(fuseFragDep == 1): #Fission
        fissionCorrect += 1
    elif(fuseFragDep == 2): #Depolarisation
        depolarisationCorrect += 1
    nextLabel()
    print("yes")

def incrementNo(widget=None):
    global fuseFragDep
    global label
    global otherIncorrect
    noList.append((fuseFragDep, label))
    removeEventsIndices.append(label)
    otherIncorrect += 1
    nextLabel()
    print("no")

def incrementUnclear(widget=None):
    global fuseFragDep
    global label
    global otherIncorrect
    unclearList.append((fuseFragDep, label))
    otherIncorrect += 1
    nextLabel()
    print("unclear")

def incrementThreshold(widget=None):
    global fuseFragDep
    global label
    global otherIncorrect
    thresholdList.append((fuseFragDep, label))
    removeEventsIndices.append(label)
    otherIncorrect += 1
    nextLabel()
    print("threshold")


#############################



def labeledImageToStack(labeledImage):
    outputStack = []

    for i in range(0, np.max(labeledImage)):
        outputStack.append((labeledImage == i) + 0)

    return outputStack




# START HERE

# for fileNameIndex in range(startFileIndex, len(labeledImageTimelapsePaths)):
for fileNameIndex in rerunIndices:
#for fileNameIndex in range(startFileIndex, startFileIndex + 1):
    fileName = labeledImageTimelapsePaths[fileNameIndex]
    print("Processing: " + fileName)

    #for timeIndex in range(startFrame, cziData.sizeT - 1): # -1 since frame 1 and frame 2 pair
    for timeIndex in range(0, 1):
        print("Time index: " + str(timeIndex) + ' for ' + fileName)

        frame1Index = timeIndex
        frame2Index = timeIndex + 1

        frame1IndexText = str(frame1Index)
        if(frame1Index < 10):
            frame1IndexText = "0" + frame1IndexText

        frame2IndexText = str(frame2Index)
        if (frame2Index < 10):
            frame2IndexText = "0" + frame2IndexText


        # load the pre-thresholded images
        originalF1 = io.imread(originalImageTimelapsePaths[frame1Index])
        originalF2 = io.imread(originalImageTimelapsePaths[frame2Index])


        # correct format for what I need
        if(originalF1.shape[0] < originalF1.shape[1] and originalF1.shape[1] == originalF1.shape[2]):
            originalF1 = np.moveaxis(originalF1,0,2)
            originalF2 = np.moveaxis(originalF2,0,2)

        if(len(originalF1.shape) == 4):
            originalF1 = originalF1[:,:,:,0]
            originalF2 = originalF2[:,:,:,0]

        print(originalF1.shape)

        # load the labeled images produced by MEL Fiji
        labelsF1 = io.imread(labeledImageTimelapsePaths[frame1Index])
        labelsF2 = io.imread(labeledImageTimelapsePaths[frame2Index])
        

        # correct format for what I need
        if(labelsF1.shape[0] < labelsF1.shape[1] and labelsF1.shape[1] == labelsF1.shape[2]):
            labelsF1 = np.moveaxis(labelsF1,0,2)
            labelsF2 = np.moveaxis(labelsF2,0,2)

        if(len(labelsF1.shape) == 4):
            labelsF1 = labelsF1[:,:,:,0]
            labelsF2 = labelsF2[:,:,:,0]

        print("labelsF1 shape " + str(labelsF1.shape))

        labelsF1_stack = labeledImageToStack(labelsF1)
        print("labelsF1_stack length " + str(len(labelsF1_stack)))
        labelsF2_stack = labeledImageToStack(labelsF2)

        # get all the event info from CSV file
        eventInfo = pd.read_csv(eventInfoPaths[frame1Index])


        if saveResults:
            newStack = generateGradientImageFromLocation(originalF1, eventInfo)         
            newStack = np.moveaxis(newStack, 2, 0) # z-axis must be first for the export to work properly 
            if(np.max(newStack) > 1):
                newStack = np.clip(newStack, 0, 255)
                ImageAnalysis.saveTifStack("{}\\P{}.tif".format(outputPath, frame1IndexText), (newStack).astype(np.uint8))
            else:
                newStack = np.clip(newStack, 0, 1)
                ImageAnalysis.saveTifStack("{}\\P{}.tif".format(outputPath, frame1IndexText), (newStack * 255).astype(np.uint8))

        print("Frame 1 num structures: {}  Frame 2 num structures: {}".format(len(labelsF1_stack), len(labelsF2_stack)))
        print(eventInfo)


        # try:
        # filteredBinary, stackLabels, numLabels = Morphology.labelStack(ImageAnalysis.binarizeStack(frame1Stack))
        # MEL.showMesh(filteredBinary, labels, locations, dupLabels, dupLocations, cziData.zVoxelWidth/cziData.xVoxelWidth, 1)


        print("VALIDATING RESULT")
        ####################
        # LOG which labels
        ####################

        # I need a start item (in this case -1) in order to allow for the case where no events are reported
        yesList = [(-1,-1)]
        noList = [(-1,-1)]
        unclearList = [(-1,-1)]
        thresholdList = [(-1,-1)]

        fusionCorrect = 0
        fissionCorrect = 0
        depolarisationCorrect = 0
        otherIncorrect = 0

        fuseFragDep = 0
        label = -1 # start by -1 since first display is test screen
        currentFrame = 0

        scaledStack = None

        # the list of dataframe indices that must be removed
        removeEventsIndices = []
            


        

        window = pyglet.window.Window(width=1400, height=650, caption="Check MEL events")
        gui = glooey.Gui(window)

        @window.event
        def on_key_press(symbol, modifiers):
            #print(symbol)
            #print(modifiers)
            global currentFrame
            global scaledStack
            global imageWidget

            if (modifiers == 16 or modifiers == 0) and symbol == 65363 and (currentFrame + 1) < originalF1.shape[2]: # right arrow
                currentFrame += 1

                imageWidget.update_image(scaledStack[currentFrame])


            elif (modifiers == 16 or modifiers == 0) and symbol == 65361 and (currentFrame - 1) >= 0: # left arrow
                currentFrame -= 1

                imageWidget.update_image(scaledStack[currentFrame])

            elif (modifiers == 16 or modifiers == 0) and symbol == 65362: # up arrow
                incrementYes()
                #print("YES")

            elif (modifiers == 16 or modifiers == 0) and symbol == 65364: # down arrow
                incrementNo()
                #print("NO")

            elif (modifiers == 16 or modifiers == 0) and symbol == 51539607552: # center
                incrementUnclear()
                #print("UNCLEAR")

            elif (modifiers == 16 or modifiers == 0) and symbol == 65365: # 9
                incrementThreshold()
                #print("THRESHOLD")
            
            elif symbol == 65293:
                nextLabel()

        #############################


        vBox = glooey.VBox()

        from trimesh.viewer import SceneWidget
        sc = trimesh.Scene()
        sphere = trimesh.creation.icosphere(1, 1, [0.0, 1.0, 0.0, 0.5])
        sc.add_geometry(sphere)
        sceneWidget = SceneWidget(sc)

        displayHBox = glooey.HBox()
        displayHBox.add_right(sceneWidget, size=650)


        im = np.dstack((np.ones((250, 750)) * 255, np.zeros((250, 750)), np.zeros((250, 750)))).astype(np.uint8)
        #im = np.stack((cziData.getStack(0,0)[0],) * 3, axis=-1)
        #imageWidget = GUI.ImageWidget(scaledStack[currentFrame])
        imageWidget = GUI.ImageWidget(im)
        displayHBox.add_left(imageWidget, size=750)


        vBox.add_top(displayHBox)

        hbox = glooey.HBox()
        hbox.alignment = 'bottom'


        buttons = [
            GUI.MyButton("Yes", "Yes", height=50, on_click=incrementYes),
            GUI.MyButton("No", "No", height=50, on_click=incrementNo),
            GUI.MyButton("Unclear", "Unclear", height=50, on_click=incrementUnclear),
            GUI.MyButton("Threshold Mistake", "Threshold", height=50, on_click=incrementThreshold),
        ]

        # infoLabel = GUI.MyLabel("Fusion: {}/{}  Fission: {}/{}  Depolarisation: {}/{} Other: {}".format(
        #      fusionCorrect,len(labels[0]), fissionCorrect, len(labels[1]), depolarisationCorrect, len(labels[2]),otherIncorrect
        #))
        for button in buttons:
            hbox.add(button)
            #hbox.add(infoLabel)

        vBox.add_bottom(hbox, size=50)
        gui.add(vBox)

        nextLabel()
        pyglet.app.run()

        print('Done')


        yesListNP = np.array(yesList)
        noListNP = np.array(noList)
        unclearListNP = np.array(unclearList)
        thresholdListNP = np.array(thresholdList)


        print("EVENTS BEFORE REMOVE")
        #print(eventInfo)
        print("Fusion: {} Fission: {} Depolarisation: {}".format(len(eventInfo.loc[eventInfo['EventType'] == 0].index),len(eventInfo.loc[eventInfo['EventType'] == 1].index),len(eventInfo.loc[eventInfo['EventType'] == 2].index)))

        eventInfoPostValidation = eventInfo.drop(removeEventsIndices).reset_index()

        print("EVENTS AFTER REMOVE")
        # print(eventInfoRemoved)
        print("Fusion: {} Fission: {} Depolarisation: {}".format(len(eventInfoPostValidation.loc[eventInfoPostValidation['EventType'] == 0].index),len(eventInfoPostValidation.loc[eventInfoPostValidation['EventType'] == 1].index),len(eventInfoPostValidation.loc[eventInfoPostValidation['EventType'] == 2].index)))
        eventInfoPostValidation.to_csv(outputPath + eventInfoPaths[frame1Index][eventInfoPaths[frame1Index].rfind("\\")+1:])


        #output new TIF and outcomes
        if saveResults:
            newStack = generateGradientImageFromLocation(originalF1, eventInfoPostValidation)         
            newStack = np.moveaxis(newStack, 2, 0) # z-axis must be first for the export to work properly 
            if(np.max(newStack) > 1):
                newStack = np.clip(newStack, 0, 255)
                ImageAnalysis.saveTifStack("{}\\P{}.tif".format(outputPath, frame1IndexText), (newStack).astype(np.uint8))
            else:
                newStack = np.clip(newStack, 0, 1)
                ImageAnalysis.saveTifStack("{}\\P{}.tif".format(outputPath, frame1IndexText), (newStack * 255).astype(np.uint8))



        
    # except Exception as e:
    #     print(e)
    #     print("SOMETHING WENT WRONG")

    startFrame = 0
