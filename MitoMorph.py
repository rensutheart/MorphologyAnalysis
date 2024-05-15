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

install flowdec manually
'''

import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True

# import tensorflow as tf
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

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

from CZI_Processor import cziFile

import MEL



import pyglet
import glooey
import GUI



personReviewName = "Rensu"
validate = True
saveResults = False
scaleFactor = 1.5

#writePath = "E:\\PhD\\MEL_2020_Output_2nd\\"
writePath = "C:\\PhD\\Output\\"
# writePath = "E:\\PhD\\MEL_2020_Output_CHL\\"

# cropPath = "G:\\PhD\\MEL_2020_CHOSEN\\CROPPED\\PostTreat\\"
# #Control
# cropX = [429,120]
# cropY = [466,120]
#makeRectangle(421, 453, 120, 120);

#Pre-treat (61)
#160x160
#makeRectangle(147, 84, 160, 160);

#Post-treat (62)
# cropX = [431,120]
# cropY = [384,120]
#makeRectangle(431, 384, 120, 120);

# Pre-treat (31)
# cropX = [335,120]
# cropY = [94,120]
#makeRectangle(335, 94, 120, 120);


#Post-treat (32)
# cropX = [0,120]
# cropY = [319,120]
# #makeRectangle(0, 319, 120, 120);

# cropImageList = []

#filePath = "G:\\PhD\\MEL_2020_Orignal_samples\\Con cells raw\\"
# filePath = "G:\\PhD\\MEL_2020_Orignal_samples\\2nd set\\New\\"
filePath = "C:\\PhD\\Original\\"
#filePath = "C:\\Users\\Rensu\\Downloads\\Treated2 cells\\"
stackTimelapsePath = [f for f in listdir(filePath) if isfile(join(filePath, f)) and (f.endswith("czi"))]
#fileName = "Con1.czi"
positionNum = 0

for i in range(0, len(stackTimelapsePath)):
    print(i, ' ', stackTimelapsePath[i])

startFileIndex = 0
startFrame = 0
print('Start file: ', stackTimelapsePath[startFileIndex])

#0,1,2,7,8,
#11, 12
rerunIndices = [0] #,15,16] #0,1,2,3,4,5,6,7,8,9,10,11,
# rerunIndices = [18,19,20,21,22,23,24] # 11, 8,9,10,17,14,15,0,1,2,3,4,5,6,7,12,13




def nextLabel():
    global fuseFragDep
    global label
    global sceneWidget
    global imageWidget
    global displayHBox
    global scaledStack
    global currentFrame

    label += 1
    print("fuseFragDep", fuseFragDep)
    print("label", label)
    if label < len(labels[fuseFragDep]):
        
        typeEvent = MEL.transType.NOTHING
        if fuseFragDep == 0:
            typeEvent = MEL.transType.FUSE
        elif fuseFragDep == 1:
            typeEvent = MEL.transType.FRAGMENT
        elif fuseFragDep == 2:
            typeEvent = MEL.transType.DEPOLARIZE

        highlighedLabels, scene = MEL.checkEvent(typeEvent, checkEventsList[0], checkEventsList[1], checkEventsList[2],
                                                checkEventsList[3],
                                                labels[fuseFragDep][label], locations[fuseFragDep][label], frame1Stack,
                                                frame2Stack, cziData.zVoxelWidth / cziData.xVoxelWidth, scaleFactor)

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
        fusionCorrect,len(labels[0]), fissionCorrect, len(labels[1]), depolarisationCorrect, len(labels[2]),otherIncorrect)
        print(text)
        #infoLabel = GUI.MyLabel(text)
        #hbox.add_right(infoLabel)

    elif fuseFragDep + 1 < 3:
        fuseFragDep += 1
        label = -1
        currentFrame = 0

        nextLabel()

    else:
        print("End of images")
        window.close()

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
    locationsCopy[fuseFragDep][label] = False
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
    locationsCopy[fuseFragDep][label] = False
    otherIncorrect += 1
    nextLabel()
    print("threshold")


#############################





# START HERE

# for fileNameIndex in range(startFileIndex, len(stackTimelapsePath)):
for fileNameIndex in rerunIndices:
#for fileNameIndex in range(startFileIndex, startFileIndex + 1):
    fileName = stackTimelapsePath[fileNameIndex]
    print("Processing: " + fileName)
    cziData = cziFile(filePath + fileName)
    cellOutcomes = []
    totalStructures = []
    averageVolume = []

    deconvPath = '{}\\{}\\Deconvolution\\'.format(writePath, fileName)
    if not path.exists(deconvPath):
        makedirs(deconvPath)

    outputPath = '{}\\{}\\'.format(writePath, fileName)
    if not path.exists(outputPath):
        makedirs(outputPath)

    filteredBinaryF2 = None
    stackLabelsF2 = None
    numLabelsF2 = None
    labeledStackF2 = None
    filteredLabeledStackF2 = None
    cannyLabeledStackF2 = None

    for timeIndex in range(startFrame, cziData.sizeT - 1): # -1 since frame 1 and frame 2 pair
        print("Time index: " + str(timeIndex) + ' for ' + fileName)

        frame1Index = timeIndex
        frame2Index = timeIndex + 1

        frame1IndexText = str(frame1Index)
        if(frame1Index < 10):
            frame1IndexText = "0" + frame1IndexText

        frame2IndexText = str(frame2Index)
        if (frame2Index < 10):
            frame2IndexText = "0" + frame2IndexText

        cziData.printSummary()
        # frame1Deconv = cziData.getStack(positionNum, frame1Index)
        # frame2Deconv = cziData.getStack(positionNum, frame2Index)

        # frame1Deconv = cziData.runDeconvolution(positionNum, frame1Index)
        if isfile("{}{}.tif".format(deconvPath, frame1IndexText)):
            frame1Deconv = io.imread("{}{}.tif".format(deconvPath, frame1IndexText))
            # this is only necessary for some cases, where the dimensions are swapped
            if frame1Deconv.shape[0] > frame1Deconv.shape[-1]:
                frame1Deconv = frame1Deconv.swapaxes(0, 2)
            print("loaded deconvolution from memory for Frame 1: ", frame1Deconv.shape)
        else:
            frame1Deconv = cziData.runDeconvolution(positionNum, frame1Index)
            ImageAnalysis.saveTifStack("{}{}.tif".format(deconvPath,  frame1IndexText), frame1Deconv/np.max(frame1Deconv))

        if isfile("{}{}.tif".format(deconvPath, frame2IndexText)):
            frame2Deconv = io.imread("{}{}.tif".format(deconvPath, frame2IndexText))
            # this is only necessary for some cases, where the dimensions are swapped
            if frame2Deconv.shape[0] > frame2Deconv.shape[-1]:
                frame2Deconv = frame2Deconv.swapaxes(0, 2)
            print("loaded deconvolution from memory for Frame 2: ", frame2Deconv.shape)
        else:
            frame2Deconv = cziData.runDeconvolution(positionNum, frame2Index)
            ImageAnalysis.saveTifStack("{}{}.tif".format(deconvPath, frame2IndexText), frame2Deconv/np.max(frame2Deconv))

        if (frame1Deconv.shape[0] == 1 or len(frame1Deconv) < 3):
            print("It appears as if the frame is 2D and not a z-stack. Padded with blank slices at the top and bottom")
            frame1Deconv = ImageAnalysis.padImageTo3D(frame1Deconv)
            frame2Deconv = ImageAnalysis.padImageTo3D(frame2Deconv)

        print('frame1Deconv.shape', frame1Deconv.shape)
        print('frame2Deconv.shape', frame2Deconv.shape)
        #plt.imshow(frame1Deconv[frame1Deconv.shape[0]//2])

        frame1Stack = ImageAnalysis.preprocess(frame1Deconv, scaleFactor=scaleFactor) # .copy()
        frame2Stack = ImageAnalysis.preprocess(frame2Deconv, scaleFactor=scaleFactor) #.copy()

        (low, high) = ImageAnalysis.determineHysteresisThresholds(frame1Stack, "{}\\Hist{}.png".format(outputPath, frame1IndexText))


        # ImageAnalysis.chooseHysteresisParams(frame1Stack)
        # continue

        try:
            (outputStack, filteredBinaryF1, outcomes, locations, labels, dupOutcomes, dupLocations, dupLabels, checkEventsList, totalStruc, averageStrucVolume,
            filteredBinaryF2, stackLabelsF2, numLabelsF2, labeledStackF2, filteredLabeledStackF2, cannyLabeledStackF2) = MEL.runMEL(frame1Stack, frame2Stack, '{}-{}'.format(frame1IndexText, frame2IndexText), 10, filteredBinaryF2, stackLabelsF2, numLabelsF2, labeledStackF2, filteredLabeledStackF2, cannyLabeledStackF2)
            
            #Create cropped panel
            # manuallyRemovedStack = ImageAnalysis.loadGenericImage("{}R{}.tif".format(outputPath, frame1IndexText))
            # miniPanel = ImageAnalysis.saveCroppedImagePanel(frame1Stack, frame2Stack, manuallyRemovedStack/255, cropX[0], cropX[1], cropY[0], cropY[1], "{}test_R{}.png".format(cropPath, timeIndex))
            # cropImageList.append(miniPanel)
            # cropImageList.append(np.ones((2,cropX[1]*3+4,3)))
            # ImageAnalysis.saveGenericImage("{}testFULL_R{}.png".format(cropPath, timeIndex), (np.vstack(cropImageList)*255).astype(np.uint8))
            # continue

            print("Outcomes: ", outcomes)
            print("Total Structures: ", totalStruc)
            print("Average Structure volume: ", averageStrucVolume)
            print()
            
            if type(outputStack) == bool: # something went wrong
                break

            # filteredBinary, stackLabels, numLabels = Morphology.labelStack(ImageAnalysis.binarizeStack(frame1Stack))
            # MEL.showMesh(filteredBinary, labels, locations, dupLabels, dupLocations, cziData.zVoxelWidth/cziData.xVoxelWidth, 1)

            if not validate:
                if saveResults:
                    ImageAnalysis.saveTifStack("{}\\{}.tif".format(outputPath, frame1IndexText), (outputStack * 255).astype(np.uint8))
                    print("SHAPE: ", filteredBinaryF1.shape)
                    ImageAnalysis.saveTifStack("{}\\T{}.tif".format(outputPath, frame1IndexText), (np.stack((filteredBinaryF1,filteredBinaryF1,filteredBinaryF1),axis=-1)*255).astype(np.uint8))        

                    cellOutcomes.append(outcomes)
                    totalStructures.append(totalStruc)
                    averageVolume.append(averageStrucVolume)
            else:
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

                # store whether the event should be retained or discareded
                locationsCopy = [[],[],[]]
                for i in range(0,3):
                    for loc in locations[i]:
                        locationsCopy[i].append(True)


                

                window = pyglet.window.Window(width=1400, height=650, caption="Check MEL events")
                gui = glooey.Gui(window)

                @window.event
                def on_key_press(symbol, modifiers):
                    #print(symbol)
                    #print(modifiers)
                    global currentFrame
                    global scaledStack
                    global imageWidget

                    if (modifiers == 16 or modifiers == 0) and symbol == 65363 and (currentFrame + 1) < outputStack.shape[0]: # right arrow
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


                # fusionDict ={'YES': yesListNP[np.12(yesListNP[:,0]==0),1][0],
                #              'NO': noListNP[np.where(noListNP[:,0]==0),1][0],
                #              'UNCLEAR': unclearListNP[np.where(unclearListNP[:,0]==0),1][0],
                #              'THRESHOLD': thresholdListNP[np.where(thresholdListNP[:,0]==0),1][0]}
                #
                # fissionDict ={'YES': yesListNP[np.where(yesListNP[:,0]==1),1][0],
                #              'NO': noListNP[np.where(noListNP[:,0]==1),1][0],
                #              'UNCLEAR': unclearListNP[np.where(unclearListNP[:,0]==1),1][0],
                #              'THRESHOLD': thresholdListNP[np.where(thresholdListNP[:,0]==1),1][0]}
                #
                # depDict ={'YES': yesListNP[np.where(yesListNP[:,0]==2),1][0],
                #              'NO': noListNP[np.where(noListNP[:,0]==2),1][0],
                #              'UNCLEAR': unclearListNP[np.where(unclearListNP[:,0]==2),1][0],
                #              'THRESHOLD': thresholdListNP[np.where(thresholdListNP[:,0]==2),1][0]}

                fusionDict = {}
                fissionDict = {}
                depDict = {}
                columnNames = ['YES', 'NO', 'UNCLEAR', 'THRESHOLD']
                typeDicts = [fusionDict, fissionDict, depDict]
                for t in range(0,3): # the type of event fusion fission depolarisation
                    lists = [yesListNP[np.where(yesListNP[:,0]==t),1][0],
                            noListNP[np.where(noListNP[:,0]==t),1][0],
                            unclearListNP[np.where(unclearListNP[:,0]==t),1][0],
                            thresholdListNP[np.where(thresholdListNP[:,0]==t),1][0]]
                    for i in range(0,4):
                        for l in lists[i]:
                            typeDicts[t][l] =  columnNames[i]


                if saveResults:
                    fusion_df = pd.DataFrame(sorted(typeDicts[0].items()))
                    fission_df = pd.DataFrame(sorted(typeDicts[1].items()))
                    dep_df = pd.DataFrame(sorted(typeDicts[2].items()))

                    fusion_df.to_csv('{}-{}({})-{}-{}_fusion_classification.csv'.format(personReviewName, fileName, positionNum, frame1Index, frame2Index))
                    fission_df.to_csv('{}-{}({})-{}-{}_fission_classification.csv'.format(personReviewName, fileName, positionNum, frame1Index, frame2Index))
                    dep_df.to_csv('{}-{}({})-{}-{}_depolarisation_classification.csv'.format(personReviewName, fileName, positionNum, frame1Index, frame2Index))

                #output new TIF and outcomes
                for i in range(0,3):
                    while False in locationsCopy[i]:
                        for index in range(0, len(locations[i])):
                            if not locationsCopy[i][index]:
                                del locations[i][index]
                                del locationsCopy[i][index]
                                break;

                if saveResults:
                    newStack = MEL.generateGradientImageFromLocation(frame1Stack, locations)
                    newStack = np.clip(newStack, 0, 1)
                    ImageAnalysis.saveTifStack("{}\\R{}.tif".format(outputPath, frame1IndexText), (newStack * 255).astype(np.uint8))

                outcomesDf = pd.DataFrame()
                outcomesDf['0'] = [len(locations[0])]
                outcomesDf['1'] = [len(locations[1])]
                outcomesDf['2'] = [len(locations[2])]
                outcomesDf.index += timeIndex
                
                if saveResults:
                    try:
                        existingDf = pd.read_csv("{}outcomesHuman.csv".format(outputPath))
                        existingDf = existingDf.drop('Unnamed: 0', 1)
                        existingDf = existingDf.append(outcomesDf, ignore_index=False)
                        existingDf.to_csv("{}outcomesHuman.csv".format(outputPath))
                        print("Appended to outcomesHuman")
                    except:
                        print("outcomesHuman probably doesn't exist. CREATED")
                        outcomesDf.to_csv("{}outcomesHuman.csv".format(outputPath))

            
        except Exception as e:
            print(e)
            print("SOMETHING WENT WRONG")

    startFrame = 0

    if not validate:
        if saveResults:
            df = pd.DataFrame(cellOutcomes)
            df['Total'] = totalStructures
            df['AverageVol'] = averageVolume
            df.to_csv("{}outcomes.csv".format(outputPath))

    




'''
TO GENERATE AN IMAGE PANEL
'''
'''
from os import listdir
from skimage import io
import numpy as np
import SliceViewer
import ImageAnalysis

path = 'C:\\Users\\rensu\\Downloads\\NTV Images\\'
outpath = 'C:\\Users\\rensu\\Downloads\\Output\\'
dir = listdir(path)

totalImagePanel = None

for d in dir:
    if 'No Denoising' not in d:
        casePath = path + d
        imagesPaths = listdir(casePath)

        casePanel = None

        for imP in imagesPaths:
            if 'FRAME1' in imP:
                print(imP)
                imPath = casePath + '\\' + imP
                image = io.imread(imPath)
                #if (type(image[0,0,0]) == np.float32):
                image = np.float32(image/np.max(image))

                print(type(image[0,0,0]))
                print(np.max(image))
                if image.shape[1] != 768:
                    image = ImageAnalysis.rescaleStackXY(image, 1.5)
                    image = np.float32(image / np.max(image))
                if casePanel is None:
                    casePanel = image
                else:
                    casePanel = np.dstack((casePanel, image))

        # SliceViewer.multi_slice_viewer(casePanel)

        # if totalImagePanel is None:
        #     totalImagePanel = casePanel.T
        # else:
        #     print(totalImagePanel.shape)
        #     print(casePanel.shape)
        #     totalImagePanel = np.hstack((totalImagePanel, casePanel))

        io.imsave("{}{}.tif".format(outpath, d), (casePanel * 255).astype(np.uint8))

# io.imsave("{}Total.tif".format(outpath, d), (totalImagePanel * 255).astype(np.uint8))


'''

'''

tuples, foundHWP, duplicates, duplicateHWP = findCloseEvents(withinAssociatedLabelsF1, withinAssociatedLabelsF1_HWP, withinChosenStatusF1_Fuse, 5)

sc = trimesh.Scene()
for index in range(0, len(tuples)):
    t = tuples[index]
    combinedLabels = np.int8(stack4D[t[0]] + stack4D[t[1]])
    # combinedLabels = np.int8(stack4D[labels[0][match][0]] + stack4D[labels[0][match][1]])
    labelMesh = Morphology.fullStackToMesh(combinedLabels, [1, 1, 1, 1])

    sphere = trimesh.creation.icosphere(1, 2.5, [0.0, 1.0, 0.0, 0.75])
    sphere.apply_translation(foundHWP[index])

    sc.add_geometry(labelMesh)
    sc.add_geometry(sphere)

for index in range(0, len(duplicates)):
    t = duplicates[index]
    combinedLabels = np.int8(stack4D[t[0]] + stack4D[t[1]])
    # combinedLabels = np.int8(stack4D[labels[0][match][0]] + stack4D[labels[0][match][1]])
    labelMesh = Morphology.fullStackToMesh(combinedLabels, [1, 1, 1, 1])

    sphere = trimesh.creation.icosphere(1, 1, [1.0, 0.0, 1.0, 0.5])
    sphere.apply_translation(duplicateHWP[index])

    sc.add_geometry(labelMesh)
    sc.add_geometry(sphere)

sc.show()





labelIndexMatch = []
for index in range(0, len(duplicates)):
    t = tuples[index]
    revT = (t[1], t[0])


    for labelIndex in range(0, len(labels[0])):
        if ((t[0] == labels[0][labelIndex][0] and t[1] == labels[0][labelIndex][1]) or
            (t[1] == labels[0][labelIndex][0] and t[0] == labels[0][labelIndex][1])):

            print("{} tuple {} labels {}".format(index, t, labels[0][labelIndex]))
            labelIndexMatch.append(labelIndex)

            combinedLabels = np.int8(stack4D[t[0]] + stack4D[t[1]])
            # combinedLabels = np.int8(stack4D[labels[0][match][0]] + stack4D[labels[0][match][1]])
            labelMesh = Morphology.fullStackToMesh(combinedLabels, [1, 1, 1, 1])

            sphere = trimesh.creation.icosphere(1, 2, [index / 255, 1.0, index / 255])
            sphere.apply_translation(locations[0][labelIndex])

            sc = trimesh.Scene()
            sc.add_geometry(labelMesh)
            sc.add_geometry(sphere)
            sc.show()

'''


'''
Paths
'''
# skipFrames = 1
# intensityParentPath = "D:\\PhD\\ForPaperNewSample\\"
# outputFolder = "D:\\PhD\\MEL_Output\\ReOutput_June2019\\labeled{}\\".format(skipFrames)
#
# numFolders = 3
#
# startFolderNum = 1  # 1 is lowest
# startFrameNumGlobal = 0  # 0 is lowest




'''
Loading image data
'''
'''
path = "D:\\PhD\\ForPaperOldSamples\\1\\" # laptop
# path = "E:\\PhD\\Paper4\\ForPaperOldSamples\\1\\"  # work PC

stackTimelapsePath = [f for f in listdir(path) if isfile(join(path, f)) and (f.endswith("tif") or f.endswith("tiff"))]

scaleFactor = 2

timelapse, metadata = ImageAnalysis.loadTimelapseTif(path, scaleFactor=scaleFactor, pad=True)
stack = timelapse [0]
'''

'''
Process a single stack
'''
'''
# Apply CONTRAST STRETCHING
stackCS = ImageAnalysis.contrastStretch(ImageAnalysis.contrastStretchSliceInStack(stack,0,100), 70, 98) # change 98 to different values for different results
binarizedStackCS = ImageAnalysis.binarizeStack(stackCS)
filteredBinaryCS, stackLabels, numLabels = Morphology.labelStack(binarizedStackCS)

SliceViewer.multi_slice_viewer(stackCS)
# SliceViewer.multi_slice_viewer(binarizedStackCS)
SliceViewer.multi_slice_viewer(filteredBinaryCS)
'''
# WITHOUT contrast stretching
#binarizedStack = ImageAnalysis.binarizeStack(stack)
#filteredBinary, stackLabels, numLabels = Morphology.labelStack(binarizedStack)

# TODO: I don't need stack4D since measure.regionprops already contains all that info in pr.coords...
# stack4D = Morphology.stack3DTo4D(stackLabels, numLabels)

#SliceViewer.multi_slice_viewer(stack)







'''
Morphology.calculateAllParameters(stackLabels,
                                  stack,
                                  metadata[0],
                                  scaleFactor=2,
                                  exportStructures3D=True,
                                  exportPath=writePath,
                                  andExportPanel=False,
                                  dataExportFileName='output')
'''

'''
This is for 3D render
'''
# import TiffStackViewer
# TiffStackViewer.showVolStack(timelapse[0])
#
# # get info from file (relating to depth)
# im = TiffStackViewer.getImages(path + stackTimelapsePath[0])
# TiffStackViewer.showVolume(im[0])
