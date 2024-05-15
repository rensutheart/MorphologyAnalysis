# [Fuse locations, Frag locations, depolarize locations]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skimage.transform import rescale
from scipy.misc import imresize

import math

import csv

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

matplotlib.rcParams.update({'font.size': 12})


# calculate 4D histogram
def getHist(locations, im, x_res, y_res, z_res):
    hists = []
    
    if(locations.shape[0]==0):
        print("RETURN NONE")
        return None
    
    for z in range(0, math.ceil(im.shape[0]/z_res)):
        z_slices_locations = locations[z_res * z <= locations[:,0]]
        z_slices_locations = z_slices_locations[z_slices_locations[:,0] < z_res * (z+1)]
        histOut, _, _ = np.histogram2d(z_slices_locations[:,1], z_slices_locations[:,2], bins=[np.ceil(im.shape[1] / x_res),np.ceil(im.shape[2] / y_res)], range=[[0, im.shape[1]], [0, im.shape[2]]])
        
        #duplicate to ensure that the final image is the same depth as the original
        for i in range(0, z_res):
            hists.append(histOut)
    
    return np.array(hists)

def histStackToRGB(histStack, stackType = 0): # 0 is frag, 1 is fuse, 2 is dep
    maximum = np.max(histStack)
    
    values = np.linspace(0.0, 1.0, maximum+1)

    outStack = np.zeros((histStack.shape[0], histStack.shape[1], histStack.shape[2], 3))
    
    for z in range(0,histStack.shape[0]):
        for x in range(0,histStack.shape[1]):
            for y in range(0,histStack.shape[2]):
                colour = np.zeros(3)
                colour[stackType] = values[int(histStack[z, x, y])]
                outStack[z, x, y] = colour
                
    return outStack


def saveFrame(outputPathIn, imageStack, frameNum, eventType = None, Frame1=None, Frame2=None):
    frameText = str(frameNum)
    if (frameNum < 10):
        frameText = "0" + frameText
    
    if not os.path.exists(outputPathIn):
        os.makedirs(outputPathIn)     
   
    if(eventType == None):
        io.imsave("{}{}.tif".format(outputPathIn, frameText), (imageStack*255).astype(np.uint8))
        
        
        if not os.path.exists(outputPath + "Panel\\"):
            os.makedirs(outputPath + "Panel\\")  
    
        output = np.dstack((Frame1/255, Frame2/255, imageStack))
        io.imsave("{}{}.tif".format(outputPath + "Panel\\", frameText), (output*255).astype(np.uint8))
    
    elif(eventType == 0): #fusion
        if not os.path.exists(outputPathIn + "Fusion\\"):
            os.makedirs(outputPathIn + "Fusion\\")  
        io.imsave("{}{}.tif".format(outputPathIn + "Fusion\\", frameText), (imageStack*255).astype(np.uint8))
    elif(eventType == 1): #fragmentation
        if not os.path.exists(outputPathIn + "Fission\\"):
            os.makedirs(outputPathIn + "Fission\\")  
        io.imsave("{}{}.tif".format(outputPathIn + "Fission\\", frameText), (imageStack*255).astype(np.uint8))
    elif(eventType == 2):
        if not os.path.exists(outputPathIn + "Depolarization\\"):
            os.makedirs(outputPathIn + "Depolarization\\")  
        io.imsave("{}{}.tif".format(outputPathIn + "Depolarization\\", frameText), (imageStack*255).astype(np.uint8))
    
def countEvents(allDat, endFrame = None):
    outcomeCount = []
    end = 0
    if(endFrame == None):
        end = len(allDat[0])
    else:
         end = endFrame + 1 
        
    for frame in range(0, end):
        csOutcomes = []
        for cs in range(0, len(allDat)):
            try:
                fuseCount = allDat[cs][frame][0].shape[0]
            except:
                fuseCount = 0
                
            try:
                fragCount = allDat[cs][frame][1].shape[0]
            except:
                fragCount = 0
            
            try:
                depCount = allDat[cs][frame][2].shape[0]
            except:
                depCount = 0
                
            csOutcomes.append([fuseCount,fragCount, depCount ])
            
        outcomeCount.append(np.array(csOutcomes))
        
    return np.array(outcomeCount)

def moving_average(a, n=2) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plotOutcomeCounts(counts, average_total_structures, average_volume, cs_index=None, moving_average_n = 2, folderNum=0, showPlot=False, save=True, outputPathIn=""):
    x = np.linspace(0, counts.shape[0] - 1, counts.shape[0])
    x_ma = np.linspace(moving_average_n-1, counts.shape[0] - 1, counts.shape[0]-(moving_average_n-1))
    useCounts = np.zeros_like(counts[:,cs_index,:])
    useCountsMin = np.zeros_like(counts[:,cs_index,:])
    useCountsMax = np.zeros_like(counts[:,cs_index,:])
    if(cs_index == None): # take average
        print("Taking averages of counts")
        useCounts = np.average(counts, axis=1)
        useCountsMin = np.ndarray.min(counts, axis=1)
        useCountsMax = np.ndarray.max(counts, axis=1)
    else:
        useCounts = counts[:,cs_index,:]
        useCountsMin = counts[:,cs_index,:]
        useCountsMax = counts[:,cs_index,:]
    
    fig =  plt.figure(figsize=(4, 11), dpi=200)
    (ax1, ax2, ax3, ax4) = fig.subplots(4,1,sharex=True)


    ax1.plot(x_ma, moving_average(useCounts[:, 0], moving_average_n), color='green', label="Fusion")
    ax1.fill_between(x_ma, moving_average(useCountsMin[:, 0], moving_average_n), moving_average(useCountsMax[:, 0], moving_average_n), facecolor="green", alpha=0.25 )
    
    ax1.plot(x_ma, moving_average(useCounts[:, 1], moving_average_n), color='red', label="Fission")
    ax1.fill_between(x_ma, moving_average(useCountsMin[:, 1], moving_average_n), moving_average(useCountsMax[:, 1], moving_average_n), facecolor="red", alpha=0.25 )
    
    ax1.plot(x_ma, moving_average(useCounts[:, 2], moving_average_n), color='blue', label="Depolarization")
    ax1.fill_between(x_ma, moving_average(useCountsMin[:, 2], moving_average_n), moving_average(useCountsMax[:, 2], moving_average_n), facecolor="blue", alpha=0.25 )
    
    
    #OTHER COLORS
#    ax1.plot(x_ma, moving_average(useCounts[:, 0], moving_average_n), color='#66c2a5', label="Fusion")
#    ax1.plot(x_ma, moving_average(useCounts[:, 1], moving_average_n), color='#fc8d62', label="Fission")
#    ax1.plot(x_ma, moving_average(useCounts[:, 2], moving_average_n), color='#8da0cb', label="Depolarization")
    
#    ax1.plot(x_ma, moving_average(average_total_structures, moving_average_n), linestyle=':', linewidth=3, color='black')
#    ax1.set_title("Folder {} | Moving average period = {}".format(folderNum, moving_average_n))
    ax1.set_ylabel('Number of events')
    ax1.set_ylim(bottom=0)
    ax1.set_ylim(top=28)
    ax1.yaxis.set_ticks(np.arange(0, 29, 5))

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    legend1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, fontsize='medium', ncol=3)
    ax1.grid(which='major', axis='both', linestyle='--')

#    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#    ax2.plot(x_ma, moving_average(average_volume, moving_average_n), linestyle='--', linewidth=3, color='gray')
    
#    plt.show()
#    
#    if(save):
#        plt.savefig(outputPathIn + "Graph.png")
#        
#    if(not showPlot):
#        plt.close()
#        
#    fig, ax1 = plt.subplots()


#    plt.figure()
    # FIRST moving average then divide
    '''
    print("RATIO FISSION:FUSION")
    averageRatio = (useCounts[:, 1]/useCounts[:, 0])
    for a in averageRatio:
        print(a)
    '''
    
    ax2.plot(x_ma, moving_average(useCounts[:, 1], moving_average_n)/moving_average(useCounts[:, 0], moving_average_n), color='orange', label="Ratio")
#    ax2.fill_between(x_ma, moving_average(useCountsMin[:, 1], moving_average_n)/moving_average(useCountsMin[:, 0], moving_average_n), 
#                     moving_average(useCountsMax[:, 1], moving_average_n)/moving_average(useCountsMax[:, 0], moving_average_n), facecolor="orange", alpha=0.25 )
    
    # FIRST divide then moving average
    #ax2.plot(x_ma, moving_average(useCounts[:, 1]/useCounts[:, 0], moving_average_n), color='orange', label="Ratio")
#    ax3.title("Folder {} | Moving average period = {}".format(folderNum, moving_average_n))
    ax2.set_ylabel('Fission:Fusion ratio')
    ax2.axhline(y=1,linewidth=1.5, linestyle="--", color='black')
    ax2.set_ylim(bottom=0.3)
    ax2.set_ylim(top=1.5)    
    ax2.grid(which='major', axis='both', linestyle='--')
    #box = ax2.get_position()
    #ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    #legend2 = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, fontsize='medium', ncol=1)
#    legend3 = ax2.legend(loc='lower center', fancybox=True, shadow=False, fontsize='medium', ncol=1)
    
    
    
    maTot = moving_average(average_total_structures, moving_average_n)
    totLine = ax3.plot(x_ma, maTot , color='black', label="# structures")
    ax3.set_ylabel('Total number of structures')
#    ax2.set_title("Folder {} | Moving average period = {}".format(folderNum, moving_average_n))
    
    #ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    maAv =  moving_average(average_volume, moving_average_n)
    avLine = ax4.plot(x_ma, maAv,   color='magenta', label="Avg. volume")
    ax4.set_ylabel('Average structure volume\n[voxels]')
    ax3.set_ylim(bottom=(np.min(maTot)-0.4*(np.max(maTot) - np.min(maTot))))
    ax4.set_ylim(bottom=(np.min(maAv)-0.4*(np.max(maAv) - np.min(maAv))))
    ax4.set_xlabel('Frame')
    

    if folder == 1 and isNew:
        ax3.set_ylim(bottom=(75))
        ax3.set_ylim(top=(90))
        ax3.yaxis.set_ticks(np.arange(75, 91, 2.5))
    elif folder == 3 and isNew:
        ax3.set_ylim(bottom=(35))
        ax3.set_ylim(top=(50))
        ax3.yaxis.set_ticks(np.arange(35, 51, 2.5))
    elif folder == 1 and not isNew:
        ax3.set_ylim(bottom=(30))
        ax3.set_ylim(top=(65))
        ax3.yaxis.set_ticks(np.arange(30, 66, 5))
    elif folder == 2 and not isNew:
        ax3.set_ylim(bottom=(40))
        ax3.set_ylim(top=(75))
        ax3.yaxis.set_ticks(np.arange(40, 76, 5))

    ax3.grid(which='major', axis='both', linestyle='--')
    
    if folder == 1 and isNew:
        ax4.set_ylim(bottom=(2750))
        ax4.set_ylim(top=(3750))
    elif folder == 3 and isNew:
        ax4.set_ylim(bottom=(5000))
        ax4.set_ylim(top=(6000))
    elif folder == 1 and not isNew:
        ax4.set_ylim(bottom=(2250))
        ax4.set_ylim(top=(5250))
        ax4.yaxis.set_ticks(np.arange(2500, 5001, 500))
    elif folder == 2 and not isNew:
        ax4.set_ylim(bottom=(5000))
        ax4.set_ylim(top=(8000))
        ax4.yaxis.set_ticks(np.arange(5000, 8001, 500))
        
    ax4.grid(which='major', axis='both', linestyle='--')
    
#    ax3.set_ylim(bottom=(np.min(maTot)-0.4*(np.max(maTot)+30 - np.min(maTot))))
#    ax4.set_ylim(bottom=(np.min(maAv)-0.4*(np.max(maAv)+8000 - np.min(maAv))))
#    
#    ax3.set_ylim(top=(np.min(maTot)-0.4*(np.max(maTot)+30 - np.min(maTot))+30))
#    ax4.set_ylim(top=(np.min(maAv)-0.4*(np.max(maAv)+8000 - np.min(maAv))+8000))
#    box = ax3.get_position()
#    ax3.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    
    import matplotlib.ticker as plticker
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax4.xaxis.set_major_locator(loc)
    
    
#    lns = totLine + avLine
#    labs = [l.get_label() for l in lns]
    #legend3 = ax3.legend(lns, labs, loc='lower center', fancybox=True, shadow=False, fontsize='medium', ncol=2)
    
#    legend3 = ax3.legend(loc='lower center', fancybox=True, shadow=False, fontsize='medium', ncol=1)
#    legend4 = ax4.legend(loc='lower center', fancybox=True, shadow=False, fontsize='medium', ncol=1)

    
    


#    plt.show()
#    
#    if(save):
#        plt.savefig(outputPathIn + "GraphVol.png")
#        
#    if(not showPlot):
#        plt.close()
#    
    


    
    if(save):
        plt.savefig(outputPathIn + "GraphFullNarrowArea.png",bbox_inches='tight' )
        
    if(not showPlot):
        plt.close()
        
def countStructures(filePath):
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        volumes = []
        for row in csv_reader:
            volumes.append(float(row[0]))
        
    csv_file.close()
    return volumes

def getAverageCounts(counts):
    output = []
    
    for frame in counts:
        output.append(np.average(frame,axis=0))
        
    return np.array(output)

if __name__ == "__main__":
    '''
    THIS Code makes a MIP from the output TIF files
    '''
    from os.path import isdir, exists, join, isfile
    from os import makedirs, listdir
    import numpy as np
    from skimage import io

    dataPath = "E:\\PhD\\MEL_2020_Output_CHL\\"

    folderPaths = [join(dataPath, f) for f in listdir(dataPath) if isdir(join(dataPath, f)) and (f.endswith("czi"))]

    for folder in folderPaths:
        filePaths = [f for f in listdir(folder) if isfile(join(folder, f)) and (f.endswith("tif"))]

        for file in filePaths:
            img = io.imread(folder + '\\' + file)
            mip = np.max(img, axis=0)

            outPath = '{}\\MIP\\'.format(folder)
            if not exists(outPath):
                makedirs(outPath)

            io.imsave(outPath + file + ".png", mip)
            print('Saved', outPath + file + ".png")

    '''END OF SEGMENT'''






    
    isNew = False
    folder = 2
    
    endFrame = 14 #15
    
    if folder == 1 and isNew:
        endFrame = 14
    
    csStart = 95
    csEnd = 100
    
    if isNew:
        csStart = 97
        csEnd = 99
    else:
        csStart = 98
        csEnd = 100        
    
    penalizeLowFrequency = False;
    
    if isNew:
        # NEW
        path = "D:\\PhD\\MEL_Output_new\\ReOutput_June2019\\labeled1\\"
        outputPath = "D:\\PhD\\MEL_Output_new\\ReOutput_June2019_reduced\\AverageOutput\\{}\\".format(folder)
        imagePath = "D:\\PhD\\ForPaperNewSample\\{}\\Preproc\\".format(folder)
   
    else:
        # OLD
        path = "D:\\PhD\\MEL_Output\\ReOutput_OLD_June2019\\labeled1\\"
        outputPath = "D:\\PhD\\MEL_Output\\ReOutput_OLD_June2019\\AverageOutput\\{}\\".format(folder)
        imagePath = "D:\\PhD\\ForPaperOldSamples\\{}\\Preproc\\".format(folder)
    
    
    
    imagesPaths = [(imagePath+f) for f in listdir(imagePath) if isfile(join(imagePath, f)) and f.endswith("tif")]
    
    all_cs_filenames = []
    all_cs_volumes = []
    
    np.set_printoptions(suppress=True)
    
    # read file names
    for cs in range(csStart, csEnd + 1):
        csPath = "{}_cs{}\\{}\\".format(path, cs, folder)
        
        csFileNames = [(csPath+f) for f in listdir(csPath) if isfile(join(csPath, f)) and f.endswith("npy")]
        all_cs_filenames.append(csFileNames)
        
            
    
    # load files
    all_data = []
    for cs in range(csStart,csEnd + 1):
        i = cs - csStart
        csData = []
        frameIndex = 0
        for file in all_cs_filenames[i]:
            csData.append(np.load(file))
            
            # remove duplicates
            for event in range(0, 3):
                try:
                    csData[frameIndex][event] = np.unique(csData[frameIndex][event], axis=0)
                except:
                    print("cs {} frame {} event {} failed to perform unique (size=0?)".format(cs, frameIndex, event))
            
            frameIndex += 1
            
        all_data.append(csData)
        
    # read structure volumes
    for cs in range(csStart, csEnd + 1):
        csPath = "{}_cs{}\\{}\\".format(path, cs, folder)
        
        csVolumes = [(csPath+f) for f in listdir(csPath) if isfile(join(csPath, f)) and f.endswith("csv")]
        tempVol = []
        for volPath in csVolumes:
            tempVol.append(countStructures(volPath))
            
        all_cs_volumes.append(tempVol)
        
        
        
    #    
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #ax.set_zlim3d(0,20)
    #
    #frame = 0
    #for i in range(0,6):
    #    ax.scatter(data[i][frame][0][:,2], data[i][frame][0][:,1], data[i][frame][0][:,0], c="Green")
    #    ax.scatter(data[i][frame][1][:,2], data[i][frame][1][:,1], data[i][frame][1][:,0], c="Red")
    #    ax.scatter(data[i][frame][2][:,2], data[i][frame][2][:,1], data[i][frame][2][:,0], c="Blue")
    
    
#    for frame in range(0, len(imagesPaths)-1):
    for frame in range(0, endFrame + 1):
        # get all fused
        fuseLocations = []
        sepFuseLocations = []
        for i in range(0,csEnd - csStart + 1):
            tempSet = []
            for d in all_data[i][frame][0]:
                fuseLocations.append(d)
                tempSet.append(d)
                
            sepFuseLocations.append(np.array(tempSet))
        
        fuseLocations = np.array(fuseLocations)
        sepFuseLocations = np.array(sepFuseLocations)
        
        # get all fragmented
        fragLocations = []
        sepFragLocations = []
        for i in range(0,csEnd - csStart + 1):
            tempSet = []
            for d in all_data[i][frame][1]:
                fragLocations.append(d)
                tempSet.append(d)
                
            sepFragLocations.append(np.array(tempSet))
        
        fragLocations = np.array(fragLocations)
        sepFragLocations = np.array(sepFragLocations)
        
        # get all depolarized
        depLocations = []
        sepDepLocations = []
        for i in range(0,csEnd - csStart + 1):
            tempSet = []
            for d in all_data[i][frame][2]:
                depLocations.append(d)
                tempSet.append(d)
                
            sepDepLocations.append(np.array(tempSet))
        
        depLocations = np.array(depLocations)
        sepDepLocations = np.array(sepDepLocations)
        
        plot3D = False
        if(plot3D):
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_zlim3d(0,20)
            
            ax.scatter(fuseLocations[:,2], fuseLocations[:,1], fuseLocations[:,0], c="Green")
            ax.scatter(fragLocations[:,2], fragLocations[:,1], fragLocations[:,0], c="Red")
            ax.scatter(depLocations[:,2], depLocations[:,1], depLocations[:,0], c="Blue")
        
        
        
            
        
        im = io.imread(imagesPaths[frame])
        try:
            im2 = io.imread(imagesPaths[frame + 1])
        except:
            pass
        
        x_resolution = 10
        y_resolution = 10
        z_resolution = 2
        
        
        #calculate an average of the different thresholded images, instead of just the sum
        fuseHistSep = getHist(fuseLocations, im, x_res=x_resolution, y_res=y_resolution, z_res=z_resolution)
        #print(fuseHistSep)
        if(fuseHistSep is not None):
            fuseHistSep[:,:] = 0
            for cs_locations in sepFuseLocations:
                temp = getHist(cs_locations, im, x_res=x_resolution, y_res=y_resolution, z_res=z_resolution)
                if(temp is not None):
                    fuseHistSep += temp/np.max(temp)
                else:
                    fuseHistSep += np.zeros((math.ceil(im.shape[0]),math.ceil(im.shape[1]/x_resolution),math.ceil(im.shape[2]/y_resolution)))
                
            fuseHistSep = fuseHistSep / sepFuseLocations.shape[0] # take average
            a = np.unique(fuseHistSep)
            fuseHistSep = fuseHistSep / a[1] # scale according to different frequencies to show
            if(penalizeLowFrequency):
                fuseHistSep = np.multiply(fuseHistSep,fuseHistSep) # diminish the dim ones even more
            
            
            fuseRGBStack = histStackToRGB(fuseHistSep, 1)
        else:
            fuseRGBStack = np.zeros((math.ceil(im.shape[0]),math.ceil(im.shape[1]/x_resolution),math.ceil(im.shape[2]/y_resolution),3))
        #plt.figure()
        #plt.imshow(fuseRGBStack[0])
        
        # frag histogram
        fragHistSep = getHist(fragLocations, im, x_res=x_resolution, y_res=y_resolution, z_res=z_resolution)
        if(fragHistSep is not None):
            fragHistSep[:,:] = 0
            for cs_locations in sepFragLocations:
                temp = getHist(cs_locations, im, x_res=x_resolution, y_res=y_resolution, z_res=z_resolution)
                if(temp is not None):
                    fragHistSep += temp/np.max(temp)
                else:
                    fragHistSep += np.zeros((math.ceil(im.shape[0]),math.ceil(im.shape[1]/x_resolution),math.ceil(im.shape[2]/y_resolution)))
                
            fragHistSep = fragHistSep / sepFragLocations.shape[0]
            a = np.unique(fragHistSep)
            fragHistSep = fragHistSep / a[1]
            if(penalizeLowFrequency):
                fragHistSep = np.multiply(fragHistSep,fragHistSep) # diminish the dim ones even more
            
            fragRGBStack = histStackToRGB(fragHistSep, 0)
        else:
            fragRGBStack = np.zeros((math.ceil(im.shape[0]),math.ceil(im.shape[1]/x_resolution),math.ceil(im.shape[2]/y_resolution),3))
        #plt.figure()
        #plt.imshow(fragRGBStack[0])
        
        # dep histogram
        depHistSep = getHist(depLocations, im, x_res=x_resolution, y_res=y_resolution, z_res=z_resolution)
        if(depHistSep is not None):
            depHistSep[:,:] = 0
            for cs_locations in sepDepLocations:
                temp = getHist(cs_locations, im, x_res=x_resolution, y_res=y_resolution, z_res=z_resolution)
                if(temp is not None):
                    depHistSep += temp/np.max(temp)
                else:
                    depHistSep += np.zeros((math.ceil(im.shape[0]),math.ceil(im.shape[1]/x_resolution),math.ceil(im.shape[2]/y_resolution)))
                
            depHistSep = depHistSep / sepDepLocations.shape[0]
            a = np.unique(depHistSep)
            depHistSep = depHistSep / a[1]
            if(penalizeLowFrequency):
                depHistSep = np.multiply(depHistSep,depHistSep) # diminish the dim ones even more
            
            depRGBStack = histStackToRGB(depHistSep, 2)
        else:
            fragRGBStack = np.zeros((math.ceil(im.shape[0]),math.ceil(im.shape[1]/x_resolution),math.ceil(im.shape[2]/y_resolution),3))
        #plt.figure()
        #plt.imshow(depRGBStack[0])
        
        '''    
        # this is taking the average over everything (instead of per threshold)
        # fuse histogram
        fuseHist = getHist(fuseLocations, im, x_res=x_resolution, y_res=y_resolution, z_res=z_resolution)
        fuseRGBStack = histStackToRGB(fuseHist, 1)
        #figHist = plt.figure()
        #plt.imshow(fuseHist[0], cmap="Greens", vmin=0, vmax=np.max(fuseHist))
        #figHist = plt.figure()
        #plt.imshow(fuseRGBStack[0])
        
        # frag histogram
        fragHist = getHist(fragLocations, im, x_res=x_resolution, y_res=y_resolution, z_res=z_resolution)
        fragRGBStack = histStackToRGB(fragHist, 0)
        #figHist = plt.figure()
        #plt.imshow(fragHist[0], cmap="Reds", vmin=0, vmax=np.max(fragHist))
        #plt.imshow(fragRGBStack[0])
        
        # depolarize histogram
        depHist = getHist(depLocations, im, x_res=x_resolution, y_res=y_resolution, z_res=z_resolution)
        depRGBStack = histStackToRGB(depHist, 2)
        #figHist = plt.figure()
        #plt.imshow(depHist[0], cmap="Blues", vmin=0, vmax=np.max(depHist))
        #plt.imshow(depRGBStack[0])
        '''
        
        
        jointRGBStack = (fuseRGBStack + fragRGBStack + depRGBStack)
        #upscale
        jointRGBStackUp = []
        for j in jointRGBStack:
            upscaled = imresize(j, (im.shape[1], im.shape[2]), 'cubic')
            jointRGBStackUp.append(upscaled)
            
        jointRGBStackUp = np.array(jointRGBStackUp)
        overlaid = np.clip(jointRGBStackUp/255+im/255, 0.0, 1.0)
        
        
        #figHist = plt.figure()
        #plt.imshow(overlaid[0])
                
        #save TIFF file output stack
        #saveFrame(outputPath, overlaid, frame, Frame1 = im, Frame2 = im2)
        
        
        #FUSION
        #upscale
        fuseRGBStackUp = []
        for j in fuseRGBStack:
            upscaled = imresize(j, (im.shape[1], im.shape[2]), 'cubic')
            fuseRGBStackUp.append(upscaled)
            
        fuseRGBStackUp = np.array(fuseRGBStackUp)
        overlaid = np.clip(fuseRGBStackUp/255+im/255, 0.0, 1.0)
                
        #save TIFF file output stack
        #saveFrame(outputPath, overlaid, frame, 0)
        
        #FRAGMENTATION
        #upscale
        fragRGBStackUp = []
        for j in fragRGBStack:
            upscaled = imresize(j, (im.shape[1], im.shape[2]), 'cubic')
            fragRGBStackUp.append(upscaled)
            
        fragRGBStackUp = np.array(fragRGBStackUp)
        overlaid = np.clip(fragRGBStackUp/255+im/255, 0.0, 1.0)
                
        #save TIFF file output stack
        #saveFrame(outputPath, overlaid, frame, 1)
        
        
        #DEPOLARIZATION
        #upscale
        depRGBStackUp = []
        for j in depRGBStack:
            upscaled = imresize(j, (im.shape[1], im.shape[2]), 'cubic')
            depRGBStackUp.append(upscaled)
            
        depRGBStackUp = np.array(depRGBStackUp)
        overlaid = np.clip(depRGBStackUp/255+im/255, 0.0, 1.0)
                
        #save TIFF file output stack
        #saveFrame(outputPath, overlaid, frame, 2)       
        
        
        
        print("Done with frame",frame)
        
   
    # calculate the average number of structures and average structure volume
    average_total_structures = []
    average_volume = []
#    for frame in range(0, len(all_cs_volumes[0])):
    for frame in range(0, endFrame + 1):
        average = 0
        averageVol = 0
        for cs in range(0, len(all_cs_volumes)):
            average += len(all_cs_volumes[cs][frame])
            averageVol += np.average(all_cs_volumes[cs][frame])
            
        average = average / (csEnd - csStart + 1)
        averageVol = averageVol / (csEnd - csStart + 1)
            
        average_total_structures.append(average)
        average_volume.append(averageVol)
         
    average_total_structures = np.array(average_total_structures)
    average_volume = np.array(average_volume)
    
    frameCounts = countEvents(all_data, endFrame)
    plotOutcomeCounts(frameCounts, average_total_structures, average_volume, None, moving_average_n = 5, folderNum=folder, showPlot=False, save=True, outputPathIn=outputPath)


''' This code plots the event percentage '''
'''
def getAverage(counts):
    output = []
    
    for frame in counts:
        output.append(np.average(frame,axis=0))
        
    return np.array(output)


# this block of code plots the event percentage
moving_average_n = 5

avCount = getAverageCounts(frameCounts)
averTotStucMA = moving_average(average_total_structures, moving_average_n)

fusePerc = moving_average(avCount[:,0], moving_average_n)/averTotStucMA
fragPerc = moving_average(avCount[:,1], moving_average_n)/averTotStucMA
depPerc = moving_average(avCount[:,2], moving_average_n)/averTotStucMA


x_ma = np.linspace(moving_average_n-1, avCount.shape[0] - 1, avCount.shape[0]-(moving_average_n-1))
    
fig =  plt.figure(figsize=(4.6, 7.5), dpi=200)
(ax1) = fig.subplots(1,1,sharex=True)

ax1.plot(x_ma, fusePerc, color='green', label="Fusion")
ax1.plot(x_ma, fragPerc, color='red', label="Fission")
ax1.plot(x_ma, depPerc, color='blue', label="Depolarization")
#    ax1.plot(x_ma, moving_average(average_total_structures, moving_average_n), linestyle=':', linewidth=3, color='black')
#    ax1.set_title("Folder {} | Moving average period = {}".format(folderNum, moving_average_n))
ax1.set_ylabel('Percent of total structures')

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
legend1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, fontsize='medium', ncol=3)

# this block of code creates a table that matches the figures I have in the paper
fuse = avCount[:,0]/average_total_structures
frag = avCount[:,1]/average_total_structures
dep = avCount[:,2]/average_total_structures
skip = 4

for i in range(0, 4):
    print("{}, {}, {}, {}, {}".format(average_total_structures[i*skip], average_volume[i*skip], fuse[i*skip], frag[i*skip], dep[i*skip]))


'''
plt.close()
    