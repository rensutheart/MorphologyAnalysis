import ImageAnalysis
import SliceViewer

from os import listdir
from os.path import isfile, join

import numpy as np


'''
Loading image data
'''
path = "D:\\PhD\\ForPaperOldSamples\\1\\" # "path/to/samples/here"

stackTimelapsePath = [f for f in listdir(path) if isfile(join(path, f)) and (f.endswith("tif") or f.endswith("tiff"))]

timelapse = ImageAnalysis.loadTimelapseTif(path, scaleFactor=2, pad=True)
stack = timelapse [0]


stackCS = ImageAnalysis.contrastStretch(ImageAnalysis.contrastStretchSliceInStack(stack,0,100), 70, 98) # change 98 to different values for different results
binarizedStackCS = ImageAnalysis.binarizeStack(stackCS)

SliceViewer.multi_slice_viewer(stack)
SliceViewer.multi_slice_viewer(stackCS)
SliceViewer.multi_slice_viewer(binarizedStackCS)

ImageAnalysis.plotStackHist(stack)
ImageAnalysis.plotStackHist(stackCS)
