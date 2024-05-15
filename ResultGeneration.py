'''
pip install pandas
    pip install numpy
pip install scikit-image
    pip install scipy

pip install psutil
pip install plotly
conda install -c plotly plotly-orca

'''

import pandas as pd
import numpy as np
from os import listdir, path, makedirs
from os.path import isfile, join, isdir

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skimage import io
import ImageAnalysis

import scipy

#inputPath = "E:\\PhD\\MEL_2020_Output\\"
# inputPath = "E:\\PhD\\MEL_2020_Output_CHL\\"
inputPath = "G:\\PhD\\MEL_2020_CHOSEN\\Output"
# inputPath = "G:\\PhD\\MEL_2020_CHOSEN\\Single"
# inputPath = "C:\\RESEARCH\\MEL_2020_CHOSEN\\Output\\"

typeData = "outcomes"  #outcomes outcomesRelative  outcomes(human)  humanComparison

scaleFactor = 1.5

skipFirstFrames = 0
totalFrames = 29

outputMIP = False

folders = [f for f in listdir(inputPath)]


for folder in folders:
    try:
        df = pd.read_csv("{}//{}//{}.csv".format(inputPath, folder, typeData)) 
        x_labels = df['Unnamed: 0'].to_numpy()
        df = df.drop('Unnamed: 0', 1)
        df = df[skipFirstFrames:totalFrames]
        df['AverageVol'] = df['AverageVol']/(scaleFactor*scaleFactor)
        movingAverage = df.rolling(5, center=True).mean()

        #df = df.rename(columns={'0': 'Fusion', '1': 'Fragmentation', '2': 'Depolarisation'})
        #movingAverage = movingAverage.rename(columns={'0': 'Fusion MA', '1': 'Fragmentation MA', '2': 'Depolarisation MA'})
        #df2 = df.join(movingAverage)

        # calculate relative change to start point
        dfRelative = pd.DataFrame()
        fusionRelative = []
        try:
            fusionFirstNum = (df['0'].tolist()[0] + df['0'].tolist()[1] +  df['0'].tolist()[2] + df['0'].tolist()[3] + df['0'].tolist()[4])/5
        except:
            fusionFirstNum = df['0'].tolist()[0]
        for fusionCount in df['0'].tolist():
            fusionRelative.append(fusionCount/fusionFirstNum)
        dfRelative['fusion'] = fusionRelative

        fissionRelative = []
        try:
            fissionFirstNum = (df['1'].tolist()[0] + df['1'].tolist()[1] + df['1'].tolist()[2] + df['1'].tolist()[3] + df['1'].tolist()[4])/5
        except:
            fissionFirstNum = df['1'].tolist()[0]
        for fissionCount in df['1'].tolist():
            fissionRelative.append(fissionCount/fissionFirstNum)
        dfRelative['fission'] = fissionRelative

        depRelative = []
        try:
            depFirstNum = (df['2'].tolist()[0] + df['2'].tolist()[1] + df['2'].tolist()[2] + df['2'].tolist()[3] + df['2'].tolist()[4])/5
        except:
            depFirstNum = df['2'].tolist()[0]
        for depCount in df['2'].tolist():
            if(depFirstNum != 0):
                depRelative.append(depCount/depFirstNum)
            else:
                depRelative.append(0)
        dfRelative['depolarisation'] = depRelative

        totalRelative = []
        try:
            totalFirstNum = (df['Total'].tolist()[0] + df['Total'].tolist()[1] + df['Total'].tolist()[2] + df['Total'].tolist()[3] + df['Total'].tolist()[4])/5
        except:
            totalFirstNum = df['Total'].tolist()[0]        
        for totalCount in df['Total'].tolist():
            totalRelative.append(totalCount/totalFirstNum)
        dfRelative['total'] = totalRelative

        avgRelative = []
        try:
            avgFirstNum = (df['AverageVol'].tolist()[0] + df['AverageVol'].tolist()[1] + df['AverageVol'].tolist()[2] + df['AverageVol'].tolist()[3] + df['AverageVol'].tolist()[4])/5
        except:
            avgFirstNum = df['AverageVol'].tolist()[0]
        for avgCount in df['AverageVol'].tolist():
            avgRelative.append(avgCount/avgFirstNum)
        dfRelative['average'] = avgRelative

        dfRelative.to_csv("{}//{}//outcomesRelative.csv".format(inputPath, folder))
        print("Saved Relative")

        movingAverageRelative = dfRelative.rolling(5, center=True).mean()


        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.5, 0.5])#,
            #subplot_titles=(['Control sample'])) #'Control sample', 'Pre-treatment sample', 'Post-treatment sample'))

        plotlyColors = px.colors.qualitative.Plotly
        colors = [plotlyColors[2], plotlyColors[1], plotlyColors[0], plotlyColors[3], plotlyColors[4]]

        # change x labels to time
        x_labels = np.linspace(0,28,29)*10

        markerSize = 3
        fig.add_trace(go.Scatter(x=x_labels, y=df['0'].tolist(), name='Fusion',
                                    mode='markers', marker=dict(size=markerSize,color=colors[0])),
                                 #line=dict(color='green', width=1, dash='dot')),
                                 row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=df['1'].tolist(), name='Fission',
                                    mode='markers', marker=dict(size=markerSize,color=colors[1])),
                                 #line=dict(color='red', width=1, dash='dot')),
                                 row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=df['2'].tolist(), name='Depolarisation',
                                mode='markers', marker=dict(size=markerSize,color=colors[2])),
                                # line=dict(color='blue', width=1, dash='dot')),
                                 row=1, col=1)

        fig.add_trace(go.Scatter(x=x_labels, y=movingAverage['0'].tolist(), name='Fusion MA',
                                 line=dict(color=colors[0], width=4)),
                                 row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=movingAverage['1'].tolist(), name='Fission MA',
                                 line=dict(color=colors[1], width=4)),
                                 row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=movingAverage['2'].tolist(), name='Depolarisation MA',
                                 line=dict(color=colors[2], width=4)),
                                 row=1, col=1)

        ###############
        #relative plots
        ###############
        # fig.add_trace(go.Scatter(x=x_labels, y=dfRelative['fusion'].tolist(), name='Fusion',
        #                         mode='markers', marker=dict(size=markerSize,color='green')),
        #                         # line=dict(color='green', width=1, dash='dot')),
        #                          row=2, col=1)
        # fig.add_trace(go.Scatter(x=x_labels, y=dfRelative['fission'].tolist(), name='Fission',
        #                         mode='markers', marker=dict(size=markerSize,color='red')),
        #                         #  line=dict(color='red', width=1, dash='dot')),
        #                          row=2, col=1)
        # fig.add_trace(go.Scatter(x=x_labels, y=dfRelative['depolarisation'].tolist(), name='Depolarisation',
        #                         mode='markers', marker=dict(size=markerSize,color='blue')),
        #                         #  line=dict(color='blue', width=1, dash='dot')),
        #                          row=2, col=1)

        # fig.add_trace(go.Scatter(x=x_labels, y=movingAverageRelative['fusion'].tolist(), name='Fusion MA',
        #                          line=dict(color='green', width=4)),
        #                          row=2, col=1)
        # fig.add_trace(go.Scatter(x=x_labels, y=movingAverageRelative['fission'].tolist(), name='Fission MA',
        #                          line=dict(color='red', width=4)),
        #                          row=2, col=1)
        # fig.add_trace(go.Scatter(x=x_labels, y=movingAverageRelative['depolarisation'].tolist(), name='Depolarisation MA',
        #                          line=dict(color='blue', width=4)),
        #                          row=2, col=1)

        #Ratio
        # dfRatio = pd.DataFrame()
        # dfRatio['ratio'] = df['1']/df['0']
        # movingAverageRatio = dfRatio.rolling(5, center=True).mean()
        # fig.add_trace(go.Scatter(x=x_labels, y=(movingAverageRatio['ratio']), name='Ratio',
        #                          line=dict(color='gold', width=4)),
        #                          row=2, col=1)

        fig.add_trace(go.Scatter(x=x_labels, y=(movingAverage['Total'].tolist()), name='Total',
                                 line=dict(color='black', width=4)),
                                 row=2, col=1)

        fig.add_trace(go.Scatter(x=x_labels, y=(movingAverage['AverageVol'].tolist()), name='Average',
                                 line=dict(color='magenta', width=4)),
                                 row=3, col=1)

        gridColor = 'lightgray'
        fig.update_yaxes(title_text="Number of events", row=1, col=1, range=[0, 15],  gridcolor=gridColor) 
        # fig.update_yaxes(title_text="% change from start", row=2, col=1) # , range=[0, 50]
        # fig.update_yaxes(title_text="Fission:Fusion ratio", range=[0.25, 1.75], dtick=0.25, row=2, col=1,  gridcolor=gridColor)
        if folder == 'Con001.czi':
            fig.update_yaxes(title_text="Total number of structures", row=2, col=1, dtick=10, range=[80, 130],  gridcolor=gridColor)
            fig.update_yaxes(title_text="Average structure volume <br> [voxels]", row=3, col=1, dtick=200, range=[100, 1200],  gridcolor=gridColor)
        elif folder == 'H2O231.czi':
            fig.update_yaxes(title_text="Total number of structures", row=2, col=1, dtick=10, range=[80, 130], gridcolor=gridColor)
            fig.update_yaxes(title_text="Average structure volume <br> [voxels]", row=3, col=1, dtick=200, range=[100, 1200],  gridcolor=gridColor)
        elif folder == 'H2O232.czi':
            fig.update_yaxes(title_text="Total number of structures", row=2, col=1, dtick=10, gridcolor=gridColor)
            fig.update_yaxes(title_text="Average structure volume <br> [voxels]", row=3, col=1, dtick=200, range=[700, 1800],  gridcolor=gridColor)
        # elif folder == 'H2O261.czi':
        #     fig.update_yaxes(title_text="Total number of structures", row=2, col=1, dtick=10, range=[50, 100],  gridcolor=gridColor)
        #     fig.update_yaxes(title_text="Average structure volume <br> [voxels]", row=3, col=1, dtick=200, range=[600, 1600],  gridcolor=gridColor)
        # elif folder == 'H2O262.czi':
        #     fig.update_yaxes(title_text="Total number of structures", row=2, col=1, dtick=10, range=[50, 100],  gridcolor=gridColor)
        #     fig.update_yaxes(title_text="Average structure volume <br> [voxels]", row=3, col=1, dtick=200, range=[200, 1200],  gridcolor=gridColor)
        else:
            fig.update_yaxes(title_text="Total number of structures", row=2, col=1,  gridcolor=gridColor)
            fig.update_yaxes(title_text="Average structure volume <br> [voxels]", row=3, col=1,  gridcolor=gridColor)

        fig.update_xaxes(dtick=50, row=1, col=1, tickangle = 0, range=[0,280],  gridcolor=gridColor) #Frame
        fig.update_xaxes(dtick=50, row=2, col=1, range=[0,280],  gridcolor=gridColor) #Frame
        fig.update_xaxes(dtick=50, row=3, col=1, range=[0,280],  gridcolor=gridColor) #Frame
        fig.update_xaxes(title_text="Time [s]", row=3, col=1,  gridcolor=gridColor) #Frame
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')
        fig.update_layout(title=folder, height=800, width=500)
        fig.update_layout(showlegend=False, plot_bgcolor="#FFF",   font=dict(
        family="Arial",
        size=16
    )) 
        if folder == 'Con001.czi' or folder == 'H2O231.czi' or folder == 'H2O232.czi':
            print(movingAverage['0'])
            dataDf = pd.DataFrame(index=['Fusion', 'Fission', 'Depolarisation'])
            dataDf['Average'] = [np.mean(df['0']), np.mean(df['1']), np.mean(df['2'])]
            # dataDf['Variance'] = [np.var(df['0'], ddof=1), np.var(df['1'], ddof=1), np.var(df['2'], ddof=1)]
            dataDf['Std Dev'] = [np.std(df['0'], ddof=1), np.std(df['1'], ddof=1), np.std(df['2'], ddof=1)]
            print(dataDf)
            print(folder)
            # print("Fusion Var: {}  Fission Var: {}  Depo Var: {}".format(np.var(df['0'], ddof=1), np.var(df['1'], ddof=1), np.var(df['2'], ddof=1)))
            # print("Fusion Avg: {}  Fission Avg: {}  Depo Avg: {}".format(np.mean(df['0']), np.mean(df['1']), np.mean(df['2'])))
            fig.show()
        fig.write_image("{}//{}//eventsSummary.png".format(inputPath, folder))
        print("Wrote", folder);


    except FileNotFoundError:
        print("Could not find outcomes.csv for " + folder)
    except Exception as e:
        print("SKIPPED " + folder)
        print(e)
        


def outputAllAsMIP(inPath):
    folders = [f for f in listdir(inPath) if isdir(join(inPath, f))]

    for folder in folders:
        if(folder.startswith("H2O23")):
            folderPath = inPath + "\\" + folder
            files = [f for f in listdir(folderPath) if isfile(join(folderPath, f))  and (f.endswith(".tif"))  and (f.startswith("R"))]
            for file in files:
                img = io.imread("{}\\{}".format(folderPath, file))

                outpath = "{}\\MIP\\".format(folderPath)
                if not path.exists(outpath):
                    makedirs(outpath)
                io.imsave("{}{}.png".format(outpath, file), ImageAnalysis.stackToMIP(img))
                print("Done with ", file);


if(outputMIP):
    outputAllAsMIP(inputPath)

def averageOfSet(df, averageFusion, averageFission, averageRatio, averageDepolarisation, averageVolume, averageTotal, avgCount):
    startIndex = 1
    if(len(averageFusion) == 0):
        averageFusion = df[df.columns[0 + startIndex]]
        averageFission = df[df.columns[1 + startIndex]]
        averageRatio = df[df.columns[1 + startIndex]]/df[df.columns[0 + startIndex]]
        averageDepolarisation = df[df.columns[2 + startIndex]]
        averageTotal = df[df.columns[3 + startIndex]]
        averageVolume = df[df.columns[4 + startIndex]]
        avgCount = 1
    else: # calculate average
        if(len( df[df.columns[0 + startIndex]]) < 29):
            print(folder, " IS TOO SHORT")
        else:
            averageFusion = averageFusion*avgCount + df[df.columns[0 + startIndex]]
            averageFission = averageFission*avgCount + df[df.columns[1 + startIndex]]
            averageRatio = averageRatio*avgCount + df[df.columns[1 + startIndex]]/df[df.columns[0 + startIndex]]
            averageDepolarisation = averageDepolarisation*avgCount + df[df.columns[2 + startIndex]]
            averageTotal = averageTotal*avgCount + df[df.columns[3 + startIndex]]
            averageVolume = averageVolume*avgCount + df[df.columns[4 + startIndex]]
            avgCount += 1
            averageFusion /= avgCount
            averageFission /= avgCount
            averageRatio /= avgCount
            averageDepolarisation /= avgCount
            averageTotal /= avgCount
            averageVolume /= avgCount

    return (averageFusion, averageFission, averageRatio, averageDepolarisation, averageVolume, averageTotal, avgCount)

def addPlots(fig, x_labels, df, movingAverageControl, row, markerSize = 3):
    plotlyColors = px.colors.qualitative.Plotly
    colors = [plotlyColors[2], plotlyColors[1], plotlyColors[0], plotlyColors[3], plotlyColors[4]]

    fig.add_trace(go.Scatter(x=x_labels, y=df['fusion'].tolist(), mode='markers', marker=dict(size=markerSize,color=colors[0]), name='Fusion',
                            #line=dict(color='green', width=1, dash='dot')
                            ),
                            row=row, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=df['fission'].tolist(), mode='markers', marker=dict(size=markerSize,color=colors[1]), name='Fission',
                                #line=dict(color='red', width=1, dash='dot')
                                ),
                                row=row, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=df['depolarisation'].tolist(), mode='markers', marker=dict(size=markerSize,color=colors[2]), name='Depolarisation',
                                #line=dict(color='blue', width=1, dash='dot')
                                ),
                                row=row, col=1)
    # fig.add_trace(go.Scatter(x=x_labels, y=df['total'].tolist(), mode='markers', marker=dict(size=markerSize,color='black'), name='Total',
    #                             #line=dict(color='black', width=1, dash='dot')
    #                             ),
    #                             row=row, col=1)
    # fig.add_trace(go.Scatter(x=x_labels, y=df['average'].tolist(), mode='markers', marker=dict(size=markerSize,color='magenta'), name='Average Volume',
    #                             #line=dict(color='magenta', width=1, dash='dot')
    #                             ),
    #                             row=row, col=1)

    # PLOT Ratio
    # fig.add_trace(go.Scatter(x=x_labels, y=df['ratio'].tolist(), mode='markers', marker=dict(size=markerSize,color='gold'), name='Fission:Fusion',
    #                             #line=dict(color='green', width=1, dash='dot')
    #                             ),
    #                             row=row, col=2)


    fig.add_trace(go.Scatter(x=x_labels, y=movingAverageControl['fusion'].tolist(), name='Fusion MA',
                                line=dict(color=colors[0], width=4)),
                                row=row, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=movingAverageControl['fission'].tolist(), name='Fission MA',
                                line=dict(color=colors[1], width=4)),
                                row=row, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=movingAverageControl['depolarisation'].tolist(), name='Depolarisation MA',
                                line=dict(color=colors[2], width=4)),
                                row=row, col=1)
    # fig.add_trace(go.Scatter(x=x_labels, y=movingAverageControl['total'].tolist(), name='Total MA',
    #                             line=dict(color='black', width=4)),
    #                             row=row, col=1)
    # fig.add_trace(go.Scatter(x=x_labels, y=movingAverageControl['average'].tolist(), name='Average Volume MA',
    #                             line=dict(color='magenta', width=4)),
    #                             row=row, col=1)

    # PLOT Ratio
    # fig.add_trace(go.Scatter(x=x_labels, y=movingAverageControl['ratio'].tolist(), name='Ratio MA',
    #                             line=dict(color='gold', width=4)),
    #                             row=row, col=2)



def addPlotsSeparate(fig2, x_labels, dfContr, dfPre, dfPost, dfEnd, typeName, row, markerSize = 3):
    fig2.add_trace(go.Scatter(x=x_labels, y=dfContr[typeName].tolist(), mode='markers', marker=dict(size=markerSize,color='#FFB14E'), name='Control',
                            #line=dict(color='green', width=1, dash='dot')
                            ),
                            row=row, col=1)
    fig2.add_trace(go.Scatter(x=x_labels, y=dfPre[typeName].tolist(), mode='markers', marker=dict(size=markerSize,color='#CD34B5'), name='Pre-treatment',
                                #line=dict(color='red', width=1, dash='dot')
                                ),
                                row=row, col=1)
    fig2.add_trace(go.Scatter(x=x_labels, y=dfPost[typeName].tolist(), mode='markers', marker=dict(size=markerSize,color='#0000FF'), name='Post-treatment',
                                #line=dict(color='blue', width=1, dash='dot')
                                ),
                                row=row, col=1)
    fig2.add_trace(go.Scatter(x=x_labels, y=dfEnd[typeName].tolist(), mode='markers', marker=dict(size=markerSize,color='#000000'), name='End State',
                                #line=dict(color='blue', width=1, dash='dot')
                                ),
                                row=row, col=1)

    movingAverageContr = dfContr.rolling(5, center=True).mean()
    movingAveragePre = dfPre.rolling(5, center=True).mean()
    movingAveragePost = dfPost.rolling(5, center=True).mean()
    movingAverageEnd = dfEnd.rolling(5, center=True).mean()
    fig2.add_trace(go.Scatter(x=x_labels, y=movingAverageContr[typeName].tolist(), name='Control MA',
                                line=dict(color='#FFB14E', width=4)),
                                row=row, col=1)
    fig2.add_trace(go.Scatter(x=x_labels, y=movingAveragePre[typeName].tolist(), name='Pre-treatment MA',
                                line=dict(color='#CD34B5', width=4)),
                                row=row, col=1)
    fig2.add_trace(go.Scatter(x=x_labels, y=movingAveragePost[typeName].tolist(), name='Post-treatment MA',
                                line=dict(color='#0000FF', width=4)),
                                row=row, col=1)
    fig2.add_trace(go.Scatter(x=x_labels, y=movingAverageEnd[typeName].tolist(), name='End State MA',
                                line=dict(color='#000000', width=4)),
                                row=row, col=1)

#inputPath = "G:\\PhD\\MEL_2020_CHOSEN\\Output"

folders = [f for f in listdir(inputPath)]

controlList = []
preTreatList = []
postTreatList = []
endList = []

averageFusionControl = []
averageFissionControl = []
averageRatioControl = []
averageDepolarisationControl = []
averageVolumeControl = []
averageTotalControl = []
avgControlCount = 0

averageFusionPreTreat = []
averageFissionPreTreat = []
averageRatioPreTreat = []
averageDepolarisationPreTreat = []
averageVolumePreTreat = []
averageTotalPreTreat = []
avgPreTreatCount = 0

averageFusionPostTreat = []
averageFissionPostTreat = []
averageRatioPostTreat = []
averageDepolarisationPostTreat = []
averageVolumePostTreat = []
averageTotalPostTreat = []
avgPostTreatCount = 0

averageFusionEnd = []
averageFissionEnd= []
averageRatioEnd = []
averageDepolarisationEnd = []
averageVolumeEnd = []
averageTotalEnd = []
avgEndCount = 0

for folder in folders:
    try:
        print(folder)
        df = pd.read_csv("{}\\{}\\{}.csv".format(inputPath, folder, typeData))  #outcomesRelative  outcomes
        df = df[skipFirstFrames:totalFrames]
        df['AverageVol'] = df['AverageVol']/(scaleFactor*scaleFactor)
        if(folder.startswith("Con")): # control
            print(df.columns)
            controlList.append(df)
            (averageFusionControl, averageFissionControl, averageRatioControl, averageDepolarisationControl, averageVolumeControl, averageTotalControl, avgControlCount) = \
                averageOfSet(df, averageFusionControl, averageFissionControl, averageRatioControl, averageDepolarisationControl, averageVolumeControl, averageTotalControl, avgControlCount)
        elif(folder.startswith("H2O2E")): # end
            endList.append(df)
            # (averageFusionEnd, averageFissionEnd, averageRatioEnd, averageDepolarisationEnd, averageVolumeEnd, averageTotalEnd, avgEndCount) = \
            #     averageOfSet(df, averageFusionEnd, averageFissionEnd, averageRatioEnd, averageDepolarisationEnd, averageVolumeEnd, averageTotalEnd, avgEndCount)
        elif(folder.startswith("H2O2") and folder.endswith("1.czi")): # pre-treatment
            preTreatList.append(df)
            (averageFusionPreTreat, averageFissionPreTreat, averageRatioPreTreat, averageDepolarisationPreTreat, averageVolumePreTreat, averageTotalPreTreat, avgPreTreatCount) = \
                averageOfSet(df, averageFusionPreTreat, averageFissionPreTreat, averageRatioPreTreat, averageDepolarisationPreTreat, averageVolumePreTreat, averageTotalPreTreat, avgPreTreatCount)
        elif(folder.startswith("H2O2") and folder.endswith("2.czi")): # post-treatment
            postTreatList.append(df)
            (averageFusionPostTreat, averageFissionPostTreat, averageRatioPostTreat, averageDepolarisationPostTreat, averageVolumePostTreat, averageTotalPostTreat, avgPostTreatCount) = \
                averageOfSet(df, averageFusionPostTreat, averageFissionPostTreat, averageRatioPostTreat, averageDepolarisationPostTreat, averageVolumePostTreat, averageTotalPostTreat, avgPostTreatCount)
        else:
            print("Skipped", folder)
    except Exception as e:
        print("File does not exist in " + folder)
        print(e)

dfContr = pd.DataFrame()
dfContr['fusion'] = averageFusionControl
dfContr['fission'] = averageFissionControl
dfContr['ratio'] = averageRatioControl
dfContr['depolarisation'] = averageDepolarisationControl
dfContr['total'] = averageTotalControl
dfContr['average'] = averageVolumeControl
movingAverageControl = dfContr.rolling(5, center=True).mean()

import numpy as np  
x_labels = np.linspace(0,28,29)*10

fig = make_subplots(
    rows=3, cols=1,
    row_heights=[1.0, 1.0, 1.0],
    subplot_titles=('Average of control samples', 'Average of pre-treatment samples', 'Average of post-treatment samples'))

addPlots(fig, x_labels, dfContr, movingAverageControl, 1)

dfPre = pd.DataFrame()
dfPre['fusion'] = averageFusionPreTreat
dfPre['fission'] = averageFissionPreTreat
dfPre['ratio'] = averageRatioPreTreat
dfPre['depolarisation'] = averageDepolarisationPreTreat
dfPre['total'] = averageTotalPreTreat
dfPre['average'] = averageVolumePreTreat
movingAveragePreTreat = dfPre.rolling(5, center=True).mean()

addPlots(fig, x_labels, dfPre, movingAveragePreTreat, 2)

dfPost = pd.DataFrame()
dfPost['fusion'] = averageFusionPostTreat
dfPost['fission'] = averageFissionPostTreat
dfPost['ratio'] = averageRatioPostTreat
dfPost['depolarisation'] = averageDepolarisationPostTreat
dfPost['total'] = averageTotalPostTreat
dfPost['average'] = averageVolumePostTreat
movingAveragePostTreat = dfPost.rolling(5, center=True).mean()

addPlots(fig, x_labels, dfPost, movingAveragePostTreat, 3)

dfEnd = pd.DataFrame()
# dfEnd['fusion'] = averageFusionEnd
# dfEnd['fission'] = averageFissionEnd
# dfEnd['depolarisation'] = averageDepolarisationEnd
# dfEnd['total'] = averageTotalEnd
# dfEnd['average'] = averageVolumeEnd
# movingAverageEnd = dfEnd.rolling(5, center=True).mean()

# addPlots(fig, x_labels, dfEnd, movingAverageEnd, 4)

fig.update_layout(title="Average Number of Events", height=800, width=600)
fig.update_layout(showlegend=False,   plot_bgcolor="#FFF",  font=dict(
        family="Arial",
        size=14
    )) 
fig.update_yaxes(title_text="Number of events", row=1, col=1, range=[0, 20], dtick=5, gridcolor=gridColor) # , range=[0, 50]
fig.update_yaxes(title_text="Number of events", row=2, col=1, range=[0, 20], dtick=5, gridcolor=gridColor) # , range=[0, 50]
fig.update_yaxes(title_text="Number of events", row=3, col=1, range=[0, 20], dtick=5, gridcolor=gridColor) # , range=[0, 50]
# fig.update_yaxes(title_text="End point", row=4, col=1)#, range=[0, 2.5], dtick=0.5) # , range=[0, 50]

# For Ratio
# fig.update_yaxes( row=1, col=2, range=[0, 2], dtick=0.5)
# fig.update_yaxes( row=2, col=2, range=[0, 2], dtick=0.5)
# fig.update_yaxes( row=3, col=2, range=[0, 2], dtick=0.5)

fig.update_xaxes(title_text="Time [s]", dtick=20, row=1, col=1, gridcolor=gridColor)
fig.update_xaxes(title_text="Time [s]", dtick=20, row=2, col=1, gridcolor=gridColor)
fig.update_xaxes(title_text="Time [s]", dtick=20, row=3, col=1, gridcolor=gridColor)

fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')
fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')
fig.show()


# this plots the different groups on the sample plot looking at only a single type of event
# x_labels = np.linspace(0,28,29)*10

# fig2 = make_subplots(
#     rows=5, cols=1,
#     row_heights=[1.0, 1.0, 1.0, 1.0, 1.0])


# addPlotsSeparate(fig2, x_labels, dfContr, dfPre, dfPost, dfEnd, 'fusion', 1, markerSize = 3)
# addPlotsSeparate(fig2, x_labels, dfContr, dfPre, dfPost, dfEnd, 'fission', 2, markerSize = 3)
# addPlotsSeparate(fig2, x_labels, dfContr, dfPre, dfPost, dfEnd, 'depolarisation', 3, markerSize = 3)
# addPlotsSeparate(fig2, x_labels, dfContr, dfPre, dfPost, dfEnd, 'average', 4, markerSize = 3)
# addPlotsSeparate(fig2, x_labels, dfContr, dfPre, dfPost, dfEnd, 'total', 5, markerSize = 3)

# fig2.update_layout(title="Precentage change from start", height=900, width=700)
# fig2.update_layout(showlegend=False,     font=dict(
#         family="Arial",
#         size=16
#     )) 
# fig2.update_yaxes(title_text="Fusion", row=1, col=1)#, range=[0.5, 2], dtick=0.5) # , range=[0, 50]
# fig2.update_yaxes(title_text="Fission", row=2, col=1)#, range=[0.5, 2], dtick=0.5) # , range=[0, 50]
# fig2.update_yaxes(title_text="Depolarisation", row=3, col=1)#, range=[0.5, 4], dtick=0.5) # , range=[0, 50]
# fig2.update_yaxes(title_text="Average Volume", row=4, col=1)#, range=[0.5, 2], dtick=0.5) # , range=[0, 50]
# fig2.update_yaxes(title_text="Total Structures", row=5, col=1)#, range=[0.5, 2], dtick=0.5) # , range=[0, 50]
# fig2.update_xaxes(title_text="Time [s]", row=5, col=1)#, range=[0, 50]
# fig2.show()



# export the combined data for external use
controlDF = pd.concat(controlList)
preTreatDF = pd.concat(preTreatList)
postTreatDF = pd.concat(postTreatList)

controlDF.to_csv("{}//{}Control.csv".format(inputPath, typeData))
preTreatDF.to_csv("{}//{}Pretreat.csv".format(inputPath, typeData))
postTreatDF.to_csv("{}//{}Posttreat.csv".format(inputPath, typeData))



dfControlPercentage = pd.DataFrame()
dfControlPercentage['fusion'] = dfContr['fusion']/dfContr['total']*100
dfControlPercentage['fission'] = dfContr['fission']/dfContr['total']*100
dfControlPercentage['depolarisation'] = dfContr['depolarisation']/dfContr['total']*100
movingAverageControlPercentage = dfControlPercentage.rolling(5, center=True).mean()

dfPrePercentage = pd.DataFrame()
dfPrePercentage['fusion'] = dfPre['fusion']/dfPre['total']*100
dfPrePercentage['fission'] = dfPre['fission']/dfPre['total']*100
dfPrePercentage['depolarisation'] = dfPre['depolarisation']/dfPre['total']*100
movingAveragePreTreatPercentage = dfPrePercentage.rolling(5, center=True).mean()

dfPostPercentage = pd.DataFrame()
dfPostPercentage['fusion'] = dfPost['fusion']/dfPost['total']*100
dfPostPercentage['fission'] = dfPost['fission']/dfPost['total']*100
dfPostPercentage['depolarisation'] = dfPost['depolarisation']/dfPost['total']*100
movingAveragePostTreatPercentage = dfPostPercentage.rolling(5, center=True).mean()



# This section of the code breaks the data into groups of 5 (stepSize) datapoints and compares them between groups with the T-test, plots the p values

# inputPath = "G:\\PhD\\MEL_2020_CHOSEN\\Output"
# inputPath = "C:\\RESEARCH\\MEL_2020_CHOSEN\\Output"

folders = [f for f in listdir(inputPath)]

averageFissionControl = []
averageFissionPreTreat = []
averageFissionPostTreat = []

stepSize = 5

typeColumn = ['fusion', 'fission', 'depolarisation', 'average', 'total']
for c in typeColumn:
    for folder in folders:
        try:
            print(folder)
            df = pd.read_csv("{}\\{}\\outcomesRelative.csv".format(inputPath, folder))  #outcomesRelative  outcomes
            fissionTemp = []        
            if(folder.startswith("Con")): # control
                for i in range(0, len(df[c].index), stepSize):
                    fissionTemp.append(np.average(df[c][i:i+stepSize]))
                averageFissionControl.append(fissionTemp)
            elif(folder.startswith("H2O2") and folder.endswith("1.czi")): # pre-treatment
                for i in range(0, len(df[c].index), stepSize):
                    fissionTemp.append(np.average(df[c][i:i+stepSize]))
                averageFissionPreTreat.append(fissionTemp)
            elif(folder.startswith("H2O2") and folder.endswith("2.czi")): # post-treatment
                for i in range(0, len(df[c].index), stepSize):
                    fissionTemp.append(np.average(df[c][i:i+stepSize]))
                averageFissionPostTreat.append(fissionTemp)
            else:
                print("Skipped", folder)
        except:
            print("SKIPPED " + folder)

    # (statC_PR, pC_PR) = scipy.stats.ttest_ind(averageFissionControl,averageFissionPreTreat, equal_var=False)
    # (statPR_PS, pPR_PS) = scipy.stats.ttest_ind(averageFissionPreTreat,averageFissionPostTreat, equal_var=False)
    # (statC_PS, pC_PS) = scipy.stats.ttest_ind(averageFissionControl,averageFissionPostTreat, equal_var=False)

    df = pd.DataFrame()
    df['pC_PR'] = pC_PR[1:]
    df['pPR_PS'] = pPR_PS[1:]
    df['pC_PS'] = pC_PS[1:]

    # dfAverage = pd.DataFrame()
    # dfAverage['averageControl'] = np.average(averageFissionControl, axis=0)
    # dfAverage['averagePre'] = np.average(averageFissionPreTreat, axis=0)
    # dfAverage['averagePost'] = np.average(averageFissionPostTreat, axis=0)
    import plotly.express as px
    fig = px.line(df, title=c)
    fig.show()