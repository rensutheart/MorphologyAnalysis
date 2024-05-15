# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:18:24 2020

@author: rensu
"""

import pandas as pd
import numpy as np
from os import listdir, path, makedirs
from os.path import isfile, join, isdir

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats, optimize, interpolate
import scipy

#path = "C:\\Users\\rensu\\Downloads\\RawData\\"
# path = "G:\\PhD\\MEL_2020_CHOSEN\\RawData(backup)\\"
#path = "C:\\RESEARCH\\MEL_2020_CHOSEN\\RawData(backup - Revision)\\"
path = "G:\\PhD\\MEL_2020_CHOSEN\\RawData(backup - Revision)\\"
# path = "G:\\PhD\\MEL_2020_CHOSEN\\ComparisonData\\"
# path = "G:\\PhD\\MEL_2020_CHOSEN\\HumanData2\\"
# path = "G:\\PhD\\MEL_2020_CHOSEN\\ComparisonData2\\"

files = [f for f in listdir(path)]
print(files)

controlData = pd.read_csv(path + files[0])
postTreatData = pd.read_csv(path + files[1])
preTreatData = pd.read_csv(path + files[2])

# if 'Unnamed: 0' in controlData:
#     controlData = controlData.drop('Unnamed: 0', 1)
# if 'Unnamed: 0' in postTreatData:
#     postTreatData = postTreatData.drop('Unnamed: 0', 1)
# if 'Unnamed: 0' in preTreatData:
#     preTreatData = preTreatData.drop('Unnamed: 0', 1)


# combinedData = pd.DataFrame()
# combinedData['Control Fission'] = controlData['1']
# combinedData['PreTreat Fission'] = preTreatData['1']
# combinedData['PostTreat Fission'] = postTreatData['1']

#combinedData.to_csv(path + 'fission.csv')

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

alpha = 0.05
labels = controlData.columns[2:]
indices = ['length','total','average', 'variance', 'stdDev', 'ci+','ci-'] # ,'t-test (pval)'     
types = ['Fusion C-Pr', 'Fusion Pr-Po', 'Fusion C-Po',
         'Fission C-Pr', 'Fission Pr-Po', 'Fission C-Po',
         'Depolarisation C-Pr', 'Depolarisation Pr-Po', 'Depolarisation C-Po',
         'Total C-Pr', 'Total Pr-Po', 'Total C-Po',
         'Avearge C-Pr', 'Avearge Pr-Po', 'Avearge C-Po']

rawTypes = ['Fusion Control', 'Fusion PreTreat', 'Fusion PostTreat',
            'Fission Control', 'Fission PreTreat', 'Fission PostTreat',
            'Depolarisation Control', 'Depolarisation PreTreat', 'Depolarisation PostTreat',
            'Total Control', 'Total PreTreat', 'Total PostTreat',
            'Avearge Control', 'Avearge PreTreat', 'Avearge PostTreat']

dfStats = pd.DataFrame(columns=indices, index=rawTypes)
dfT_Test = pd.DataFrame(columns=['t-test (pval)'], index=types)

count = 0
for label in labels:
    for typeIndex in range(0,3):
        if typeIndex == 0:
            X = controlData[label]
            Y = preTreatData[label]
            pVal = scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1] # check for equal variance
        elif typeIndex == 1:
            X = preTreatData[label]
            Y = postTreatData[label]
            pVal = scipy.stats.ttest_rel(X,Y)[1] # check for equal variance
        else:
            X = controlData[label]
            Y = postTreatData[label]
            pVal = scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1] # check for equal variance

        
        dfT_Test.at[types[count],'t-test (pval)'] = pVal

        count += 1

#calculate main stats
data = [controlData, preTreatData, postTreatData]
count = 0
for label in labels:
    for typeIndex in range(0,3):
        X = data[typeIndex][label]

        varX = np.var(X, ddof=1)
        stdX = np.std(X, ddof=1)
        mX, c1X, c2X = mean_confidence_interval(X)
        
        dfStats.at[rawTypes[count], indices[0]] = len(X)
        dfStats.at[rawTypes[count], indices[1]] = np.sum(X)
        dfStats.at[rawTypes[count], indices[2]] = mX
        dfStats.at[rawTypes[count], indices[3]] = varX
        dfStats.at[rawTypes[count], indices[4]] = stdX
        dfStats.at[rawTypes[count], indices[5]] = c1X - mX
        dfStats.at[rawTypes[count], indices[6]] = mX - c2X
        
        count += 1
        



def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    print(val)
    color = 'red' if val < 0.05 else 'black'
    return 'color: %s;' % color

print(dfStats)
print(dfT_Test)

dfFusion = dfStats.loc['Fusion Control':'Fusion PostTreat']
dfFission = dfStats.loc['Fission Control':'Fission PostTreat']
dfDepolarisation = dfStats.loc['Depolarisation Control':'Depolarisation PostTreat']
dfTotal = dfStats.loc['Total Control':'Total PostTreat']
dfAverage = dfStats.loc['Avearge Control':'Avearge PostTreat']

listDF = [dfFusion,dfFission,dfDepolarisation,dfTotal,dfAverage]

import plotly.express as px
px.colors.qualitative.Plotly
# fig = px.scatter(dfStats, y="average", error_y="ci+", error_y_minus="ci-")
# fig.show()

fig = make_subplots(
    rows=4, cols=1,
    row_heights=[0.3, 0.3, 0.3, 0.3])#,
     #subplot_titles=(['Mean and confidence interval']))

gridColor = 'lightgray'
labels = ['Control', 'Pre-Treatment', 'Post-Treatment']
typeHeadings = ['Number of events', 'Average number <br> of events', 'Total Structures', 'Average structure volume <br> [voxels]']
name = ['Fission', 'Fusion', 'Depolarisation', 'Total Structures', 'Average structure volume <br> [voxels]']
plotlyColors = px.colors.qualitative.Plotly
colors = [plotlyColors[2], plotlyColors[1], plotlyColors[0], plotlyColors[3], plotlyColors[4]]
row = 1
count = 0
typeEvents = ['Fusion', 'Fission', 'Depolarisation']

for df in listDF:
    fig.add_trace(go.Scatter(
            x=labels,y=df['average'],
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=df['ci+'],
                arrayminus=df['ci-'],
                symmetric=False,
                visible=True),
                line=dict(color=colors[count]), #, width=0
            name=name[count]
        ),row=row, col=1)
    
    fig.update_yaxes(title_text=typeHeadings[row-1], row=row, col=1 )#, dtick=2, range=[0, 15])
    count += 1
    if(count >= 3):
        if(row == 1):
            row += 1
        row += 1



fig.add_trace(go.Bar(
    x=typeEvents,
    y=[dfFusion['average'][0],dfFission['average'][0],dfDepolarisation['average'][0]],
    name='Control',
    marker_color='#0d0887'
),row=2, col=1)
fig.add_trace(go.Bar(
    x=typeEvents,
    y=[dfFusion['average'][1],dfFission['average'][1],dfDepolarisation['average'][1]],
    name='Pre-Treatment',
    marker_color='#bd3786'
),row=2, col=1)
fig.add_trace(go.Bar(
    x=typeEvents,
    y=[dfFusion['average'][2],dfFission['average'][2],dfDepolarisation['average'][2]],
    name='Post-Treatment',
    marker_color='#fdca26'
),row=2, col=1)
fig.update_yaxes(title_text=typeHeadings[1], row=2, col=1 , dtick=2, range=[0, 15])

fig.update_layout(title='Interval plots', height=900, width=450, barmode='group')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside',  gridcolor=gridColor)
fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside',  gridcolor=gridColor)
fig.update_yaxes(dtick=50, row=4, col=1)
fig.update_layout(showlegend=False, plot_bgcolor="#FFF",   font=dict(
        family="Arial",
        size=16
    )) 
fig.show()



#WHEN CONSIDERING THE TIMELAPSE AS SEVERAL DUPLICATE MEASUREMENTS OF THE SAME. THEREFORE N = 5 and not 203…
# # AVERAGE
# controlTimelapseAverages = []
# preTimelapseAverages = []
# postTimelapseAverages = []

# count = 0
# currentAverage = 0
# eventType = '0'
# for line in controlData[eventType]:
#     # print(line)
#     currentAverage += line
#     count += 1

#     if count > 28:
#         # print("APPEND")
#         controlTimelapseAverages.append(currentAverage/29)
#         currentAverage = 0
#         count = 0

# count = 0
# currentAverage = 0
# for line in preTreatData[eventType]:
#     # print(line)
#     currentAverage += line
#     count += 1

#     if count > 28:
#         # print("APPEND")
#         preTimelapseAverages.append(currentAverage/29)
#         currentAverage = 0
#         count = 0

# count = 0
# currentAverage = 0
# for line in postTreatData[eventType]:
#     # print(line)
#     currentAverage += line
#     count += 1

#     if count > 28:
#         # print("APPEND")
#         postTimelapseAverages.append(currentAverage/29)
#         currentAverage = 0
#         count = 0


# X = controlTimelapseAverages
# Y = postTimelapseAverages
# pValCPo = scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1] # check for equal variance

# X = controlTimelapseAverages
# Y = preTimelapseAverages
# pValCPr = scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1] # check for equal variance

# X = preTimelapseAverages
# Y = postTimelapseAverages
# pValPrPo = scipy.stats.ttest_rel(X,Y)[1] # check for equal variance

# print("CPo: ", pValCPo)
# print("CPr: ", pValCPr)
# print("PrPO: ", pValPrPo)





# # MAX
# controlTimelapseAverages = []
# preTimelapseAverages = []
# postTimelapseAverages = []

# count = 0
# currentAverage = 0
# eventType = '2'
# for line in controlData[eventType]:
#     # print(line)
#     currentAverage = max(currentAverage, line)
#     count += 1

#     if count > 28:
#         # print("APPEND")
#         controlTimelapseAverages.append(currentAverage)
#         currentAverage = 0
#         count = 0

# count = 0
# currentAverage = 0
# for line in preTreatData[eventType]:
#     # print(line)
#     currentAverage = max(currentAverage, line)
#     count += 1

#     if count > 28:
#         # print("APPEND")
#         preTimelapseAverages.append(currentAverage)
#         currentAverage = 0
#         count = 0

# count = 0
# currentAverage = 0
# for line in postTreatData[eventType]:
#     # print(line)
#     currentAverage = max(currentAverage, line)
#     count += 1

#     if count > 28:
#         # print("APPEND")
#         postTimelapseAverages.append(currentAverage)
#         currentAverage = 0
#         count = 0


# X = controlTimelapseAverages
# Y = postTimelapseAverages
# pValCPo = scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1] # check for equal variance

# X = controlTimelapseAverages
# Y = preTimelapseAverages
# pValCPr = scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1] # check for equal variance

# X = preTimelapseAverages
# Y = postTimelapseAverages
# pValPrPo = scipy.stats.ttest_rel(X,Y)[1] # check for equal variance

# print("CPo: ", pValCPo)
# print("CPr: ", pValCPr)
# print("PrPO: ", pValPrPo)


# # Area under curve
# import numpy as np
# from scipy.integrate import simps
# from numpy import trapz
# from sklearn.linear_model import LinearRegression


# controlTimelapseAverages = []
# preTimelapseAverages = []
# postTimelapseAverages = []

# count = 0
# tempList = []
# eventType = '2'
# for line in controlData[eventType]:
#     # print(line)
#     tempList.append(line)
#     count += 1

#     if count > 28:
#         # print("APPEND")
#         controlTimelapseAverages.append(tempList)
#         tempList = []
#         count = 0

# count = 0
# tempList = []
# for line in preTreatData[eventType]:
#     # print(line)
#     tempList.append(line)
#     count += 1

#     if count > 28:
#         # print("APPEND")
#         preTimelapseAverages.append(tempList)
#         tempList = []
#         count = 0

# count = 0
# tempList = []
# for line in postTreatData[eventType]:
#     # print(line)
#     tempList.append(line)
#     count += 1

#     if count > 28:
#         # print("APPEND")
#         postTimelapseAverages.append(tempList)
#         tempList = []
#         count = 0


# X = trapz(controlTimelapseAverages, dx=10)
# Y = trapz(postTimelapseAverages, dx=10)
# pValCPo = scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1] # check for equal variance

# X = trapz(controlTimelapseAverages, dx=10)
# Y = trapz(preTimelapseAverages, dx=10)
# pValCPr = scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1] # check for equal variance

# X = trapz(preTimelapseAverages, dx=10)
# Y = trapz(postTimelapseAverages, dx=10)
# pValPrPo = scipy.stats.ttest_rel(X,Y)[1] # check for equal variance

# print("CPo: ", pValCPo)
# print("CPr: ", pValCPr)
# print("PrPO: ", pValPrPo)



# # gradient of linear regression trend line
# x_labels = (np.linspace(0,28,29)*10).reshape((-1, 1))

# controlSlopes = []
# preTreatSlopes = []
# postTreatSlopes = []

# for sample in controlTimelapseAverages:
#     model = LinearRegression()
#     model.fit(x_labels,sample)
#     controlSlopes.append(model.coef_[0])
    
# for sample in preTimelapseAverages:
#     model = LinearRegression()
#     model.fit(x_labels,sample)
#     preTreatSlopes.append(model.coef_[0])

# for sample in postTimelapseAverages:
#     model = LinearRegression()
#     model.fit(x_labels,sample)
#     postTreatSlopes.append(model.coef_[0])


# X = controlSlopes
# Y = postTreatSlopes
# pValCPo = scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1] # check for equal variance

# X = controlSlopes
# Y = preTreatSlopes
# pValCPr = scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1] # check for equal variance

# X = preTreatSlopes
# Y = postTreatSlopes
# pValPrPo = scipy.stats.ttest_rel(X,Y)[1] # check for equal variance

# print("CPo: ", pValCPo)
# print("CPr: ", pValCPr)
# print("PrPO: ", pValPrPo)






# At each time point in the time lapse
controlTimelapseAverages = []
preTimelapseAverages = []
postTimelapseAverages = []

count = 0
tempList = []
eventType = '2'
for line in controlData[eventType]:
    # print(line)
    tempList.append(line)
    count += 1

    if count > 28:
        # print("APPEND")
        controlTimelapseAverages.append(tempList)
        tempList = []
        count = 0

count = 0
tempList = []
for line in preTreatData[eventType]:
    # print(line)
    tempList.append(line)
    count += 1

    if count > 28:
        # print("APPEND")
        preTimelapseAverages.append(tempList)
        tempList = []
        count = 0

count = 0
tempList = []
for line in postTreatData[eventType]:
    # print(line)
    tempList.append(line)
    count += 1

    if count > 28:
        # print("APPEND")
        postTimelapseAverages.append(tempList)
        tempList = []
        count = 0


controlSummary = np.array(controlTimelapseAverages)
preSummary = np.array(preTimelapseAverages)
postSummary = np.array(postTimelapseAverages)

CPo_pvals = []
CPr_pvals = []
PrPo_pvals = []

for i in range(0,29):
    X = controlSummary[:,i]
    Y = postSummary[:,i]
    CPo_pvals.append(scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1]) # check for equal variance

    X = controlSummary[:,i]
    Y = preSummary[:,i]
    CPr_pvals.append(scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1]) # check for equal variance

    X = preSummary[:,i]
    Y = postSummary[:,i]
    PrPo_pvals.append(scipy.stats.ttest_rel(X,Y)[1]) # check for equal variance

print("CPo: ", np.average(CPo_pvals))
print("CPr: ", np.average(CPr_pvals))
print("PrPO: ", np.average(PrPo_pvals))


x_labels = np.linspace(0,28,29)*10

import pandas as pd
df = pd.DataFrame()
df["CPo_pvals"] = CPo_pvals
df["CPr_pvals"] = CPr_pvals
df["PrPo_pvals"] = PrPo_pvals

movingAverage = df.rolling(5, center=True).mean()
import plotly.express as px

fig = px.line(movingAverage)
fig.show()


# THIS IS THE TOTALS PER CELL
# columnNames = ['control', 'pre-treat', 'post-treat']
# fusionTotals = pd.DataFrame(columns=columnNames)
# fissionTotals = pd.DataFrame(columns=columnNames)
# depTotals = pd.DataFrame(columns=columnNames)


controlFusion = [243,179,284,369]
controlFission = [223,130,177,309]
controlDep = [59,66,33,108]

preFusion = [147,365,209,375]
preFission = [116,330,196,332]
preDep = [75,45,49,63]

postFusion = [145,243,252,268]
postFission = [113,198,241,209]
postDep = [102,92,89,83]

X = preFission
Y = postDep

print(scipy.stats.ttest_ind(X,Y, equal_var=(scipy.stats.bartlett(X,Y)[1] > alpha))[1]) # this must be used for C-Pr and C-Po
print(scipy.stats.ttest_rel(X,Y)[1]) # this must be used for Pre-Post


# Calcuate cumalative totals
headings = ['0', '1', '2']
cellsConList = []
avConList = []
for h in headings:
    cellA_Control = pd.DataFrame(controlData[h][0:29].values, columns=["A"])
    cellB_Control = pd.DataFrame(controlData[h][29:29*2].values, columns=["B"])
    cellC_Control = pd.DataFrame(controlData[h][29*2:29*3].values, columns=["C"])
    cellD_Control = pd.DataFrame(controlData[h][29*3:29*4].values, columns=["D"])

    cellsControl = pd.concat([cellA_Control,cellB_Control,cellC_Control,cellD_Control], axis=1)
    cellsConList.append(cellsControl.cumsum())

    # fig = px.line(cellsControl.cumsum(), title='Control ' + h)
    # fig.show()

    cellsConAv = (cellA_Control.cumsum().values + 
                    cellB_Control.cumsum().values +
                    cellC_Control.cumsum().values +
                    cellD_Control.cumsum().values)/4
    # fig2 = px.line(cellsConAv, title='Pre ave ' + h)
    # fig2.show()

    avConList.append(cellsConAv)

cellsPreList = []
avPreList = []
for h in headings:
    cellA_Pre = pd.DataFrame(preTreatData[h][0:29].values, columns=["A"])
    cellB_Pre = pd.DataFrame(preTreatData[h][29:29*2].values, columns=["B"])
    cellC_Pre = pd.DataFrame(preTreatData[h][29*2:29*3].values, columns=["C"])
    cellD_Pre = pd.DataFrame(preTreatData[h][29*3:29*4].values, columns=["D"])

    cellsPre = pd.concat([cellA_Pre,cellB_Pre,cellC_Pre,cellD_Pre], axis=1)
    cellsPreList.append(cellsPre.cumsum())

    # fig = px.line(cellsPre.cumsum(), title='Pre ' + h)
    # fig.show()

    cellsPreAv = (cellA_Pre.cumsum().values + 
                    cellB_Pre.cumsum().values +
                    cellC_Pre.cumsum().values +
                    cellD_Pre.cumsum().values)/4
    # fig2 = px.line(cellsPreAv, title='Pre ave ' + h)
    # fig2.show()

    avPreList.append(cellsPreAv)


cellsPostList = []
avPostList = []
for h in headings:
    cellA_Post = pd.DataFrame(postTreatData[h][0:29].values, columns=["A"])
    cellB_Post = pd.DataFrame(postTreatData[h][29:29*2].values, columns=["B"])
    cellC_Post = pd.DataFrame(postTreatData[h][29*2:29*3].values, columns=["C"])
    cellD_Post = pd.DataFrame(postTreatData[h][29*3:29*4].values, columns=["D"])

    cellsPost = pd.concat([cellA_Post,cellB_Post,cellC_Post,cellD_Post], axis=1)    
    cellsPostList.append(cellsPost.cumsum())
    # fig = px.line(cellsPost.cumsum(), title='Post ' + h)
    # fig.show()

    cellsPostAv = (cellA_Post.cumsum().values + 
                    cellB_Post.cumsum().values +
                    cellC_Post.cumsum().values +
                    cellD_Post.cumsum().values)/4
    # fig2 = px.line(cellsPostAv, title='Post ave ' + h)
    # fig2.show()

    avPostList.append(cellsPostAv)


avFusion = pd.DataFrame({'Con':avConList[0].ravel(), 'Pre':avPreList[0].ravel(), 'Post':avPostList[0].ravel()})
fig3 = px.line(avFusion, title='Fusion ave compare')
fig3.show()


avFission = pd.DataFrame({'Con':avConList[1].ravel(), 'Pre':avPreList[1].ravel(), 'Post':avPostList[1].ravel()})
fig3 = px.line(avFission, title='Fission ave compare')
fig3.show()


avDepolarisation = pd.DataFrame({'Con':avConList[2].ravel(), 'Pre':avPreList[2].ravel(), 'Post':avPostList[2].ravel()})
fig3 = px.line(avDepolarisation, title='Depolarisation ave compare')
fig3.show()



x_labels = np.linspace(0,280,29)
fig = make_subplots(
    rows=3, cols=1,
    row_heights=[1.0, 1.0, 1.0],
    subplot_titles=('Commulative sum of fusion events', 'Commulative sum of fission events', 'Commulative sum of depolarisation events'))

for eventType in range(0,3):

    mXListC = [] 
    c1XList = []
    c2XList = []
    for i in range(0,29):
        mX, c1X, c2X = mean_confidence_interval(cellsConList[eventType].loc[i].values,0.8)
        mXListC.append(mX)
        c1XList.append(c1X)
        c2XList.append(c2X)


    fig.add_trace(go.Scatter(x=x_labels, y=c1XList, fill=None, fillcolor='rgba(13, 8, 135,0.2)', line=dict(width=1.0, color='#0d0887')), row=eventType+1, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=c2XList, fill='tonexty', fillcolor='rgba(13, 8, 135,0.2)', line=dict(width=1.0, color='#0d0887')), row=eventType+1, col=1)
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsConList[eventType]['A'],mode='markers',  marker=dict(size=2, color='#0d0887')))
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsConList[eventType]['B'],mode='markers',  marker=dict(size=2,color='#0d0887')))
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsConList[eventType]['C'],mode='markers',  marker=dict(size=2,color='#0d0887')))
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsConList[eventType]['D'],mode='markers',  marker=dict(size=2,color='#0d0887')))





    mXListPr = [] 
    c1XList = []
    c2XList = []
    for i in range(0,29):
        mX, c1X, c2X = mean_confidence_interval(cellsPreList[eventType].loc[i].values,0.8)
        mXListPr.append(mX)
        c1XList.append(c1X)
        c2XList.append(c2X)


    fig.add_trace(go.Scatter(x=x_labels, y=c1XList, fill=None, fillcolor='rgba(189, 55, 134,0.2)', line=dict(width=1.0, color='#bd3786')), row=eventType+1, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=c2XList, fill='tonexty', fillcolor='rgba(189, 55, 134,0.2)', line=dict(width=1.0, color='#bd3786')), row=eventType+1, col=1)
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsPreList[eventType]['A'],mode='markers',  marker=dict(size=2, color='#bd3786')))
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsPreList[eventType]['B'],mode='markers',  marker=dict(size=2,color='#bd3786')))
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsPreList[eventType]['C'],mode='markers',  marker=dict(size=2,color='#bd3786')))
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsPreList[eventType]['D'],mode='markers',  marker=dict(size=2,color='#bd3786')))





    mXListPo = [] 
    c1XList = []
    c2XList = []
    for i in range(0,29):
        mX, c1X, c2X = mean_confidence_interval(cellsPostList[eventType].loc[i].values,0.8)
        mXListPo.append(mX)
        c1XList.append(c1X)
        c2XList.append(c2X)


    fig.add_trace(go.Scatter(x=x_labels, y=c1XList, fill=None, fillcolor='rgba(253, 202, 38,0.2)', line=dict(width=1.0, color='#fdca26')), row=eventType+1, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=c2XList, fill='tonexty', fillcolor='rgba(253, 202, 38,0.2)', line=dict(width=1.0, color='#fdca26')), row=eventType+1, col=1)
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsPostList[eventType]['A'],mode='markers',  marker=dict(size=2, color='#fdca26')))
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsPostList[eventType]['B'],mode='markers',  marker=dict(size=2,color='#fdca26')))
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsPostList[eventType]['C'],mode='markers',  marker=dict(size=2,color='#fdca26')))
    # fig.add_trace(go.Scatter(x=x_labels,y=cellsPostList[eventType]['D'],mode='markers',  marker=dict(size=2,color='#fdca26')))



    fig.add_trace(go.Scatter(x=x_labels, y=mXListC, mode='lines', fill=None, line=dict(width=3.0, color='#0d0887')), row=eventType+1, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=mXListPr, mode='lines', fill=None, line=dict(width=3.0, color='#bd3786')), row=eventType+1, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=mXListPo, mode='lines', fill=None, line=dict(width=3.0, color='#fdca26')), row=eventType+1, col=1)

    if eventType == 2:
        fig.update_yaxes(title_text="Total", range=[0, 140], dtick=20, row=eventType+1, col=1) #range=[0, 120], dtick=20)
    else:
        fig.update_yaxes(title_text="Total", range=[0, 400], dtick=50, row=eventType+1, col=1) #range=[0, 120], dtick=20)
    
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside',  gridcolor=gridColor, row=eventType+1, col=1)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside',  gridcolor=gridColor, row=eventType+1, col=1)
    fig.update_xaxes(title_text="Time [s]", dtick=20, gridcolor='lightgray', row=eventType+1, col=1)

fig.update_layout(title='Commulative sum of events', height=800, width=600)
fig.update_layout(showlegend=False,   plot_bgcolor="#FFF",  font=dict(
        family="Arial",
        size=14
    )) 


fig.show()



