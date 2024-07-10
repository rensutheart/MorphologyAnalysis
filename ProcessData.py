'''
LOAD DATA
'''
import os
import pandas as pd
pd.options.plotting.backend = "plotly"

# path = "C:/RESEARCH/Sholto_Data/Data_output/"
basePath = "C:/RESEARCH/Sholto_Data_N2_N3/N2/"
path = basePath + "Data_output/"
# out_file_name = "start_data.csv" #"middle_data.csv"
# startFrame = 0
# n = 2

out_file_name = "middle_data.csv"
startFrame = 10
n = 2

groupsNames = os.listdir(path)
groupDataFiles = {}

for group in groupsNames:
    groupDataFiles[group] = os.listdir(path + group)
    

groupDataFramesCollection = {}
for group in groupsNames:
    groupDataFrames = []
    for file in groupDataFiles[group]:
        groupDataFrames.append(pd.read_csv(path + group + "/" + file))
    groupDataFramesCollection[group] = groupDataFrames

'''
REPLACE IMAGE NAME WITH TIME
'''
for group in groupsNames:
    for df in groupDataFramesCollection[group]:
        try:
            df.rename(columns={'Image Name': 'Time[s]'}, inplace=True)
        except:
            print("Could not change Image Name column. Possibly already changed")
        try:
            df['Time[s]'] = range(0, df.shape[0]*10, 10) 
        except:
            print("Could not change Time column data")

'''
EXTRACT FIRST n SAMPLES FROM THE DF (the greater the n the more the treatment or photo toxicisty plays a role)
'''

reducedGroupDataFrameCollection = {}
for group in groupsNames:
    print('\n')
    # print(group)
    for df in groupDataFramesCollection[group]:
        df['Group'] = group
        # print(df.iloc[0:n])
        if group in reducedGroupDataFrameCollection.keys():
            reducedGroupDataFrameCollection[group] = pd.concat([reducedGroupDataFrameCollection[group], df.iloc[startFrame:startFrame+n]], axis=0, ignore_index=True)
        else:
            reducedGroupDataFrameCollection[group] = df.iloc[startFrame:startFrame+n]

'''
GROUP THE SAME PARAMETER FOR DIFFERENT GROUPS IN SAME dictionary
'''
# columnsNames = df.columns[1:] # I'm parasitically using df even though it is outside of scope
# groupedResults = {key: None for key in columnsNames}
# for col in columnsNames:
#     tempDict = {key: None for key in groupsNames}
#     for group in groupsNames:
#         tempDict[group] = list(reducedGroupDataFrameCollection[group][col])
#     #print(tempDict)
#     groupedResults[col] = tempDict # pd.DataFrame(tempDict)


# '''
# Plot
# '''
# import plotly.express as px

# output_path = 'C:/RESEARCH/Sholto_Data_N2_N3/N2/Plots/'

# for col in columnsNames:
#     fig = px.box(combinedDataFrame, x="Group", y=col)
#     # fig.show()
#     fig.write_image(output_path + col.replace('/','-') + '.png')


'''
ALTERNATIVE: Create 1 dataframe with an extra field to denote the type
'''
combinedDataFrame = pd.concat(reducedGroupDataFrameCollection, ignore_index=True)

output_path = basePath + '/Plots/CSV/'
combinedDataFrame.to_csv(output_path + out_file_name)


