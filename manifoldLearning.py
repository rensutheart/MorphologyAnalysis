# Authors: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2011

print(__doc__)
from time import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

allTogether = True
index = 0

if allTogether:
    basePaths = ["C:/RESEARCH/Sholto_Data/", "C:/RESEARCH/Sholto_Data_N2_N3/N2/", "C:/RESEARCH/Sholto_Data_N2_N3/N3/"]
    output_path = basePaths[0] + 'Plots/'
    combinedDataFrame = pd.DataFrame()
    for basePath in basePaths:
        path = basePath + "Plots/CSV/"

        
        save = True

        df = pd.read_csv(
             filepath_or_buffer=path + "start_data.csv",  #"middle_data.csv" "start_data.csv"
            #filepath_or_buffer=path + "middle_data.csv",
            #header=None, 
            sep=',')

        df.dropna(how="all", inplace=True) # drops the empty line at file-end
        df.tail()
        if(combinedDataFrame.columns.size == 0):
            combinedDataFrame = df
        else:
            combinedDataFrame = combinedDataFrame.append(df, sort=False)

    dfSelected = combinedDataFrame.loc[:,'Count':'Mean Branch Diameter']
    X = np.copy(dfSelected)  

    y = np.copy(combinedDataFrame.loc[:,'Group'])
    labels = ['B', 'C+B', 'Ct','M+C+B', 'M']

else:
    basePaths = ["C:/RESEARCH/Sholto_Data", "C:/RESEARCH/Sholto_Data_N2_N3/N2/", "C:/RESEARCH/Sholto_Data_N2_N3/N3/"]
    path = basePaths[index] + "Plots/CSV/"

    output_path = basePaths[index] + 'Plots/'
    save = True

    df = pd.read_csv(
        # filepath_or_buffer=path + "start_data.csv",  #"middle_data.csv" "start_data.csv"
        filepath_or_buffer=path + "middle_data.csv",
        #header=None, 
        sep=',')

    df.dropna(how="all", inplace=True) # drops the empty line at file-end
    df.tail()

    dfSelected = df.loc[:,'Count':'Mean Branch Diameter']
    X = np.copy(dfSelected)  

    y = np.copy(df.loc[:,'Group'])
    labels = ['B', 'C+B', 'Ct','M+C+B', 'M']




# normalize data
from sklearn import preprocessing
data_scaled = pd.DataFrame(preprocessing.scale(dfSelected),columns = dfSelected.columns) 
X = np.copy(data_scaled)


#digits = datasets.load_digits(n_class=6)
#X = digits.data
#y = digits.target

n_samples, n_features = X.shape
n_neighbors = 10


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    labelsUnique = np.unique(y)
    # TODO: This should not be hardcoded for generalisation
    labelColor = {labelsUnique[0]:0, labelsUnique[1]:0.2, labelsUnique[2]:0.4, labelsUnique[3]:0.6, labelsUnique[4]:0.8}
    labelNames = {"Baf": labels[0], "CCCP + Baf": labels[1], "Control": labels[2], "Metf+CCCP+Baf": labels[3], "Metformin":labels[4]}

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], labelNames[y[i]],
                 color=plt.cm.Set2(labelColor[y[i]]),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            '''
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
            '''
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    if save:
        plt.savefig(output_path + title + ".png")


#----------------------------------------------------------------------
# Plot images
'''
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 64-dimensional digits dataset')
'''


#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection")


#----------------------------------------------------------------------
# Projection on to the first 2 principal components

print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(X_pca,
               "Principal Components projection (time %.2fs)" %
               (time() - t0))

#----------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components

print("Computing Linear Discriminant Analysis projection")
X2 = X.copy()
X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
t0 = time()
LDA_obj = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
X_lda = LDA_obj.fit_transform(X2, y)


fields = list(dfSelected.columns)
print("Main fields distinguishing classes")
scalings = np.abs(LDA_obj.scalings_)
print("\n  Class 1:")
for i, fieldIndex in enumerate(np.flip(np.argsort(scalings[:,0])), start=1):
    print("    {}. {} - {:.2f}".format(i, fields[fieldIndex], scalings[fieldIndex, 0]))

print("\n  Class 2:")
for i, fieldIndex in enumerate(np.flip(np.argsort(scalings[:,1])), start=1):
    print("    {}. {} - {:.2f}".format(i, fields[fieldIndex], scalings[fieldIndex, 1]))

print("\n  Class 3")
for i, fieldIndex in enumerate(np.flip(np.argsort(scalings[:,2])), start=1):
    print("    {}. {} - {:.2f}".format(i, fields[fieldIndex], scalings[fieldIndex, 2]))

print("\n  Class 4:")
for i, fieldIndex in enumerate(np.flip(np.argsort(scalings[:,3])), start=1):
    print("    {}. {} - {:.2f}".format(i, fields[fieldIndex], scalings[fieldIndex, 3]))


plot_embedding(X_lda,
               "Linear Discriminant projection (time %.2fs)" %
               (time() - t0))



#----------------------------------------------------------------------
# Isomap projection dataset
# print("Computing Isomap embedding")
# t0 = time()
# X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
# print("Done.")
# plot_embedding(X_iso,
#                "Isomap projection (time %.2fs)" %
#                (time() - t0))


#----------------------------------------------------------------------
# Locally linear embedding dataset
# print("Computing LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='standard')
# t0 = time()
# X_lle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_lle,
#                "Locally Linear Embedding (time %.2fs)" %
#                (time() - t0))



#----------------------------------------------------------------------
# Modified Locally linear embedding dataset
# print("Computing modified LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='modified')
# t0 = time()
# X_mlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_mlle,
#                "Modified Locally Linear Embedding (time %.2fs)" %
#                (time() - t0))


#----------------------------------------------------------------------
# HLLE embedding dataset
# print("Computing Hessian LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='hessian')
# t0 = time()
# X_hlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_hlle,
#                "Hessian Locally Linear Embedding (time %.2fs)" %
#                (time() - t0))


#----------------------------------------------------------------------
# LTSA embedding dataset
# print("Computing LTSA embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='ltsa')
# t0 = time()
# X_ltsa = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_ltsa,
#                "Local Tangent Space Alignment (time %.2fs)" %
#                (time() - t0))


#----------------------------------------------------------------------
# MDS  embedding dataset
# print("Computing MDS embedding")
# clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
# t0 = time()
# X_mds = clf.fit_transform(X)
# print("Done. Stress: %f" % clf.stress_)
# plot_embedding(X_mds,
#                "MDS embedding (time %.2fs)" %
#                (time() - t0))


#----------------------------------------------------------------------
# Random Trees embedding dataset
# print("Computing Totally Random Trees embedding")
# hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
#                                        max_depth=5)
# t0 = time()
# X_transformed = hasher.fit_transform(X)
# pca = decomposition.TruncatedSVD(n_components=2)
# X_reduced = pca.fit_transform(X_transformed)

# plot_embedding(X_reduced,
#                "Random forest embedding (time %.2fs)" %
#                (time() - t0))

#----------------------------------------------------------------------
# Spectral embedding dataset
# print("Computing Spectral embedding")
# embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
#                                       eigen_solver="arpack")
# t0 = time()
# X_se = embedder.fit_transform(X)

# plot_embedding(X_se,
#                "Spectral embedding (time %.2fs)" %
#                (time() - t0))

#----------------------------------------------------------------------
# t-SNE embedding dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, perplexity=10, init='random', random_state=0, verbose=1, learning_rate=200, n_iter=10000)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding (time %.2fs)" %
               (time() - t0))





# plt.show()
