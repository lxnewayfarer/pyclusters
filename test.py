from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

data = [[1, 1], [2, 2], [3, 1], [10, 14], [11, 14], [13, 12],
        [9, 11], [2, 3], [11, 12], [1, 12], [2, 14], [3, 13], [5, 8]]

x_axis = list(x[0] for x in data)
y_axis = list(x[1] for x in data)
plt.scatter(x_axis, y_axis)
for i, txt in enumerate(data):
    plt.annotate(i, (x_axis[i], y_axis[i]), xytext=(10,10), textcoords='offset points')
    
linked = linkage(data, 'single')
print('Distances between merged clusters:\n', linked[:, 2])

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top',
           distance_sort='descending', show_leaf_counts=True)
# plt.axhline(y=4, c='k')

max_d = 4
# by clusters: fcluster(linked, k, criterion='maxclust')
clusters = fcluster(linked, max_d, criterion='distance')
print('Dots to clusters: ', clusters)

plt.figure(figsize=(10, 7))
plt.scatter(x_axis, y_axis, c=clusters, cmap='prism', label=list(x for x in range(len(data))))
for i, txt in enumerate(data):
    plt.annotate(i, (x_axis[i], y_axis[i]), xytext=(10,10), textcoords='offset points')

plt.show()
