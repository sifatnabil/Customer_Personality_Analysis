import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(x, 
              centers=None, 
              belongto=None, 
              color_list=list([]),
              ax=plt,
              y_label="feature_2",
              x_label="feature_1",
              title="KMeans"):
    ax.scatter(
        x[:, 0],
        x[:, 1],
        s=5,
        lw=10,
        c=belongto if belongto is not None else "black",
        cmap=matplotlib.colors.ListedColormap(color_list)
    )

    try:
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.title.set_text(title)
    except:
        ax.ylabel(y_label)
        ax.xlabel(x_label)
        ax.title(title)
    
    if centers is not None:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="X",
            c=color_list,
            s=200,
            edgecolors="black",
        )

def plot_cluster_3d(x, y, z, cluster_labels):
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x, y, z, c=cluster_labels, cmap="viridis")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()

def create_kdeplot(col, title, clus0, clus1, clus2):
    plt.figure(figsize=(12,8))
    # sns.kdeplot(data=outlier, x=col, label='Outliers')
    sns.kdeplot(data=clus0, x=col, label ='Cluster 0')
    sns.kdeplot(data=clus1, x=col, label ='Cluster 1')
    sns.kdeplot(data=clus2, x=col, label ='Cluster 2')
    plt.title(title)
    plt.legend()
    plt.show()
    
#Create function for plotting count of discrete values per cluster
def create_barplot(col, title, data):
    plt.figure(figsize=(8,8))
    sns.countplot(x=data[col], hue=data["Cluster"])
    plt.title(title)
    plt.legend(['Outliers', 'Cluster 0', 'Cluster 1', 'Cluster 2'])
    plt.show()