import matplotlib
import matplotlib.pyplot as plt

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