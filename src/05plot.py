from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import umap

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import numpy as np
import click
import os

@click.command()
@click.option('--input_path', default='./data3')
@click.option('--resample', type=int, default=None)
@click.option('--output_path', default='./')
def plot(input_path, resample, output_path):
    x=np.load(input_path+"/data_x.npy")
    y=np.load(input_path+"/data_y.npy")
    ###
    if len(x.shape)==3:
        x=x[:,10:110,:]
        n=x.shape[0]
        x=x.reshape((n,-1))
    ### umap
    print("... UMAP")
    start_time = time.time()
    if resample is not None:
        idx=list(range(x.shape[0]))
        np.random.shuffle(idx)
        idx=idx[:10000]
        X=x[idx,:]
        Y=y[idx]
    else:
        X=x
        Y=y
    
    X=normalize(X)
    embedding = umap.UMAP().fit_transform(X)
    interval = time.time() - start_time

    os.makedirs(output_path,exist_ok=True)
    np.save(output_path+"/umap.npy",embedding)
    print("umap time:",interval)
    plt.scatter(embedding[:,0],embedding[:,1],c=Y,cmap=cm.gist_rainbow,alpha=0.3,marker=".")
    plt.colorbar()
    plt.title("umap")
    plt.savefig(output_path+"/umap.png")
    


def main():
    plot()

if __name__ == '__main__':
    main()

