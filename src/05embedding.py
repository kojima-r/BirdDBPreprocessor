from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import umap
import trimap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
import sklearn
import sklearn.metrics

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import numpy as np
import click
import os

def make_sliding_window(X,Y,S,window,limit_length):
    out_x=[]
    out_y=[]
    for i in range(X.shape[0]):
        s=S[i]
        y=Y[i]
        if limit_length is not None and limit_length<s:
            s=limit_length
        x=X[i,:s,:]
        dest=[]
        for j in range(s-window):
            dest.append(np.reshape(x[j:j+window,:],(-1,)))
            out_y.append(y)
        if len(dest)>0:
            out_x.append(np.array(dest))
    out_x=np.concatenate(out_x,axis=0)
    out_y=np.array(out_y)
    return out_x,out_y

def plot_scatter(embedding,Y,filename,title):
    plt.scatter(embedding[:,0],embedding[:,1],c=Y,cmap=cm.gist_rainbow,alpha=0.3,marker=".")
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)

@click.command()
@click.option('--input_path', default='./data3')
@click.option('--resample', type=int, default=None)
@click.option('--output_path', default='./')
@click.option('--feature', default='mel')
@click.option('--method', default='umap')
@click.option('--limit_length', type=int, default=None)
@click.option('--n_max_epoch', type=int, default=30)
@click.option('--b', type=float, default=0.895)
def plot(input_path, resample, output_path, feature,method,limit_length,n_max_epoch,b):
    x=np.load(input_path+"/data_x."+feature+".npy")
    y=np.load(input_path+"/data_y."+feature+".npy")
    print("X (input file):",x.shape)

    ###
    if len(x.shape)==3:
        print("... generating sliding window")
        s=np.load(input_path+"/data_s."+feature+".npy")
        x,y=make_sliding_window(x,y,s,window=10,limit_length=limit_length)
        print("X (sliding window):",x.shape)

    if resample is not None:
        print("... resampling")
        idx=list(range(x.shape[0]))
        np.random.shuffle(idx)
        idx=idx[:resample]
        X=x[idx,:]
        Y=y[idx]
    else:
        X=x
        Y=y
    
    print("... preprocess: normalization and PCA")
    preprocess_start_time = time.time()
    X=normalize(X)
    if X.shape[1]>20:
        prep_model=PCA(n_components=20)
        X=prep_model.fit_transform(X)
    preprocess_interval = time.time() - preprocess_start_time

    ##
    """
    print("... saving prepprocessed data")
    filename_x=output_path+"/"+feature+"_x.npy"
    filename_y=output_path+"/"+feature+"_y.npy"
    np.save(filename_x,X)
    np.save(filename_y,Y)
    """
    ##
    print("... embedding")
    embedding_start_time = time.time()
    if method=="song":
        import song
        model = song.song_.SONG(n_max_epoch=n_max_epoch,b=b)
        model.fit(X, Y)
        embedding=model.raw_embeddings[:,:]
    else:
        if method=="tsne":
            model = TSNE(n_components=2, random_state=42)
        elif method=="trimap":
            model = trimap.TRIMAP(n_iters=500)
        else:
            model = umap.UMAP()
        embedding = model.fit_transform(X)
    embedding_interval = time.time() - embedding_start_time

    print("... plotting embedded points")
    os.makedirs(output_path,exist_ok=True)
    np.save(output_path+"/"+method+"."+feature+".npy",embedding)
    print(method,"time:",embedding_interval)
    
    title=method
    filename=output_path+"/"+method+"."+feature+".png"
    plot_scatter(embedding,Y,filename,title)
    
    print("... evaluation")
    classifier=KNeighborsClassifier(n_neighbors=10)
    pred_y=cross_val_predict(classifier,embedding,Y, cv=5)
    acc=sklearn.metrics.accuracy_score(Y,pred_y)
    print("accuracy:",acc)

    with open(output_path+"/"+method+"."+feature+".txt","w") as fp:
        fp.write("Preprocess time\t{}\n".format(preprocess_interval))
        fp.write("Embedding time\t{}\n".format(embedding_interval))
        fp.write("Accuracy\t{}\n".format(acc))

def main():
    plot()

if __name__ == '__main__':
    main()

