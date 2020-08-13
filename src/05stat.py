
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
@click.option('--feature', default='mel')
@click.option('--method', default='umap')
def stat(input_path, resample, output_path, feature,method):
    #x=np.load(input_path+"/data_x."+feature+".npy")
    #y=np.load(input_path+"/data_y."+feature+".npy")
    s=np.load(input_path+"/data_s."+feature+".npy")
    plt.hist(s, bins=100)
    plt.savefig(output_path+"/bar."+feature+".png")
 
    
def main():
    stat()

if __name__ == '__main__':
    main()

