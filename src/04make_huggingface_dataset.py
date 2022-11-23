import datasets

import os
import glob
import pickle
import pandas as pd
import librosa
import numpy as np
import click

import shutil

# huggingface-cli login
path="./dataset01/"
@click.command()
@click.option('--input_df',     default='song_df2.pkl')
@click.option('--output_path',   default=None)
@click.option('--limit_length', type=int, default=None)
def make_dataset(input_df,output_path,limit_length):
    with open(input_df, 'rb') as fp:
        song_df=pickle.load(fp)
    all_data=[]
    for i,row in song_df.iterrows():
        filename=row["SepWaveFileName"]
        length=int(row["end"])-int(row["begin"])
        if length>=44100*0.1:
            y=int(row["y"])
            if y>=0:
                all_data.append((filename,y,length,i))
    
    os.makedirs(path+"data/",exist_ok=True)
    with open(path+"data/metadata.csv","w") as fp:
        fp.write(",".join(["file_name","length","label"]))
        fp.write("\n")
        for el in all_data:
            filename=el[0]
            name=os.path.basename(filename)
            shutil.copyfile(filename, path+"/data/"+name)
            fp.write(",".join(map(str,[name,el[1],el[2]])))
            fp.write("\n")
    
    dataset = datasets.load_dataset(path, data_dir="data")
    print(dataset["train"][0])
    dataset.push_to_hub("kojima-r/birddb_small")
    
#audio_dataset = Dataset.from_dict({"audio": ["path/to/audio_1", "path/to/audio_2", ..., "path/to/audio_n"]}).cast_column("audio", Audio())
#audio_dataset[0]["audio"]
#
def main():
    make_dataset()

if __name__ == '__main__':
    main()

