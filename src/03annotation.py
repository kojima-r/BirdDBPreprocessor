import os
import glob
import pickle
import pandas as pd
import click

@click.command()
@click.option('--input_df',     default='song_df.pkl')
@click.option('--input_path',   default='./data2')
@click.option('--output_df',    default='song_df2.pkl')
@click.option('--output_csv',   default='song_df2.csv')
@click.option('--output_label', default='label.csv')
def sep_wav(input_df, input_path, output_df, output_csv, output_label):
    print("[LOAD]",input_df)
    with open(input_df, 'rb') as fp:
        song_df=pickle.load(fp)

    out_data_keys=["bird","WavTime","WavLoc","NumNote","species","WaveFileName"]
    out_data={k:[] for k in out_data_keys} 
    out_data["SepWaveFileName"]=[]
    out_data["label"]=[]
    out_data["begin"]=[]
    out_data["end"]=[]
    for i,row in song_df.iterrows():
        name,_=os.path.splitext(row["WaveFileName"])
        for filename in glob.glob(input_path+"/"+name+".*.wav"):
            for key in out_data_keys:
                out_data[key].append(row[key])
            print(filename)
            out_data["SepWaveFileName"].append(filename)
            
            temp1,_=os.path.splitext(filename)
            temp2,l=os.path.splitext(temp1)
            l=l[1:]
            temp3,span=os.path.splitext(temp2)
            span=span[1:]
            arr=span.split("-")
            
            label=row["species"]+"-"+l
            #label=row["species"]

            out_data["label"].append(label)
            out_data["begin"].append(arr[0])
            out_data["end"].append(arr[1])
    ### 
    out_data["y"]=[]
    all_data_y=out_data["label"]
    import collections
    mapping=collections.Counter(all_data_y)
    mapping_list=sorted([(v,k) for k,v in mapping.items()],reverse=True)
    cnt=0
    out_mapping={}
    for v,k in mapping_list:
        if v>=20:
            print(cnt,k,":",v)
            out_mapping[k]=cnt
            cnt+=1
    for label in out_data["label"]:
        if label in out_mapping:
            out_data["y"].append(out_mapping[label])
        else:
            out_data["y"].append(-1)
    ###
    ###
    print("[SAVE]",output_csv)
    df=pd.DataFrame(out_data)
    df.to_csv(output_csv)
    print("[SAVE]",output_df)
    with open(output_df, 'wb') as fp:
        pickle.dump(df,fp)


    o=[[v,k] for k,v in out_mapping.items()]
    print("[SAVE]",output_label)
    fp=open(output_label,"w")
    for v in sorted(o):
        fp.write(",".join(map(str,v)))
        fp.write("\n")

def main():
    sep_wav()

if __name__ == '__main__':
    main()
